r"""NOTE: this is hardcoded to citizenship laels right now..."""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import os

from tqdm import tqdm
from pprint import pprint

from src.agents.base import BaseAgent
from src.models.rnn_encoder import ProgramEncoder
from src.losses.bce import BinaryCrossEntropy
from src.datasets.citizenship_labels import CitizenshipLabels
from src.utils.metrics import AverageMeter, bc_accuracy, tier_bc_counts
from src.utils.misc import print_cuda_statistics
from src.utils.io_utils import save_json

from tensorboardX import SummaryWriter


class SupervisedRNN(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.train_dataset = CitizenshipLabels(config['problem_id'], split='train',
                                               vocab=None)
        self.val_dataset = CitizenshipLabels(config['problem_id'], split='valid',
                                             vocab=self.train_dataset.vocab)
        self.test_dataset = CitizenshipLabels(config['problem_id'], split='test', 
                                              vocab=self.train_dataset.vocab)

        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset)
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)

        self._choose_device()

        self.model = self._create_model(self.train_dataset.vocab_size,
                                        self.train_dataset.max_length)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        self.model = self.model.to(self.device)
    
        self.accuracies = []
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='RubricRNN')

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def _create_model(self, vocab_size, max_seq_len):
        return ProgramEncoder(vocab_size, 1, max_seq_len, device=self.device, 
                              **self.config.encoder_kwargs)

    def run(self):
        """
        Program entry point
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()

    def _compute_loss(self, outputs, labels):
        return F.binary_cross_entropy_with_logits(outputs, labels.float())

    def _compute_accuracy(self, outputs, labels, reduce=True):
        with torch.no_grad():
            dist = torch.sigmoid(outputs)
            outputs = torch.round(dist)
            if reduce:
                acc = np.mean(outputs.float().cpu().numpy() == 
                              labels.float().cpu().numpy())
            else:
                acc = (outputs.float().cpu().numpy() == 
                       labels.float().cpu().numpy())

        return acc

    def train(self):
        """
        Main training loop
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()

    def train_one_epoch(self):
        """
        One epoch of training
        """
        num_batches = self.train_len // self.config.batch_size
        tqdm_batch = tqdm(self.train_loader, total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch))

        # num_batches = self.overfit_debug_len // self.config.batch_size
        # tqdm_batch = tqdm(self.overfit_debug_loader, total=num_batches,
        #                   desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()

        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        for seq_src, _, seq_len, label, _ in tqdm_batch:
            batch_size = len(seq_src)
        
            seq_src = seq_src.to(self.device)
            seq_len = seq_len.to(self.device)
            label = label.to(self.device)

            # reset optimiser gradients
            self.optim.zero_grad()

            # get outputs and predictions
            output = self.model(seq_src, seq_len)
            loss = self._compute_loss(output, label)
            acc = self._compute_accuracy(output, label)

            loss.backward()
            self.optim.step()

            epoch_loss.update(loss.item(), n=batch_size)
            epoch_acc.update(acc, n=batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg, "Avg acc": epoch_acc.avg})

            self.summary_writer.add_scalars(
                "epoch/loss", 
                {'loss': epoch_loss.val}, self.current_iteration)
            self.summary_writer.add_scalars(
                "epoch/accuracy", 
                {'accuracy': epoch_acc.val}, self.current_iteration)

            self.current_iteration += 1

        tqdm_batch.close()

    def _test(self, name, loader, length):
        num_batches = length // self.config.batch_size
        tqdm_batch = tqdm(loader, total=num_batches,
                          desc="[{}]".format(name.capitalize()))

        # set the model in validation mode
        self.model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        head_accuracy_meter = AverageMeter()
        tail_accuracy_meter = AverageMeter()

        for seq_src, _, seq_len, label, tier in tqdm_batch:
            batch_size = len(seq_src)
            
            seq_src = seq_src.to(self.device)
            seq_len = seq_len.to(self.device)
            label = label.to(self.device)
            
            output = self.model(seq_src, seq_len)
            loss = self._compute_loss(output, label)
            unrolled_acc = self._compute_accuracy(output, label, reduce=False)
            unrolled_acc = unrolled_acc[:, 0].astype(np.int)
            acc = np.mean(unrolled_acc)
            
            head_acc = np.mean(unrolled_acc[tier.numpy() == 1])
            tail_acc = np.mean(unrolled_acc[tier.numpy() == 0])

            loss_meter.update(loss.item(), n=batch_size)
            accuracy_meter.update(acc, n=batch_size)
            head_accuracy_meter.update(head_acc, sum(tier.numpy() == 1))
            tail_accuracy_meter.update(tail_acc, sum(tier.numpy() == 0))

            tqdm_batch.set_postfix({"{} Loss".format(name.capitalize()): loss_meter.avg,
                                    "Avg acc": accuracy_meter.avg})

            self.summary_writer.add_scalars(
                "{}/loss".format(name), 
                {'loss': loss_meter.val}, self.current_val_iteration)
            self.summary_writer.add_scalars(
                "{}/accuracy".format(name), 
                {'accuracy': accuracy_meter.val}, self.current_val_iteration)
            self.summary_writer.add_scalars(
                "{}/headAccuracy".format(name),
                {'headAccuracy': head_accuracy_meter.val}, self.current_val_iteration)
            self.summary_writer.add_scalars(
                "{}/tailAccuracy".format(name),
                {'tailAccuracy': tail_accuracy_meter.val}, self.current_val_iteration)

            self.current_val_iteration +=  1

        tqdm_batch.close()

    def validate(self):
        self._test('validation', self.val_loader, self.val_len)

    def test(self):
        self._test('test', self.test_loader, self.test_len)
    
    def backup(self):
        save_json(self.accuracies, os.path.join(self.config.out_dir, 'accuracies.json'))
        self.summary_writer.export_scalars_to_json(
            os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

        save_json(self.accuracies, os.path.join(self.config.out_dir, 'accuracies.json'))
        self.summary_writer.export_scalars_to_json(
            os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'val_iteration': self.current_val_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))
