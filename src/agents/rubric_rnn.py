# NOTE: This is hopelessly DEPRECATED. DO NOT USE.

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import os

from tqdm import tqdm
from pprint import pprint

from src.agents.base import BaseAgent
from src.models.inference_net import DirectInferenceNet
from src.models.rnn_encoder import (
    ProgramEncoder, TokenCharProgramEncoder, 
    MegaProgramEncoder, RawAnonProgramEncoder)
from src.losses.bce import BinaryCrossEntropy
from src.datasets.rubric_samples import RubricSamples
from src.utils.metrics import AverageMeter, bc_accuracy, tier_bc_counts
from src.utils.misc import print_cuda_statistics
from src.utils.io_utils import save_json

from tensorboardX import SummaryWriter

class RubricRNN(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.train_dataset = RubricSamples( config['problem'],
                                            domain=config['domain'],
                                            sampling_strategy=config['sampling_strategy'],
                                            split='train',
                                            character_level=config['character_level'],
                                            include_anonymized=config['include_anonymized'])
        self.val_dataset = RubricSamples(   config['problem'],
                                            domain=config['domain'],
                                            sampling_strategy=config['sampling_strategy'],
                                            split='val',
                                            character_level=config['character_level'],
                                            include_anonymized=config['include_anonymized'])
        self.test_dataset = RubricSamples(  config['problem'],
                                            domain=config['domain'],
                                            sampling_strategy=config['sampling_strategy'],
                                            split='test',
                                            character_level=config['character_level'],
                                            include_anonymized=config['include_anonymized'])

        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset)
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)

        self._choose_device()

        rv_categories_dict = self.train_dataset.rv_info['num_categories']
        num_rv_nodes = len(rv_categories_dict)
        rv_domain = [rv_categories_dict[self.train_dataset.rv_info['i2w'][str(i)]]
                     for i in range(num_rv_nodes)]

        self.model = self._create_model(self.train_dataset.vocab_size,
                                        self.train_dataset.char_vocab_size,
                                        self.train_dataset.anon_vocab_size,
                                        self.train_dataset.anon_char_vocab_size,
                                        self.train_dataset.labels_size,
                                        max(self.train_dataset.lengths),
                                        max(self.train_dataset.char_lengths),
                                        max(self.train_dataset.anon_lengths),
                                        max(self.train_dataset.anon_char_lengths),
                                        num_rv_nodes, rv_domain,
                                        character_level=config['character_level'],
                                        include_anonymized=config['include_anonymized'],
                                        learnable_alpha=config['learnable_alpha'])

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        self.model = self.model.to(self.device)
    
        self.accuracies = []
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='RubricRNN')

        self.rv_domain = rv_domain
        self.num_rv_nodes = num_rv_nodes

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

    def _create_model(self, vocab_size, char_vocab_size, anon_vocab_size, anon_char_vocab_size, 
                      labels_size, max_seq_len, char_max_seq_len, anon_max_seq_len, anon_char_max_seq_len, 
                      num_rv, rv_domains, include_anonymized=False, character_level=False, 
                      learnable_alpha=False):
        
        if include_anonymized:
            if character_level:
                program_encoder = MegaProgramEncoder(vocab_size, char_vocab_size,
                                                     anon_vocab_size, anon_char_vocab_size,
                                                     labels_size, max_seq_len, char_max_seq_len,
                                                     anon_max_seq_len, anon_char_max_seq_len,
                                                     learnable_alpha=learnable_alpha, 
                                                     device=self.device, **self.config.encoder_kwargs)
            else:
                program_encoder = RawAnonProgramEncoder(vocab_size, anon_vocab_size,
                                                        labels_size, max_seq_len, anon_max_seq_len,
                                                        device=self.device, **self.config.encoder_kwargs)
        else:
            if character_level:
                program_encoder = TokenCharProgramEncoder(vocab_size, char_vocab_size,
                                                          labels_size, max_seq_len, char_max_seq_len,
                                                          learnable_alpha=learnable_alpha,
                                                          device=self.device, **self.config.encoder_kwargs)
            else:
                program_encoder = ProgramEncoder(vocab_size, labels_size, max_seq_len,
                                                 device=self.device, **self.config.encoder_kwargs)

        model = DirectInferenceNet(program_encoder, num_rv, rv_domains,
                                   hidden_size=self.config.inference_kwargs['hidden_size'],
                                   device=self.device)
        
        return model

    def run(self):
        """
        Program entry point
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()

    def _compute_loss(self, outputs_list, labels_list):
        assert len(outputs_list) == len(labels_list)
        n = len(outputs_list)
        loss = 0
        for i in range(n):
            loss_i = F.cross_entropy(outputs_list[i], labels_list[i].long())
            loss += loss_i

        return loss

    def _compute_accuracy(self, outputs_list, labels_list):
        with torch.no_grad():
            assert len(outputs_list) == len(labels_list)
            n = len(outputs_list)
            accuracies = np.zeros(n)
            for i in range(n):
                dist_i = F.softmax(outputs_list[i], dim=1)
                pred_i = torch.argmax(dist_i, dim=1)
                acc_i = np.mean(pred_i.float().cpu().numpy() == 
                                labels_list[i].float().cpu().numpy())
                accuracies[i] = acc_i

        return accuracies

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

        for data_list in tqdm_batch:
            program_args, rvAssignments = data_list[:-4], data_list[-4]
            batch_size = len(rvAssignments)

            for i in range(len(program_args)):
                program_args[i] = program_args[i].to(self.device)

            rvAssignments = rvAssignments.to(self.device)
            labels_list = [rvAssignments[:, i] for i in range(rvAssignments.size(1))]

            # reset optimiser gradients
            self.optim.zero_grad()

            # get outputs and predictions
            outputs_list = self.model(program_args)
            loss = self._compute_loss(outputs_list, labels_list)
            accuracies = self._compute_accuracy(outputs_list, labels_list)
            avg_acc = np.mean(accuracies)

            loss.backward()
            self.optim.step()

            epoch_loss.update(loss.item(), n=batch_size)
            epoch_acc.update(avg_acc, n=batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg, "Avg acc": epoch_acc.avg})

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val}, self.current_iteration)
            self.summary_writer.add_scalars("epoch/accuracy", {'accuracy': epoch_acc.val}, self.current_iteration)

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

        for data_list in tqdm_batch:
            program_args, rvAssignments = data_list[:-4], data_list[-4]
            batch_size = len(rvAssignments)

            for i in range(len(program_args)):
                program_args[i] = program_args[i].to(self.device)
            
            rvAssignments = rvAssignments.to(self.device)
            labels_list = [rvAssignments[:, i] for i in range(rvAssignments.size(1))]

            outputs_list = self.model(program_args)
            loss = self._compute_loss(outputs_list, labels_list)
            accuracies = self._compute_accuracy(outputs_list, labels_list)
            avg_acc = np.mean(accuracies)

            loss_meter.update(loss.item(), n=batch_size)
            accuracy_meter.update(avg_acc, n=batch_size)

            tqdm_batch.set_postfix({"{} Loss".format(name.capitalize()): loss_meter.avg,
                                    "Avg acc": accuracy_meter.avg})

            self.summary_writer.add_scalars("{}/loss".format(name), {'loss': loss_meter.val}, self.current_val_iteration)
            self.summary_writer.add_scalars("{}/accuracy".format(name), {'accuracy': accuracy_meter.val}, self.current_val_iteration)

            self.current_val_iteration +=  1

        tqdm_batch.close()

    def validate(self):
        self._test('validation', self.val_loader, self.val_len)

    def test(self):
        self._test('test', self.test_loader, self.test_len)
    
    def backup(self):
        save_json(self.accuracies, os.path.join(self.config.out_dir, 'accuracies.json'))
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

        save_json(self.accuracies, os.path.join(self.config.out_dir, 'accuracies.json'))
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
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
