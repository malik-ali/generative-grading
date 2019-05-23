
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import os

from tqdm import tqdm
from pprint import pprint

from src.agents.base import BaseAgent
from src.models.inference_net import AutoregressiveInferenceNet
from src.models.conv_encoder import ConvImageEncoder, ConvImageEncoder_Old, ImageEncoder
from src.models.rnn_encoder import (
    ProgramEncoder, TokenCharProgramEncoder,
    MegaProgramEncoder, RawAnonProgramEncoder)

from src.datasets.rubric_samples import RubricSamples
from src.datasets.pyramid import PyramidImages, PyramidGrammar
from src.datasets.scenegraphs import SceneGraphs
from src.utils.metrics import AverageMeter, EarlyStopping, bc_accuracy, tier_bc_counts
from src.utils.misc import print_cuda_statistics
from src.utils.io_utils import save_json

from collections import defaultdict

from tensorboardX import SummaryWriter


class PyramidConvNet(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self._load_datasets(config['domain'], config['grammar_data'])

        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)

        self._choose_device()

        self.model = self._create_model()
        self.disable_progressbar = not config['display_progress']

        self.save_model = config['save_model']

        # Only optimise parameters which should be updated 
        params_to_optimize = (param for param in self.model.parameters() if param.requires_grad == True)
        self.optim = torch.optim.Adam(params_to_optimize, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        self.model = self.model.to(self.device)

        self.accuracies = []
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='RubricRNN')

        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    def _load_datasets(self, domain, grammar_data):
        if grammar_data:
            self.train_dataset = PyramidGrammar(input_size=self.config.encoder_kwargs.input_size, split='train',
                                                knowledge_states=self.config.knowledge_states)
            self.test_dataset = PyramidGrammar(input_size=self.config.encoder_kwargs.input_size, split='test',
                                               knowledge_states=self.config.knowledge_states)
        else:
            self.train_dataset = PyramidImages(self.config['num_train'], input_size=self.config.encoder_kwargs.input_size, split='train',
                                               knowledge_states=self.config.knowledge_states)
            self.test_dataset = PyramidImages(None, input_size=self.config.encoder_kwargs.input_size, split='test',
                                              knowledge_states=self.config.knowledge_states)

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

    def _create_model(self):
        return ConvImageEncoder(self.train_dataset.labels_size, **self.config.encoder_kwargs)
        # return ConvImageEncoder_Old(self.train_dataset.labels_size, **self.config.encoder_kwargs)

    def run(self):
        """
        Program entry point
        """
        try:
            self.train()

        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            raise e

    def train(self):
        """
        Main training loop
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            self.validate()

            # self.test() -- dont do this during training... validation should be enough
            self.save_checkpoint()
            
            if self.early_stopping.stop:
                print('Early stopping...')
                return

    def train_one_epoch(self):
        """
        One epoch of training
        """
        num_batches = self.train_len // self.config.batch_size
        tqdm_batch = tqdm(self.train_loader, total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch), disable=self.disable_progressbar)

        # num_batches = self.overfit_debug_len // self.config.batch_size
        # tqdm_batch = tqdm(self.overfit_debug_loader, total=num_batches,
        #                   desc="[Epoch {}]".format(self.current_epoch))

        val_every = None if self.config['validations_per_epoch'] == 0 else max(num_batches // self.config['validations_per_epoch'], 1)
        self.model.train()

        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        for batch_i, data_list in enumerate(tqdm_batch):
            X, (counts, y) = data_list
            batch_size = len(y)

            X = X.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.long)

            scores = self.model(X)
            loss = F.cross_entropy(scores, y)
            ll = torch.softmax(scores, 1)
            preds = torch.argmax(ll, 1)

            accuracy = torch.sum(preds == y).float().cpu().numpy()/y.size(0)
            
            # reset optimiser gradients
            self.optim.zero_grad()

            loss.backward()
            
            self.optim.step()

            epoch_loss.update(loss.item(), n=batch_size)
            epoch_acc.update(accuracy, n=batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg, "Avg acc": epoch_acc.avg})

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val}, self.current_iteration)
            self.summary_writer.add_scalars("epoch/accuracy", {'accuracy': epoch_acc.val}, self.current_iteration)

            self.current_iteration += 1

            if val_every and (batch_i + 1) % val_every == 0:
                self.validate()
                self.model.train()  # put back in training mode

        tqdm_batch.close()

    def validate(self):
        self._test('validation', self.test_loader, self.test_len)

    def _test(self, name, loader, length):
        """
        Returns validation accuracy. Unlike training, we compute accuracy
        (1) per decision node and (2) sorted by HEAD, BODY, and TAIL.
        """
        num_batches = length // self.config.batch_size
        tqdm_batch = tqdm(loader, total=num_batches,
                          desc="[{}]".format(name.capitalize()), disable=self.disable_progressbar)

        # set the model in validation mode
        self.model.eval()
        loss_meter = AverageMeter()
        avg_acc_meter = AverageMeter()

        with torch.no_grad():
            for batch_i, data_list in enumerate(tqdm_batch):
                X, (counts, y) = data_list
                batch_size = len(y)

                X = X.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.long)

                scores = self.model(X)
                loss = F.cross_entropy(scores, y)
                ll = torch.softmax(scores, 1)
                preds = torch.argmax(ll, 1)
                accuracy = torch.sum(preds == y).float().cpu().numpy()/y.size(0)

                # write data and summaries
                loss_meter.update(loss.item(), n=batch_size)
                avg_acc_meter.update(accuracy, n=batch_size)

                tqdm_batch.set_postfix({"{} Loss".format(name.capitalize()): loss_meter.avg,
                                        "Avg acc": avg_acc_meter.avg})

                self.summary_writer.add_scalars("{}/loss".format(name), {'loss': loss_meter.val}, self.current_val_iteration)
                self.summary_writer.add_scalars("{}/accuracy".format(name), {'accuracy': avg_acc_meter.val}, self.current_val_iteration)

                self.current_val_iteration +=  1

        print('AVERAGE ACCURACY: {}'.format(avg_acc_meter.avg))
        tqdm_batch.close()


    def backup(self):
        """
        Backs up the model upon interrupt
        """
        save_json(self.accuracies, os.path.join(self.config.out_dir, 'accuracies.json'))
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Backs up the model upon interrupt
        """
        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

        save_json(self.accuracies, os.path.join(self.config.out_dir, 'accuracies.json'))
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param filename: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        if not self.save_model:
            return

        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'val_iteration': self.current_val_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'config': self.config,
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))

        # TODO: somehow do best model saving
        # If this is the best model, copy it to another file 'model_best.pth.tar'
        # if is_best:
        #     shutil.copyfile(self.config.checkpoint_dir + filename,
        #     self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        """
        Loads the latest checkpoint
        :param filename: name of the checkpoint file
        :return:
        """
        filename = os.path.join(self.config.checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.current_val_iteration = checkpoint['val_iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e
