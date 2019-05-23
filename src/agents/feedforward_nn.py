import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import os

from tqdm import tqdm
from pprint import pprint

from src.agents.base import BaseAgent
from src.models.inference_net import FeedforwardInferenceNet
from src.models.rnn_encoder import (
    ProgramEncoder, TokenCharProgramEncoder,
    MegaProgramEncoder, RawAnonProgramEncoder)

from src.agents.autoregressive_rnn import AutoregressiveRNN

from src.datasets.rubric_samples import RubricSamples
from src.utils.metrics import AverageMeter, EarlyStopping, bc_accuracy, tier_bc_counts
from src.utils.misc import print_cuda_statistics
from src.utils.io_utils import save_json

from collections import defaultdict

from tensorboardX import SummaryWriter


class FeedforwardNN(AutoregressiveRNN):
    def __init__(self, config):
        super().__init__(config)

        self._load_datasets(config['domain'])

        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset)
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)

        self._choose_device()

        rv_categories_dict = self.train_dataset.rv_info['num_categories']
        num_rv_nodes = len(rv_categories_dict)
        rv_domain = [rv_categories_dict[self.train_dataset.rv_info['i2w'][str(i)]]
                     for i in range(num_rv_nodes)]

        self.model = self._create_model(num_rv_nodes, rv_domain, config['domain'])

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

        self.rv_domain = rv_domain
        self.num_rv_nodes = num_rv_nodes

    def _create_model(self, num_rv, rv_domains, problem_domain):
        encoder = self._create_program_encoder(
            self.train_dataset.vocab_size,
            self.train_dataset.char_vocab_size,
            self.train_dataset.anon_vocab_size,
            self.train_dataset.anon_char_vocab_size,
            self.train_dataset.labels_size,
            max(self.train_dataset.lengths),
            max(self.train_dataset.char_lengths),
            max(self.train_dataset.anon_lengths),
            max(self.train_dataset.anon_char_lengths),
            character_level=self.config['character_level'],
            include_anonymized=self.config['include_anonymized'],
            learnable_alpha=self.config['learnable_alpha'])

        model = FeedforwardInferenceNet(encoder, num_rv, rv_domains,
                                        device=self.device,
                                        **self.config.inference_kwargs)

        return model

    def _createLabels(self, rvOrders, rvAssignments, rvOrders_lengths):
        label_info = rvOrders[:, 1:]
        labels = []

        for n in range(label_info.shape[0]):
            true_len = rvOrders_lengths[n] - 1
            curr_order = label_info[n][:true_len]
            vals = rvAssignments[n][curr_order]
            labels.append(vals)

        return labels

    def _chooseOutputs(self, rvOutputList, rvOrders, rvOrders_lengths):
        label_info = rvOrders[:, 1:]
        outputs = [] 

        for n in range(label_info.shape[0]):
            true_len = rvOrders_lengths[n] - 1
            curr_order = label_info[n][:true_len]
            curr_order_npy = curr_order.cpu().numpy()

            outputs_n = [rvOutputList[i][n] for i in curr_order_npy]
            outputs.append(outputs_n)

        return outputs

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.batch_size
        tqdm_batch = tqdm(self.train_loader, total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch), disable=self.disable_progressbar)
        
        val_every = max(num_batches // self.config['validations_per_epoch'], 1)
        self.model.train()

        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        for batch_i, data_list in enumerate(tqdm_batch):
            program_args, rvAssignments, _, rvOrders, rvOrders_lengths = \
                data_list[:-4], data_list[-4], data_list[-3], data_list[-2], data_list[-1]

            for i in range(len(program_args)):
                program_args[i] = program_args[i].to(self.device)

            rvAssignments = rvAssignments.to(self.device)
            rvOrders = rvOrders.to(self.device)
            rvOrders_lengths = rvOrders_lengths.to(self.device)

            labels = self._createLabels(rvOrders, rvAssignments, rvOrders_lengths)

            self.optim.zero_grad()

            outputs = self.model(program_args)
            outputs = self._chooseOutputs(outputs, rvOrders, rvOrders_lengths)

            loss, avg_acc, num_total = self._compute_loss(outputs, labels, rvOrders_lengths - 1)
            loss.backward()
            
            self.optim.step()

            epoch_loss.update(loss.item(), n=num_total)
            epoch_acc.update(avg_acc, n=num_total)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg, "Avg acc": epoch_acc.avg})

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val}, self.current_iteration)
            self.summary_writer.add_scalars("epoch/accuracy", {'accuracy': epoch_acc.val}, self.current_iteration)

            self.current_iteration += 1

            if (batch_i + 1) % val_every == 0:
                self.validate()
                self.model.train()  # put back in training mode

        tqdm_batch.close()

    def _test(self, name, loader, length):
        num_batches = length // self.config.batch_size
        tqdm_batch = tqdm(loader, total=num_batches,
                          desc="[{}]".format(name.capitalize()), disable=self.disable_progressbar)

        # set the model in validation mode
        self.model.eval()
        loss_meter = AverageMeter()
        avg_acc_meter = AverageMeter()

        tier_counts = np.zeros((3, self.num_rv_nodes + 1))
        tier_norm = np.zeros((3, self.num_rv_nodes + 1))

        for data_list in tqdm_batch:
            program_args, rvAssignments, tiers, rvOrders, rvOrders_lengths = \
                data_list[:-4], data_list[-4], data_list[-3], data_list[-2], data_list[-1]

            for i in range(len(program_args)):
                program_args[i] = program_args[i].to(self.device)

            rvAssignments = rvAssignments.to(self.device)
            rvOrders = rvOrders.to(self.device)
            rvOrders_lengths = rvOrders_lengths.to(self.device)
            rvOrdersShifted = rvOrders[:, 1:]

            labels = self._createLabels(rvOrders, rvAssignments, rvOrders_lengths)
            outputs = self.model(program_args)
            outputs = self._chooseOutputs(outputs, rvOrders, rvOrders_lengths)

            loss, avg_acc, num_total = self._compute_loss(outputs, labels, rvOrders_lengths - 1)
            # tier_counts_, tier_norm_ = self._compute_tier_stats(outputs, labels, tiers, rvOrders_lengths, rvOrdersShifted)

            # tier_counts += tier_counts_
            # tier_norm += tier_norm_

            # write data and summaries
            loss_meter.update(loss.item(), n=num_total)
            avg_acc_meter.update(avg_acc, n=num_total)

            tqdm_batch.set_postfix({"{} Loss".format(name.capitalize()): loss_meter.avg,
                                    "Avg acc": avg_acc_meter.avg})

            self.summary_writer.add_scalars("{}/loss".format(name), {'loss': loss_meter.val}, self.current_val_iteration)
            self.summary_writer.add_scalars("{}/accuracy".format(name), {'accuracy': avg_acc_meter.val}, self.current_val_iteration)

            self.current_val_iteration +=  1

        # tier_accuracy = tier_counts / tier_norm
        # head_acc = {'rv: {}'.format(self.train_dataset.rv_info['i2w'][str(idx)]): 
        #             tier_accuracy[0, idx] for idx in range(self.num_rv_nodes + 1)}
        # body_acc = {'rv: {}'.format(self.train_dataset.rv_info['i2w'][str(idx)]): 
        #             tier_accuracy[1, idx] for idx in range(self.num_rv_nodes + 1)}
        # tail_acc = {'rv: {}'.format(self.train_dataset.rv_info['i2w'][str(idx)]): 
        #             tier_accuracy[2, idx] for idx in range(self.num_rv_nodes + 1)}

        # acc_dict = {'head_acc': head_acc, 'body_acc': body_acc, 'tail_acc': tail_acc}
        # self.accuracies.append(acc_dict)
        
        print('AVERAGE ACCURACY: {}'.format(avg_acc_meter.avg))

        self.early_stopping.update(loss_meter.avg)
        
        # print('[HEAD] {} accuracy per RV ({} total): '.format(name.capitalize(), tier_norm[0, 1]))
        # pprint(head_acc)

        # print('[BODY] {} accuracy per RV ({} total): '.format(name.capitalize(), tier_norm[1, 1]))
        # pprint(body_acc)

        # print('[TAIL] {} accuracy per RV ({} total): '.format(name.capitalize(), tier_norm[2, 1]))
        # pprint(tail_acc)

        tqdm_batch.close()

        # return acc_dict 
