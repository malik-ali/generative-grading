
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
from src.models.conv_encoder import ImageEncoder
from src.models.rnn_encoder import (
    ProgramEncoder, TokenCharProgramEncoder,
    MegaProgramEncoder, RawAnonProgramEncoder)

from src.datasets.rubric_samples import RubricSamples
from src.datasets.scenegraphs import SceneGraphs
from src.utils.metrics import AverageMeter, EarlyStopping, bc_accuracy, tier_bc_counts
from src.utils.misc import print_cuda_statistics
from src.utils.io_utils import save_json

from collections import defaultdict

from tensorboardX import SummaryWriter

class AutoregressiveRNN(BaseAgent):

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
        self.use_attention = config.inference_kwargs['num_attention_heads'] > 0
        self.num_attention_heads = config.inference_kwargs['num_attention_heads']

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

    def _load_datasets(self, domain):
        config = self.config

        if domain in {'education', 'postagging'}:

            self.train_dataset = RubricSamples( config['problem'],
                                                domain=domain,
                                                sampling_strategy=config['sampling_strategy'],
                                                split='train',
                                                character_level=config['character_level'],
                                                include_anonymized=config['include_anonymized'])
            self.val_dataset = RubricSamples(   config['problem'],
                                                domain=domain,
                                                sampling_strategy=config['sampling_strategy'],
                                                split='val',
                                                character_level=config['character_level'],
                                                include_anonymized=config['include_anonymized'])
            self.test_dataset = RubricSamples(  config['problem'],
                                                domain=domain,
                                                sampling_strategy=config['sampling_strategy'],
                                                split='test',
                                                character_level=config['character_level'],
                                                include_anonymized=config['include_anonymized'])        
        elif domain == 'scenegraph':
            self.train_dataset = SceneGraphs( config['problem'],
                                              sampling_strategy=config['sampling_strategy'],
                                              split='train')
            self.val_dataset = SceneGraphs(   config['problem'],
                                              sampling_strategy=config['sampling_strategy'],
                                              split='val')                                              
            self.test_dataset = SceneGraphs(  config['problem'],
                                              sampling_strategy=config['sampling_strategy'],
                                              split='test')                                                
        else:
            raise ValueError('Invalid problem domain: {}. Must be one of [education, postagging, scenegraph]'.format(domain))                                                

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

    def _create_program_encoder(self, vocab_size, char_vocab_size, anon_vocab_size, anon_char_vocab_size,
                      labels_size, max_seq_len, char_max_seq_len, anon_max_seq_len, anon_char_max_seq_len, 
                      include_anonymized=False, character_level=False, learnable_alpha=False):

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

        return program_encoder

    def _create_model(self, num_rv, rv_domains, problem_domain):

        if problem_domain in {'education', 'postagging'}:
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
        elif problem_domain == 'scenegraph':
            encoder = ImageEncoder(device=self.device, **self.config.encoder_kwargs)
        else:
            raise ValueError('Invalid problem domain: {}. Must be one of [education, postagging, scenegraph]'.format(problem_domain))

        model = AutoregressiveInferenceNet(encoder, num_rv, rv_domains,
                                           device=self.device,
                                           **self.config.inference_kwargs)

        return model

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

    def _createLabels(self, rvOrders, rvAssignments, rvOrders_lengths):
        '''
            - rvOrders:
                list of shape N x T where each row contains the (padded) render order for each data point.
                Each entry in columnt t is the index of the rv that was rendered at time t for this datapoint

            - rvAssignments:
                N x all_num_rvs tesnor containing (indexes of) final values of all rvs for each data point


            Outputs list of list of shape N x (T - 1) containing the true categorical
            label for each random variable.
        '''
        label_info = rvOrders[:, 1:]
        labels = []

        for n in range(label_info.shape[0]):
            true_len = rvOrders_lengths[n] - 1
            curr_order = label_info[n][:true_len]     # ignore padding terms
            vals = list(rvAssignments[n][curr_order])
            for t in range(len(label_info[n]) - true_len):
                vals.append(-1)

            labels.append(vals)

        return labels

    def _compute_tier_stats(self, output, labels, tiers, rvOrders_lengths, rvOrders):
        '''
            - outputs:
                list of list of tensors of shape num_batches x (T-1) x c_t

            - labels:
                list of list of shape N x (T - 1) containing the true categorical
                label for each random variable.

            - tiers:
                batch_size, each index is 0, 1, or 2 for HEAD, BODY or TAIL

            - rvOrders_lengths:
                true unpadded lengths of rvOrders
        '''
        batch_size = len(output)
        trace_len = len(output[0])

        tier_counts = np.zeros((3, self.num_rv_nodes + 1))
        tier_norms = np.zeros((3, self.num_rv_nodes + 1))

        for i in range(batch_size):
            tier_i = int(tiers[i].item())

            for j in range(trace_len):
                outputs_ij = output[i][j]
                labels_ij = labels[i][j]

                if outputs_ij is None:
                    continue

                outputs_ij = outputs_ij.cpu().detach().numpy()
                outputs_ij = np.argmax(outputs_ij)
                labels_ij = labels_ij.item()

                rvIndex = rvOrders[i][j]

                tier_counts[tier_i, rvIndex] += int(outputs_ij == labels_ij)
                tier_norms[tier_i, rvIndex] += 1

        return tier_counts, tier_norms

    def _compute_loss(self, output, labels_inp, rvOrders_lengths):
        '''
            - outputs:
                list of list of tensors of shape num_batches x (T-1) x c_t

            - labels:
                list of list of shape N x (T - 1) containing the true categorical
                label for each random variable.

            - rvOrders_lengths:
                true unpadded lengths of rvOrders
        '''
        len_batches = defaultdict(list)
        for n in range(len(output)):
            for t in range(rvOrders_lengths[n]):
                pred = output[n][t]
                true_label = labels_inp[n][t]
                
                if pred is None: 
                    continue

                len_batches[len(pred)].append( (pred, true_label))
    

        loss = 0
        num_correct = 0
        num_total = 0

        for l, group in len_batches.items():

            unzip = list(zip(*group))
            preds = unzip[0]
            labels = unzip[1]

            # Should be shape N x c
            preds = torch.stack(preds)

            # Should be shape n
            labels = torch.stack(labels)

            preds_npy = preds.cpu().detach().numpy()
            labels_npy = labels.cpu().detach().numpy()
            num_correct += np.sum(np.argmax(preds_npy, axis=1) == labels_npy)
            num_total += len(labels)

            # important to not reduce using mean... bc that weights things unevenly
            loss += F.cross_entropy(preds, labels.long(), reduction='sum')

        return loss / float(num_total), num_correct / float(num_total), num_total

    def _compute_frobenius_norm(self, alphas_list):
        """
        alphas_list: list with length <execution_trace_length>
                     which each element being batch_size x num_attention_heads
        """
        n = len(alphas_list)
        frob_norm = 0
        for i in range(n):
            alphas = alphas_list[i]
            I = torch.eye(self.num_attention_heads, device=self.device)
            I = I.repeat(alphas.size(0), 1, 1)
            alphas_t = torch.transpose(alphas, 1, 2).contiguous()
            frob_norm_i = torch.norm(torch.bmm(alphas, alphas_t) - I)
            frob_norm += frob_norm_i
        # average over the nodes in the trace
        frob_norm = frob_norm / float(n)

        return frob_norm

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
            program_args, rvAssignments, _, rvOrders, rvOrders_lengths = \
                data_list[:-4], data_list[-4], data_list[-3], data_list[-2], data_list[-1]

            for i in range(len(program_args)):
                program_args[i] = program_args[i].to(self.device)

            # N x all_num_rvs tesnor containing (indexes of) final values of all rvs for each data point
            rvAssignments = rvAssignments.to(self.device)

            # rvOrders is a list of shape N x T where each row contains the (padded) render order for each data point
            # Each entry in columnt t is the index of the rv that was rendered at time t for this datapoint
            rvOrders = rvOrders.to(self.device)
            rvOrders_lengths = rvOrders_lengths.to(self.device)

            # shape N x (T - 1)
            labels = self._createLabels(rvOrders, rvAssignments, rvOrders_lengths)

            # reset optimiser gradients
            self.optim.zero_grad()

            # outputs are list of list of tensors of shape num_batches x (T-1) x c_t
            output, alphas = self.model(rvOrders.long(), rvOrders_lengths.long(), rvAssignments.long(), program_args)

            loss, avg_acc, num_total = self._compute_loss(output, labels, rvOrders_lengths)

            if self.use_attention:
                assert len(alphas) > 0
                frob_loss = self._compute_frobenius_norm(alphas)
                loss = loss + frob_loss

            loss.backward()
            self.optim.step()

            epoch_loss.update(loss.item(), n=num_total)
            epoch_acc.update(avg_acc, n=num_total)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg, "Avg acc": epoch_acc.avg})

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val}, self.current_iteration)
            self.summary_writer.add_scalars("epoch/accuracy", {'accuracy': epoch_acc.val}, self.current_iteration)

            self.current_iteration += 1

            if val_every and (batch_i + 1) % val_every == 0:
                self.validate()
                self.model.train()  # put back in training mode

        tqdm_batch.close()

    def validate(self):
        self._test('validation', self.val_loader, self.val_len)

    def test(self):
        self._test('test', self.test_loader, self.test_len)

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

        tier_counts = np.zeros((3, self.num_rv_nodes + 1))
        tier_norm = np.zeros((3, self.num_rv_nodes + 1))

        for data_list in tqdm_batch:
            program_args, rvAssignments, tiers, rvOrders, rvOrders_lengths = \
                data_list[:-4], data_list[-4], data_list[-3], data_list[-2], data_list[-1]

            for i in range(len(program_args)):
                program_args[i] = program_args[i].to(self.device)

            # N x all_num_rvs tesnor containing (indexes of) final values of all rvs for each data point
            rvAssignments = rvAssignments.to(self.device)

            # rvOrders is a list of shape N x T where each row contains the (padded) render order for each data point
            # Each entry in columnt t is the index of the rv that was rendered at time t for this datapoint
            rvOrders = rvOrders.to(self.device)
            rvOrders_lengths = rvOrders_lengths.to(self.device)
            rvOrdersShifted = rvOrders[:, 1:]        # Shifted left by one to match the T-1 predictions made

            # shape N x (T - 1)
            labels = self._createLabels(rvOrders, rvAssignments, rvOrders_lengths)

            # outputs are list of list of tensors of shape num_batches x (T-1) x c_t
            output, alphas = self.model(rvOrders.long(), rvOrders_lengths.long(),
                                        rvAssignments.long(), program_args)

            loss, avg_acc, num_total = self._compute_loss(output, labels, rvOrders_lengths)
            tier_counts_, tier_norm_ = self._compute_tier_stats(output, labels, tiers, rvOrders_lengths, rvOrdersShifted)

            tier_counts += tier_counts_
            tier_norm += tier_norm_

            if self.use_attention:
                assert len(alphas) > 0
                frob_loss = self._compute_frobenius_norm(alphas)
                loss = loss + frob_loss

            # write data and summaries
            loss_meter.update(loss.item(), n=num_total)
            avg_acc_meter.update(avg_acc, n=num_total)

            tqdm_batch.set_postfix({"{} Loss".format(name.capitalize()): loss_meter.avg,
                                    "Avg acc": avg_acc_meter.avg})

            self.summary_writer.add_scalars("{}/loss".format(name), {'loss': loss_meter.val}, self.current_val_iteration)
            self.summary_writer.add_scalars("{}/accuracy".format(name), {'accuracy': avg_acc_meter.val}, self.current_val_iteration)

            self.current_val_iteration +=  1

        tier_accuracy = tier_counts / tier_norm
        head_acc = {'rv: {}'.format(self.train_dataset.rv_info['i2w'][str(idx)]): 
                    tier_accuracy[0, idx] for idx in range(self.num_rv_nodes + 1)}
        body_acc = {'rv: {}'.format(self.train_dataset.rv_info['i2w'][str(idx)]): 
                    tier_accuracy[1, idx] for idx in range(self.num_rv_nodes + 1)}
        tail_acc = {'rv: {}'.format(self.train_dataset.rv_info['i2w'][str(idx)]): 
                    tier_accuracy[2, idx] for idx in range(self.num_rv_nodes + 1)}

        acc_dict = {'head_acc': head_acc, 'body_acc': body_acc, 'tail_acc': tail_acc}
        self.accuracies.append(acc_dict)
        
        print('AVERAGE ACCURACY: {}'.format(avg_acc_meter.avg))


        self.early_stopping.update(loss_meter.avg)
        
        print('[HEAD] {} accuracy per RV ({} total): '.format(name.capitalize(), tier_norm[0, 1]))
        pprint(head_acc)

        print('[BODY] {} accuracy per RV ({} total): '.format(name.capitalize(), tier_norm[1, 1]))
        pprint(body_acc)

        print('[TAIL] {} accuracy per RV ({} total): '.format(name.capitalize(), tier_norm[2, 1]))
        pprint(tail_acc)

        tqdm_batch.close()

        return acc_dict 

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
