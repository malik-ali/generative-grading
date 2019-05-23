#!/usr/bin/env python

import pickle
import os.path
import sys
from pprint import pprint
from tqdm import tqdm
from engine import Engine
from engineTempered import EngineTempered
from engineGuidedInference import EngineGuided

import torch
from torch.utils.data import DataLoader

from src.utils.io_utils import save_json
import src.utils.paths as paths

import matplotlib.pyplot as plt

from globals import PAD_IDX, UNK_IDX, START_IDX, END_IDX

class RecoverySampler:
    def create_data_loader(self, dataset, n):
        out = []
        raw_progs = []
        for i in range(n):
            out.append(dataset[i])
            raw_progs.append(dataset.get_raw_program(i))

        return out, raw_progs
    
    def dropout_words(self, program_args, dropout_proba, sos_idx=START_IDX, eos_idx=END_IDX, 
                      pad_idx=PAD_IDX, unk_idx=UNK_IDX):
        '''
            Corrupts the program. With probability error_rate, this
            will replace a token with the UNK token. Only corrupts the full
            program, not the anonymised or char level programs
        '''
        # first arg is featurised full program
        seq = program_args[0].clone()

        # randomly replace with unknown tokens
        prob = torch.rand(seq.size())

        # don't dropout important tokens by forcing
        # their keep probability to be 1
        prob[((seq.cpu().data == sos_idx) | \
              (seq.cpu().data == eos_idx) | \
              (seq.cpu().data == pad_idx))] = 1

        mask_seq = seq.clone()
        mask_seq[(prob < dropout_proba).to(seq.device)] = unk_idx
        
        # leave rest of args untouched
        return (mask_seq, *program_args[1:])


    # Function: Run
    # -------------
    # This function compiles and renders samples
    # from the Rubric Sample
    def run(self, results_dir, problem, exp_dir, error_rate=0):
        grammar_dir = paths.grammar_path(problem)
        inf_e = EngineGuided(grammar_dir, exp_dir)
        
        N = len(inf_e.dataset)

        n_lim = 100

        data, raw_prgs = self.create_data_loader(inf_e.dataset, N)
        data_loader = DataLoader(data, batch_size=1)
        tqdm_batch = tqdm(data_loader, total=N)

        time_data = []
        failed = []
        
        for i, data_list in enumerate(tqdm_batch):
            program_args, _, _, _, _ = data_list[:-4], data_list[-4], data_list[-3], data_list[-2], data_list[-1]
            program = raw_prgs[i]

            if error_rate > 0:
                program_args = self.dropout_words(program_args, error_rate)

            num_steps = self.infer_matches(inf_e, program, program_args, n_lim=n_lim)

            if num_steps < 0:
                failed.append(program)
            
            time_data.append(num_steps)


        error_rate_str = str(error_rate).replace('.', '_')
        # TODO: make sure failed programs are safe to save
        out_json = dict(failed_progs=failed, time_data=time_data, error_rate=error_rate)
        out_file = os.path.join(results_dir, f'data_err_{error_rate_str}.json')
        save_json(out_json, out_file)

        plt.hist(time_data, bins=50, color='purple')
        plt.title(f'Num samples to recover validation program (N={N}, max_steps={n_lim}, error_rate={error_rate})')
        plt.xlabel('Number of steps (neg. = not recovered)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(results_dir, f'validation_recovery_err_{error_rate_str}.png'))
        

    def infer_matches(self, inf_e, program, program_args, n_lim=400):
        all_progs = []
        for i in range(n_lim):
            new_prog, new_choices = self.guided_sample(inf_e, program_args)
            all_progs.append(new_prog)
            if new_prog == program:
                return i + 1

        return -n_lim // 2
        

    #####################
    # Private Helpers
    #####################
    def guided_sample(self, inf_e,  program_args):
        # something that will crash if accessed without setting
        initAssignments = 1000000 * torch.ones(1, inf_e.model.num_nodes)
        program, labels, decisions, rvOrder, rvAssignments_pred = inf_e.renderProgram(program_args, initAssignments)

        return program, decisions


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()    
    arg_parser.add_argument(
        'results_dir',
        type=str,
        help='where to save results')
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to evaluate e.g. liftoff')
    arg_parser.add_argument(
        'exp_dir',
        type=str,
        help='directory of model experiment to use for evaluation')
    arg_parser.add_argument(
        '--error_rate',
        type=float,
        default=0.0,
        help='Error rate to corrupt program with. With prob error_rate, a token will be changed to UNK')
    args = arg_parser.parse_args()

    if not 0 <= args.error_rate <= 1:
        raise ValueError(f'Error rate must be a float between 0 and 1. Input: {args.error_rate}')

    RecoverySampler().run(args.results_dir, args.problem, args.exp_dir, error_rate=args.error_rate)
