#!/usr/bin/env python

import pickle
import os.path
import sys
import generatorUtils as utils
from pprint import pprint
from tqdm import tqdm
from engine import Engine
from engineTempered import EngineTempered
from engineGuidedInference import EngineGuided
from src.datasets.citizenship_labels import CitizenshipLabels

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import getpass
USER = getpass.getuser()

GRAMMAR_DIR = 'src/rubricsampling/grammars/citizenship13'
EXP_DIR = f'/home/{USER}/generative-grading/experiments/citizenship13_100k/2019-04-12--01_26_27'

class RubricSampler(object):
    def create_data_loader(self, dataset, n=None):
        if n is None:
            n = len(dataset)

        out = []
        raw_progs = []
        for i in range(n):
            prog = dataset.raw_inputs[i]
            if prog == 'religious freedom':
                out.append(dataset[i])
                raw_progs.append(prog)
                break

        return out, raw_progs


    # Function: Run
    # -------------
    # This function compiles and renders samples
    # from the Rubric Sample
    def run(self):
        inf_e = EngineGuided(GRAMMAR_DIR, EXP_DIR)

        dataset = CitizenshipLabels(13, split='valid', vocab=inf_e.agent.train_dataset.vocab)
        # dataset = CitizenshipLabels(13, split='valid')

        N = 50

        data, raw_prgs = self.create_data_loader(dataset)
        data_loader = DataLoader(data, batch_size=1, shuffle=False)

        tqdm_batch = tqdm(data_loader, total=N)

        # inf_e = EngineTempered(GRAMMAR_DIR)

        time_data = []
        uniq_progs = set()
        failed = []

        num_all = 0
        num_correct = 0
        for i, data_list in enumerate(tqdm_batch):

            program_args = (data_list[0], data_list[2])
            label = data_list[3]
            program = raw_prgs[i]

            import pdb; pdb.set_trace()

            corr = self.infer_matches(inf_e, program, program_args, label)

            if corr:
                num_correct += 1

            num_all += 1
        # pprint(failed)
        print(f'Accuracy  = {num_correct/num_all}')


    def infer_matches(self, inf_e, program, program_args, label, n_lim=4):
        all_progs = []
        for i in range(n_lim):
            new_prog, new_choices = self.guided_sample(inf_e, program_args)
            all_progs.append(new_prog)
            # import pdb; pdb.set_trace()

            # print(program)
            # print(new_prog)
            # print(label)
            # pprint(new_choices)
            # input()
            # print()

            return int(new_choices['correctStrategy']) == label.item()



    #####################
    # Private Helpers
    #####################

    def guided_sample(self, inf_e,  program_args):
        # something that will crash if accessed without setting
        initAssignments = 1000000 * torch.ones(1, inf_e.model.num_nodes)
        program, labels, decisions, rvOrder, rvAssignments_pred = inf_e.renderProgram(program_args, initAssignments)
        # program, labels, decisions, rvOrder, rvAssignments_pred = inf_e.renderProgram()

        # print(rvAssignments[0][rvOrders[0][:rvOrders_lengths[0]]])
        # print(rvAssignments_pred[0][rvOrders[0][:rvOrders_lengths[0]]])
        # input()
        return program, decisions

    def sample(self, e):
        program, labels, decisions, _, _ = e.renderProgram()
        return program, decisions


if __name__ == '__main__':
    RubricSampler().run()
