# This engine loads a pretrained model and uses that to predict the 
# posterior distribution over random variable choices given the output
# of a program.
  
import os
from os.path import dirname, basename, isfile
import glob
import inspect
import re
import copy
import numpy as np
from pprint import pprint

from collections import deque, Counter, defaultdict
from string import Formatter

from copy import deepcopy

import torch
import torch.nn.functional as F

import generatorUtils as gu
from engine import Engine
from engineTempered import EngineTempered
from src.agents.autoregressive_rnn import AutoregressiveRNN
from src.utils.setup import load_config

class EngineGuided(Engine):
    def __init__(self, grammar_dir, exp_dir, strategy='sample'):
        super().__init__(grammar_dir)
        assert strategy in ['sample', 'map']
        self.strategy = strategy
        self.exp_dir = exp_dir
        self._load_model(exp_dir)
        self.dataset = self.agent.val_dataset
        self.all_rvs = self.dataset.rv_info['values']

    def _load_model(self, exp_dir):
        config = load_config(os.path.join(exp_dir, 'config.json'))
        # config['cuda'] = False
        self.agent = AutoregressiveRNN(config)
        self.agent.load_checkpoint('checkpoint.pth.tar')
        self.model = self.agent.model
        self.model.eval()
        self.config = config
        # self.agent.validate()

    def _choicename_to_idx(self, choice_name):
        return self.dataset.rv_info['w2i'][choice_name]

    def _preds_to_value(self, node_ip1, preds, strategy='map'):
        if strategy == 'map':
            idx = np.argmax(preds)
        elif strategy == 'sample':
            idx = np.random.choice(len(preds), p=preds)
        # just for testing
        elif strategy == 'adversarial':
            idx = np.argmin(preds)
        else:
            raise ValueError('Invalid rv value picking strategy')

        # print(node_ip1)
        rv_name = self.dataset.rv_info['i2w'][str(node_ip1)]
        vals = self.all_rvs[rv_name][0]
        return vals[idx]

    def _rnn_step(self, node_ip1):
        rvOrders_lengths = [torch.from_numpy(np.array(len(self.hidden_store))).to(self.agent.device)]
        # TODO: how do I deal with attention lengths
        node_ip1 = torch.from_numpy(np.array([node_ip1])).to(self.agent.device)
        # print(self.node_i)
        with torch.no_grad():
            output_i, h0, alphas_i = self.model.step(self.node_i, node_ip1, self.program_emb,
                                        self.h0, self.hidden_store, self.rvAssignments.long().to(self.agent.device), rvOrders_lengths)
            next_preds = F.softmax(output_i[0], dim=0).cpu().numpy()
            self.h0 = h0
        return next_preds

    def _choicename_in_model(self, choice_name):
        return choice_name in self.dataset.rv_info['w2i']

    def reset(self):
        self.state = {}
        self.choices = {}
        self.rubric = {}

        self.renderOrder = []
        self.rvChoiceOrder = []
        self.preds = []

    def renderProgram(self, program_args, rvAssignments):
        self.reset()

        self.h0 = self.model.init_rnn_hiddens(1)
        self.hidden_store = []

        program_args = [p.to(self.agent.device) for p in program_args]
        self.program_emb = self.model.program_encoder(*program_args, return_hiddens=True)
        self.rvAssignments = rvAssignments.clone()
        self.node_i = None

        program = self.render('Program')
        program = gu.fixWhitespace(program)
        rubricItems = self.getRubricItems()

        return program, rubricItems, self.choices, self.rvChoiceOrder, self.rvAssignments

    def _pick_rv(self, choice_name, values):
        # if choice name not valid for model or...
        #   if first choice. TODO: is this right?
        self.rvChoiceOrder.append(choice_name)
        if not self._choicename_in_model(choice_name) or self.node_i is None:
            val = super(EngineGuided, self)._pick_rv(choice_name, values)
            
            tuples = [(v, p) for v, p in values.items()]
            # unpack list of pairs to pair of lists
            choices, ps = list(zip(*tuples))
            ps /= np.sum(ps)            
            display_preds = list(zip(values.keys(), [round(p, 2) for p in ps]))

            # print('(Sample based choice: choice={}, values={}, chosen: {})'.format(choice_name, display_preds, val))
            return val
        
        else:
            node_ip1 = self._choicename_to_idx(choice_name)
            preds = self._rnn_step(node_ip1)

            self.preds.append(preds)

            if len(preds) != len(values):
                raise ValueError(f'pred dim [{len(preds)}] != num values [{len(values)}]')

            val = self._preds_to_value(node_ip1, preds, strategy=self.strategy)
            display_preds = list(zip(values.keys(), [round(p, 2) for p in preds]))
            # print('Making choice: choice={}, values={}, chosen: {}'.format(choice_name, display_preds, val))
            return val

    def addGlobalChoice(self, choice_name, val):
        if choice_name in self.choices:
            raise ValueError('Key [{}] already in global choices'.format(choice_name))

        # print('Made choice: [{}] = {}'.format(choice_name, val))
        self.renderOrder.append(choice_name)
        self.choices[choice_name] = val

        if self._choicename_in_model(choice_name):
            node_idx = self._choicename_to_idx(choice_name)
            val_idx = self.all_rvs[choice_name][0].index(val)
            self.rvAssignments[0][node_idx] = val_idx
            self.node_i = torch.from_numpy(np.array([node_idx])).to(self.agent.device)
