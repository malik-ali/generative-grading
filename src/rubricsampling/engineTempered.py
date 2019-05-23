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
import generatorUtils as gu
from engine import Engine

from tqdm import tqdm


SAMPLE_STANDARD = 'standard'
SAMPLE_UNIFORM = 'uniform'
SAMPLE_TEMPERED = 'tempered'

VALID_CHOICES = [SAMPLE_STANDARD, SAMPLE_UNIFORM, SAMPLE_TEMPERED]

class EngineTempered(Engine):
    def __init__(self, grammar_dir, choice_style=SAMPLE_STANDARD, reward=1, discount=0.9):
        super().__init__(grammar_dir)
        self.data_temper = defaultdict(Counter)
        self.explore_rewards = dict()
        self.reward = reward
        self.discount = discount

        if choice_style not in VALID_CHOICES:
            raise ValueError('choice_style must be one of {}. Received: {}'.format(VALID_CHOICES, choice_style))

        self.choice_style = choice_style

    def reset(self):
        self.state = {}
        self.choices = {}
        self.rubric = {}
        self.renderOrder = []
        self.rvChoiceOrder = []
        self.log_prob = 0.

    def burn(self, N, show_progress=False):
        tqdm_iter = tqdm(range(N)) if show_progress else range(N)
        for _ in tqdm_iter:
            self.renderProgram(freeze=False)

    def renderProgram(self, freeze=False, return_prob=False): 
        while True:
            self.reset()
            try:
                program = self.render('Program')
            except BaseException as e:
                if 'Choice name' in str(e):
                    continue
                else:
                    raise e

            program = gu.fixWhitespace(program)
            rubricItems = self.getRubricItems()
            choices = self.choices

            if self._isCodeOrg:
                program, rubricItems, choices, code = self.processCodeOrgProgram(
                    program, rubricItems, choices)
                self.rvChoiceOrder.append('doesNotGetNesting')

            if self.choice_style == SAMPLE_TEMPERED and not freeze:
                self.temperDistribution(self.choices, self.renderOrder)

            if return_prob:
               return program, rubricItems, choices, self.rvChoiceOrder, self.data_temper, self.log_prob
            else: 
                return program, rubricItems, choices, self.rvChoiceOrder, self.data_temper

    def _normalise(self, ctr):
        tot = 1. * sum(ctr.values())
        ret = defaultdict(float)

        for key, cnt in ctr.items():
            ret[key] = cnt / tot

        return ret

    def getDataTemper(self):
        return {
            choice: self._normalise(val_freq) for choice, val_freq in self.data_temper.items()
        }

    def temperDistribution(self, data, renderOrder):
        curr_discount = 1
        reward = self.reward
        for choice_name in reversed(renderOrder):
            choice = data[choice_name]
            # print(choice_name, choice)
            # print(self.explore_rewards[choice_name])
            # print(type(choice))
            # print([x for x in self.explore_rewards[choice_name].keys()][-1] == choice)
            # print('='*50)
            self.explore_rewards[choice_name][choice] += reward * curr_discount
            curr_discount *= self.discount


    def addGlobalChoice(self, choice_name, val):
        """
            If choice_name starts with RUBRIC_ then it is a meaningful
            pedagogical choice
        """
        if choice_name in self.choices:
            raise ValueError('Key [{}] already in global data'.format(choice_name))

        self.renderOrder.append(choice_name)
        self.choices[choice_name] = val

    def _pick_rv(self, choice_name, values):
        if type(values) is list:
            # reduce uniform case to dictionary case
            values = {val: 1 for val in values}

        self.rvChoiceOrder.append(choice_name)

        tuples = [(v, p) for v, p in values.items()]

        # unpack list of pairs to pair of lists
        choices, ps = list(zip(*tuples))
        ps /= np.sum(ps)

        if self.choice_style == SAMPLE_UNIFORM:
            choice_idx = np.random.choice(range(len(choices)))

        elif self.choice_style == SAMPLE_STANDARD:
            choice_idx = np.random.choice(range(len(choices)), p=ps)

        elif self.choice_style == SAMPLE_TEMPERED:
            # np array of length |values| or 0 if choice not seen before
            if choice_name not in self.explore_rewards:
                self.explore_rewards[choice_name] = {choice: 0 for choice in choices}

            # TODO: sometime the choice_name is repeated in different non terminals if they are exclusive.
            # Think about if this is a problem
            rewards_dict = self.explore_rewards[choice_name]
            # print(rewards_dict)
            rewards = np.array([rewards_dict[choice] for choice in choices])

            new_ps = 1 / (1/ps + rewards)
            new_ps /= np.sum(new_ps)

            # randomly selecting index instead of from val
            # because numpy typecasts contents in array
            choice_idx = np.random.choice(range(len(choices)), p=new_ps)

        else:
            raise ValueError('Invalid choice type')

            
        choice = choices[choice_idx]        
        self.log_prob += np.log(ps[choice_idx])
        
        self.data_temper[choice_name][choice] += 1

        return choice


if __name__ == "__main__":
    e = EngineTempered('grammars/liftoff')
    prog, _, choices, rvOrder, _, log_p = e.renderProgram(return_prob=True)
    pprint(choices)
    pprint(rvOrder)
    print(log_p)
    print(prog)
