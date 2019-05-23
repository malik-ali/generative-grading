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

from copy import deepcopy

from src.rubricsampling.engine import Engine
from src.rubricsampling.engineTempered import EngineTempered
from src.rubricsampling.engineTempered import (
    SAMPLE_STANDARD, SAMPLE_UNIFORM, SAMPLE_TEMPERED)


class EngineConditioned(EngineTempered):
    def __init__(self, grammar_dir, fixed_data, choice_style=SAMPLE_STANDARD, reward=1, discount=0.9):
        super().__init__(grammar_dir, choice_style=choice_style, reward=reward, discount=discount)
        # fixed data will force the global data to always have these values
        # for these fixed keys
        self.fixed_data = deepcopy(fixed_data)

    def addGlobalChoice(self, choice_name, val):
        """
            If choice_name starts with RUBRIC_ then it is a meaningful
            pedagogical choice
        """
        if choice_name in self.choices and choice_name not in self.fixed_data:
            raise ValueError('Key [{}] already in global data'.format(choice_name))

        self.renderOrder.append(choice_name)
        self.choices[choice_name] = val
        self.choices.update(self.fixed_data)

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

            if choice_name not in self.fixed_data:
                self.explore_rewards[choice_name][choice] += reward * curr_discount

            curr_discount *= self.discount

    def _pick_rv(self, choice_name, values):
        if choice_name in self.fixed_data:
            return self.fixed_data[choice_name]
        else:
            return super(EngineConditioned, self)._pick_rv(choice_name, values)

