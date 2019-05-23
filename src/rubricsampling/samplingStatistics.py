#!/usr/bin/env python

"""Job of this file is just to generate a bunch of data and store it into 
files that can be loaded into memory."""

import pickle
import os.path
import sys
import generatorUtils as utils

import hyperloglog

from engine import Engine
from engineTempered import EngineTempered
from engineTempered import SAMPLE_STANDARD, SAMPLE_UNIFORM, SAMPLE_TEMPERED

from src.utils.io_utils import save_json

import matplotlib.pyplot as plt

from tqdm import tqdm

class RubricSampler(object):
    # Function: Run
    # -------------
    # This function compiles and renders samples
    # from the Rubric Sample

    def run(self, n, out_dir, grammar_dir, problem):
        std_data = self._run_base(n, grammar_dir, SAMPLE_STANDARD)
        unif_data = self._run_base(n, grammar_dir, SAMPLE_UNIFORM)

        data = dict(standard=std_data, uniform=unif_data, tempered=dict())   

        for r in [0.01, 0.1, 1.]:
            for d in [0.3, 0.6, 0.9]:
                data['tempered'][f'{r}_{d}'] = self._run_base(n, grammar_dir, SAMPLE_TEMPERED, reward=r, discount=d)



        save_json(data, os.path.join(out_dir, f'{problem}.json'))  


    def _run_base(self, n, grammar_dir, sample_strategy=SAMPLE_STANDARD, reward=1, discount=0.9):
        assert sample_strategy in [SAMPLE_STANDARD, SAMPLE_UNIFORM, SAMPLE_TEMPERED]

        print('Running strategy: ', sample_strategy)
        e = EngineTempered(grammar_dir, choice_style=sample_strategy, reward=reward, discount=discount)

        countMap = {}
        N = 0
        N1 = 0      # keeps track of objects seen only once 
        hll_counter = hyperloglog.HyperLogLog(0.01)

        uniques = []
        goodTuringEstimates = []
        log_prod_prob = []
        newly_seen = []
        

        for _ in tqdm(range(n)):
            program, log_prob = self.sample(e)

            if not program in countMap:
                countMap[program] = 0
                newly_seen.append(True)
            else:
                newly_seen.append(False)

            countMap[program] += 1

            if countMap[program] == 1:
                N1 += 1
            elif countMap[program] == 2:
                N1 -= 1
            
            hll_counter.add(program)

            N += 1
            uniques.append(len(hll_counter))
            goodTuringEstimates.append(N1/N)
            log_prod_prob.append(log_prob)


        print('In N={}, generated #uniq ~ {}'.format(n, len(hll_counter)))

        return dict(goodTuringEstimates=goodTuringEstimates, numUniques=uniques, log_prod_probs=log_prod_prob, newly_seen=newly_seen)
        


    #####################
    # Private Helpers
    #####################

    def sample(self, e, freeze=False):
        program, _, _, _, _, log_prob = e.renderProgram(freeze=freeze, return_prob=True)
        return program, log_prob


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'out_dir',
        type=str,
        help='results dir')    
    arg_parser.add_argument(
        'problem',
        type=str,
        help='liftoff|drawCircles')
    arg_parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of programs to sample'
    )        
    
    args = arg_parser.parse_args()

    grammar_dir = os.path.join('grammars', args.problem)
    RubricSampler().run(args.num_samples, args.out_dir, grammar_dir, args.problem)
