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

from tqdm import tqdm

# -- tempering options --
# N = 10000             # total numberofsamples
N_BURN_IN = 0         # set to 0 if we want samples right away
N_CONVERGE = 0        # set to 0 if we want to take every sample
N_SNAPSHOT = None     # set to None if we want to sample every time
                      # each snapshot will contain N / N_SNAPSHOT samples
SHARD_SIZE = 1000000  # if we sample too much, can't store in one file

class RubricSampler(object):
    # Function: Run
    # -------------
    # This function compiles and renders samples
    # from the Rubric Sample

    def run(self, n, out_dir, grammar_dir, sample_strategy):
        if sample_strategy == SAMPLE_STANDARD:
            self._run_standard(n, out_dir, grammar_dir)
        elif sample_strategy == SAMPLE_UNIFORM:
            self._run_uniform(n, out_dir, grammar_dir)
        elif sample_strategy == SAMPLE_TEMPERED:
            self._run_tempered(n, out_dir, grammar_dir)
        else:
            raise Exception('sampling_stratgy {} is not recognized'.format(sample_strategy))

    def _run_base(self, n, out_dir, grammar_dir, sample_strategy=SAMPLE_STANDARD):
        assert sample_strategy in [SAMPLE_STANDARD, SAMPLE_UNIFORM]

        e = EngineTempered(grammar_dir, choice_style=sample_strategy)
        os.makedirs(out_dir, exist_ok=True)

        countMap, rvOrderMap, labelMap = {}, {}, {}
        shardNum, shardedUnique = 0, 0

        hll_counter = hyperloglog.HyperLogLog(0.01)

        for _ in tqdm(range(n)):
            program, choices, rvOrder = self.sample(e)

            labelMap[program] = choices
            rvOrderMap[program] = rvOrder

            if not program in countMap:
                countMap[program] = 0
            countMap[program] += 1

            hll_counter.add(program)

            if len(labelMap) == SHARD_SIZE:
                # save shard and reset data dictionaries
                self.save_shard(out_dir, countMap, labelMap, rvOrderMap, shardNum)

                shardNum += 1
                shardedUnique += SHARD_SIZE
                countMap = {}
                labelMap = {}
                rvOrderMap = {}

        print('In N={}, generated #uniq ~ {}'.format(n, len(hll_counter)))

        if len(countMap) > 0:  # save remaining samples
            self.save_shard(out_dir, countMap, labelMap, rvOrderMap, shardNum)
   
    def _run_standard(self, n, out_dir, grammar_dir):
        self._run_base(n, out_dir, grammar_dir, SAMPLE_STANDARD)
    
    def _run_uniform(self, n, out_dir, grammar_dir):
        self._run_base(n, out_dir, grammar_dir, SAMPLE_UNIFORM)

    def _run_tempered(self, n, out_dir, grammar_dir, n_burn_in=N_BURN_IN, 
                      n_converge=N_CONVERGE, n_snapshot=N_SNAPSHOT):
        
        e = EngineTempered(grammar_dir, choice_style=SAMPLE_TEMPERED)
        os.makedirs(out_dir, exist_ok=True)

        if n_snapshot is None:
            n_snapshot = n

        hll_counter = hyperloglog.HyperLogLog(0.01)

        countMap, rvOrderMap, labelMap = {}, {}, {}
        shardNum, shardedUnique = 0, 0

        # this is number of samples we will take per snapshot
        n_per_snap = int(n // n_snapshot)

        # burn X number of samples before we start taking any samples
        if n_burn_in > 0:
            print('Burning {} samples'.format(n_burn_in))
        e.burn(n_burn_in, show_progress=True) 

        pbar = tqdm(total=n)
        for _ in range(n_snapshot):
            for _ in range(n_per_snap):
                # we froze the graph here... no additional tempering
                # -- but dont freeze graph if  we save every example
                program, choices, rvOrder = self.sample(e, freeze=n_converge > 0)

                labelMap[program] = choices
                rvOrderMap[program] = rvOrder

                if not program in countMap:
                    countMap[program] = 0
                countMap[program] += 1
                
                hll_counter.add(program)

                if len(labelMap) == SHARD_SIZE:
                    # save shard and reset data dictionaries
                    self.save_shard(out_dir, countMap, labelMap, rvOrderMap, shardNum)

                    shardNum += 1
                    shardedUnique += SHARD_SIZE
                    countMap = {}
                    labelMap = {}
                    rvOrderMap = {}

                pbar.update()

            # wait n steps before saving any more samples
            if n_converge > 0:
                print('Burning {} samples'.format(n_converge))
            e.burn(n_converge)

        print('In N={}, generated #uniq ~ {}'.format(n, len(hll_counter)))
        #pickle.dump(data, open('sample_efficiency_temp_anneal.pkl', 'wb'))

        if len(countMap) > 0:  # save remaining samples
            self.save_shard(out_dir, countMap, labelMap, rvOrderMap, shardNum)

        pbar.close()

    #####################
    # Private Helpers
    #####################
    def save_shard(self, out_dir, shard_countMap, shard_labelMap, shard_rvOrderMap, shardNum):
        shard_countMapName = os.path.join(out_dir, 'counts_shard_{}.pkl'.format(shardNum))
        shard_labelMapName = os.path.join(out_dir, 'labels_shard_{}.pkl'.format(shardNum))
        shard_rvOrderMapName = os.path.join(out_dir, 'rvOrder_shard_{}.pkl'.format(shardNum))

        pickle.dump(shard_countMap, open(shard_countMapName, 'wb'))
        pickle.dump(shard_labelMap, open(shard_labelMapName, 'wb'))
        pickle.dump(shard_rvOrderMap, open(shard_rvOrderMapName, 'wb'))

    def sample(self, e, freeze=False):
        program, _, decisions, rvOrder, _ = e.renderProgram(freeze=freeze)
        return program, decisions, rvOrder


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'out_dir',
        type=str,
        help='where to dump raw data')
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
    arg_parser.add_argument(
        '--sample-strategy',
        default=SAMPLE_TEMPERED,
        help='sampling strategy (standard|uniform|tempered) [default: tempered]')
    args = arg_parser.parse_args()

    grammar_dir = os.path.join('grammars', args.problem)
    RubricSampler().run(args.num_samples, args.out_dir, grammar_dir, args.sample_strategy)
