#!/usr/bin/env python

"""Job of this file is to parse through the sharded files and assign
a tier (HEAD, BODY, TAIL) to every single element."""

import os
import pickle
import numpy as np
from glob import glob

# NOTE: make sure this is consistent with other files..
# TODO: or let us load from globals.py
HEAD = 0
BODY = 1
TAIL = 2

HEAD_COUNT = 20
TAIL_MIN_FREQ = 2


def get_rubric_tiers(raw_dir, head_count=20, tail_min_freq=2):
    counts_paths = glob(os.path.join(raw_dir, 'counts_shard_*.pkl'))
    n_shards = len(counts_paths)

    # do one path to collect statistics
    counts = []
    shard_ids = []  # keep track of count --> shard

    for i in range(n_shards):
        print('Reading Counts ({}/{})'.format(i + 1, n_shards))
        with open(counts_paths[i], 'rb') as f:
            counts_dict_i = pickle.load(f)
            counts_i = np.array(list(counts_dict_i.values()))
            counts.append(counts_i)

            shard_ids_i = np.ones(len(counts_i)) * i
            shard_ids.append(shard_ids_i)

    # we can def store this in memory as it is only an array of integers
    counts = np.concatenate(counts)
    shard_ids = np.concatenate(shard_ids)

    decreasing_order = np.argsort(counts)[::-1]
    # head = top <head_count> programs
    head_selection = decreasing_order[:head_count]
    # tail = programs with <= <tail_min_freq> appearances
    tail_selection = counts <= tail_min_freq
    # build dictionary from program to tier
    tiers = np.ones(len(counts)) * BODY
    tiers[head_selection] = HEAD
    tiers[tail_selection] = TAIL

    shard_tiers = []
    for i in range(n_shards):
        tiers_i = tiers[shard_ids == i]
        shard_tiers.append(tiers_i)
    
    for i in range(n_shards):
        counts_path = counts_paths[i]
        tiers_path = counts_path.replace('counts', 'tiers')
        
        with open(counts_path, 'rb') as f:
            counts_dict_i = pickle.load(f)
            programs_i = list(counts_dict_i.keys())
            tiers_i = list(shard_tiers[i])
            tier_dict_i = dict(zip(programs_i, tiers_i))

        print('Writing Tiers ({}/{})'.format(i + 1, n_shards))
        with open(tiers_path, 'wb') as f:
            pickle.dump(tier_dict_i, f)                


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'raw_dir',
        type=str,
        help='where to load raw data')
    arg_parser.add_argument(
        '--head-count',
        type=int,
        default=HEAD_COUNT,
        help='top K is considered the head [default: 20]')
    arg_parser.add_argument(
        '--tail-min-freq',
        type=int,
        default=TAIL_MIN_FREQ,
        help='everything with less than K counts is in tail [default: 2]')
    args = arg_parser.parse_args()

    get_rubric_tiers(
        args.raw_dir, 
        head_count=args.head_count, 
        tail_min_freq=args.tail_min_freq)
