#!/usr/bin/env python

r"""Job of this file is to just load each set of sharded, 
files and divy them into TRAIN VAL and TEST sets."""

import os
import pickle
import numpy as np
from glob import glob

from makeTieredData import HEAD, BODY, TAIL

TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1

TRAIN = 0
VAL = 1
TEST =  2


def make_splits(raw_dir):
    train_dir = os.path.join(raw_dir, 'train')
    val_dir = os.path.join(raw_dir, 'val')
    test_dir = os.path.join(raw_dir, 'test')

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)

    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    tiers = np.zeros(3)
    _tiers = get_tier_stats(raw_dir)
    tiers[:len(_tiers)] += _tiers
    tiers = tiers.astype(np.int)

    train_tiers, val_tiers, test_tiers = get_split_tier_stats(tiers)
    
    tier_splits_HEAD = [TRAIN] * train_tiers[HEAD] + [VAL] * val_tiers[HEAD] + [TEST] * test_tiers[HEAD]
    tier_splits_BODY = [TRAIN] * train_tiers[BODY] + [VAL] * val_tiers[BODY] + [TEST] * test_tiers[BODY]
    tier_splits_TAIL = [TRAIN] * train_tiers[TAIL] + [VAL] * val_tiers[TAIL] + [TEST] * test_tiers[TAIL]

    tier_splits_HEAD = np.array(tier_splits_HEAD)
    tier_splits_BODY = np.array(tier_splits_BODY)
    tier_splits_TAIL = np.array(tier_splits_TAIL)
    
    np.random.shuffle(tier_splits_HEAD)
    np.random.shuffle(tier_splits_BODY)
    np.random.shuffle(tier_splits_TAIL)

    counts_paths = sorted(glob(os.path.join(raw_dir, 'counts_shard_*.pkl')))
    tiers_paths = sorted(glob(os.path.join(raw_dir, 'tiers_shard_*.pkl')))
    labels_paths = sorted(glob(os.path.join(raw_dir, 'labels_shard_*.pkl')))
    rvOrder_paths = sorted(glob(os.path.join(raw_dir, 'rvOrder_shard_*.pkl')))

    assert len(counts_paths) == len(tiers_paths)
    assert len(labels_paths) == len(rvOrder_paths)
    assert len(counts_paths) == len(labels_paths)

    n_shards = len(counts_paths)
    shard_size = get_shard_size(counts_paths[0])

    countMap_TRAIN, rvOrderMap_TRAIN, labelMap_TRAIN, tierMap_TRAIN = {}, {}, {}, {}
    countMap_VAL, rvOrderMap_VAL, labelMap_VAL, tierMap_VAL = {}, {}, {}, {}
    countMap_TEST, rvOrderMap_TEST, labelMap_TEST, tierMap_TEST = {}, {}, {}, {}
    shardNum_TRAIN, shardNum_VAL, shardNum_TEST = 0, 0, 0
    seenTierNum_HEAD, seenTierNum_BODY, seenTierNum_TAIL = 0, 0, 0
    
    for i in range(n_shards):

        with open(counts_paths[i], 'rb') as f:
            counts_dict_i = pickle.load(f)
        
        with open(tiers_paths[i], 'rb') as f:
            tiers_dict_i = pickle.load(f)

        with open(labels_paths[i], 'rb') as f:
            labels_dict_i = pickle.load(f)
        
        with open(rvOrder_paths[i], 'rb') as f:
            rvOrder_dict_i = pickle.load(f)

        for program in counts_dict_i.keys():
            counts = counts_dict_i[program]
            tier = tiers_dict_i[program]
            labels = labels_dict_i[program]
            rvOrder = rvOrder_dict_i[program]

            if tier == HEAD:
                split_assignment = tier_splits_HEAD[seenTierNum_HEAD]
                seenTierNum_HEAD += 1
            elif tier == BODY:
                split_assignment = tier_splits_BODY[seenTierNum_BODY]
                seenTierNum_BODY += 1
            elif tier == TAIL:
                split_assignment = tier_splits_TAIL[seenTierNum_TAIL]
                seenTierNum_TAIL += 1
            else:
                raise Exception('tier {} not recognized.'.format(tier))

            if split_assignment == TRAIN:
                countMap_TRAIN[program] = counts
                labelMap_TRAIN[program] = labels
                rvOrderMap_TRAIN[program] = rvOrder
                tierMap_TRAIN[program] = tier
            elif split_assignment == VAL:
                countMap_VAL[program] = counts
                labelMap_VAL[program] = labels
                rvOrderMap_VAL[program] = rvOrder
                tierMap_VAL[program] = tier
            elif split_assignment == TEST:
                countMap_TEST[program] = counts
                labelMap_TEST[program] = labels
                rvOrderMap_TEST[program] = rvOrder
                tierMap_TEST[program] = tier
            else:
                raise Exception('split_assignment {} not recognized.'.format(split_assignment))

            if len(countMap_TRAIN) == shard_size:
                print('Saving New TRAIN Shard {}'.format(shardNum_TRAIN))
                save_shard(train_dir, countMap_TRAIN, labelMap_TRAIN, rvOrderMap_TRAIN, tierMap_TRAIN, shardNum_TRAIN)
                shardNum_TRAIN += 1
                countMap_TRAIN = {}
                labelMap_TRAIN = {}
                rvOrderMap_TRAIN = {}
                tierMap_TRAIN = {}

            if len(countMap_VAL) == shard_size:
                print('Saving New VAL Shard {}'.format(shardNum_VAL))
                save_shard(val_dir, countMap_VAL, labelMap_VAL, rvOrderMap_VAL, tierMap_VAL, shardNum_VAL)
                shardNum_VAL += 1
                countMap_VAL = {}
                labelMap_VAL = {}
                rvOrderMap_VAL = {}
                tierMap_VAL = {}

            if len(countMap_TEST) == shard_size:
                print('Saving New TEST Shard {}'.format(shardNum_TEST))
                save_shard(test_dir, countMap_TEST, labelMap_TEST, rvOrderMap_TEST, tierMap_TEST, shardNum_TEST)
                shardNum_TEST += 1
                countMap_TEST = {}
                labelMap_TEST = {}
                rvOrderMap_TEST = {}
                tierMap_TEST = {}

    if len(countMap_TRAIN) > 0:
        print('Saving New TRAIN Shard {}'.format(shardNum_TRAIN))
        save_shard(train_dir, countMap_TRAIN, labelMap_TRAIN, rvOrderMap_TRAIN, tierMap_TRAIN, shardNum_TRAIN)

    if len(countMap_VAL) > 0:
        print('Saving New VAL Shard {}'.format(shardNum_VAL))
        save_shard(val_dir, countMap_VAL, labelMap_VAL, rvOrderMap_VAL, tierMap_VAL, shardNum_VAL)

    if len(countMap_TEST) > 0:
        print('Saving New TEST Shard {}'.format(shardNum_TEST))
        save_shard(test_dir, countMap_TEST, labelMap_TEST, rvOrderMap_TEST, tierMap_TEST, shardNum_TEST)


def get_tier_stats(raw_dir):
    tiers = []
    tiers_paths = sorted(glob(os.path.join(raw_dir, 'tiers_shard_*.pkl')))
    
    for tiers_path in tiers_paths:
        with open(tiers_path, 'rb') as f:
            tiers_dict_i = pickle.load(f)
        tiers.extend(list(tiers_dict_i.values()))
    
    tiers = np.array(tiers).astype(np.int)
    
    return np.bincount(tiers)


def get_split_tier_stats(tiers):
    train_tiers = np.floor(TRAIN_PERC * tiers).astype(np.int)
    val_tiers = np.floor(VAL_PERC * tiers).astype(np.int)
    test_tiers = tiers - (train_tiers + val_tiers) 

    sanity_tiers = train_tiers + val_tiers + test_tiers
    assert np.sum(sanity_tiers != tiers) == 0

    return train_tiers, val_tiers, test_tiers


def get_shard_size(path):
    with open(path, 'rb') as f:
        shard_dict = pickle.load(f)
    
    return len(shard_dict)


def save_shard(out_dir, shard_countMap, shard_labelMap, shard_rvOrderMap, shard_tierMap, shardNum):
    shard_countMapName = os.path.join(out_dir, 'counts_shard_{}.pkl'.format(shardNum))
    shard_labelMapName = os.path.join(out_dir, 'labels_shard_{}.pkl'.format(shardNum))
    shard_rvOrderMapName = os.path.join(out_dir, 'rvOrder_shard_{}.pkl'.format(shardNum))
    shard_tierMapName = os.path.join(out_dir, 'tiers_shard_{}.pkl'.format(shardNum))

    pickle.dump(shard_countMap, open(shard_countMapName, 'wb'))
    pickle.dump(shard_labelMap, open(shard_labelMapName, 'wb'))
    pickle.dump(shard_rvOrderMap, open(shard_rvOrderMapName, 'wb'))
    pickle.dump(shard_tierMap, open(shard_tierMapName, 'wb'))


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'raw_dir',
        type=str,
        help='where to load raw data')
    args = arg_parser.parse_args()

    make_splits(args.raw_dir)    
