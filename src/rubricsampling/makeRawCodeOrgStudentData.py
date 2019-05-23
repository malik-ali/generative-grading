r"""Preprocess code.org data into a form that we can handle."""

import os
import pickle
import getpass
import numpy as np

USER = getpass.getuser()
DATA_ROOT = '/mnt/fs5/{}/generative-grading/data/real/education/'.format(USER)
ZIPF_CLASS = {'head': 0, 'body': 1, 'tail': 2}


def load_data(problem_name):
    problem_id = int(problem_name.split('_')[0].replace('codeorg', ''))
    root = '/mnt/fs5/{}/datasets/code_org_public_dataset_v1/data/p{}'.format(USER, problem_id)
    programs_file = os.path.join(root, 'programs-{}.pickle'.format(problem_id))
    counts_file = os.path.join(root, 'countMap-{}.pickle'.format(problem_id))
    annotations_file = os.path.join(root, 'annotations-{}.pickle'.format(problem_id))

    with open(programs_file, 'rb') as fp:
        programs = pickle.load(fp)

    with open(counts_file, 'rb') as fp:
        counts = pickle.load(fp)

    program2count = {}
    for _id in programs.keys():
        program2count[programs[_id]] = counts[_id]

    ranks = build_rank_map_from_count_map(program2count)
    zipfs = build_zipf_map(program2count, ranks)

    with open(annotations_file, 'rb') as fp:
        annotations = pickle.load(fp)

    countMap = {}
    labelMap = {}
    zipfMap = {}  # program to head/body/tail

    for _id, labels in annotations.items():
        program = programs[_id]
        countMap[program] = counts[_id]
        labelMap[program] = annotations[_id]
        zipfMap[program] = zipfs[program]

    return countMap, labelMap, zipfMap


def build_rank_map_from_count_map(count_map):
    programs = np.array(list(count_map.keys()))
    counts = np.array(list(count_map.values()))
    # NOTE: need to reverse b/c most common = rank 0
    ranks = np.argsort(counts)[::-1]
    programs = np.array(programs)
    programs = programs[ranks]
    return dict(zip(programs, np.arange(len(programs))))


def build_zipf_map(count_map, rank_map):
    programs = np.array(list(count_map.keys()))
    counts = np.array(list(count_map.values()))
    ranks = np.array(list(rank_map.values()))

    # by default, everything is a body
    zipf_classes = np.ones(len(programs)) * ZIPF_CLASS['body']
    # every with less than 3 counts is in the tail
    zipf_classes[counts < 3] = ZIPF_CLASS['tail']
    # every in top 20 is in the head
    zipf_classes[ranks <= 20] = ZIPF_CLASS['head']

    return dict(zip(programs, zipf_classes))


def save_data(out_dir, counts, labels, zipfs):
    with open(os.path.join(out_dir, 'counts.pkl'), 'wb') as fp:
        pickle.dump(counts, fp)

    with open(os.path.join(out_dir, 'labels.pkl'), 'wb') as fp:
        pickle.dump(labels, fp)

    with open(os.path.join(out_dir, 'zipfs.pkl'), 'wb') as fp:
        pickle.dump(zipfs, fp)


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
        help='codeorg1|codeorg9')
    args = arg_parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    counts, labels, zipfs = load_data(args.problem)
    save_data(args.out_dir, counts, labels, zipfs)
