import os
from os.path import dirname, basename, isfile
import glob
import inspect
import re
import copy
from engine import Engine
from base import ReusableDecision
from pprint import pprint

import json
import numpy as np

def get_rv_categories(values):
    if type(values) is list:
        # reduce uniform case to dictionary case
        values = {val: 1 for val in values}

    tuples = sorted([(v, p) for v, p in values.items()], key=lambda x: str(x[0]))
    choices, ps = list(zip(*tuples))
    ps /= np.sum(ps)

    return list(choices), list(ps)


def getReusableRandomVariables(nonterm):
    ids = set(nonterm._getValidIds())
    union_rvs = dict()

    # for varId in sorted(list(ids)):
    for varId in ids:
        nonterm._setParams({'id': varId})
        nonterm.registerChoices()
        rvs = nonterm._getRandomVariables()
        union_rvs.update(rvs)

    return union_rvs

def getNonTerminals(grammar_dir):
    e = Engine(grammar_dir)
    nonterminals, reusableNonterms = e._getNonTerminals(grammar_dir)

    all_rvs = dict()
    for name, nonterm in nonterminals.items():
        e.choices = {}
        if name in  ['Decompose', 'PermuteLines', 'PermuteDict']:
            print('## WARNING ##\t Skipping RV: {}'.format(name))
            continue

        # if nonterm is rusable decisions, get valid ids for it
        if name in reusableNonterms:
            rvs = getReusableRandomVariables(nonterm)
        else:
            nonterm.registerChoices()
            rvs = nonterm._getRandomVariables()

        for rv, vals in rvs.items():
            if rv in all_rvs:
                print(f'Warning: found re-used random variable: {rv}')
                if all_rvs[rv].keys() != vals.keys():
                    raise ValueError(f'Two RVs with same name and different values: [{vals.keys()}], [{all_rvs[rv].keys()}]')
        all_rvs.update(rvs)

    ret = {k: get_rv_categories(v)for k, v in all_rvs.items()}

    return ret


def addCodeOrgPostFunction(choices):
    choices['doesNotGetNesting'] = ([False, True], [0.95, 0.05])
    return choices


def main(out_dir, grammar_dir):
    out_file = os.path.join(out_dir, 'random_variables.json')
    choices = getNonTerminals(grammar_dir)

    if 'codeorg' in grammar_dir:
        choices = addCodeOrgPostFunction(choices)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)    

    pprint(choices)

    with open(out_file, 'w') as f:
        json.dump(choices, f)


if __name__ == "__main__":
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
    args = arg_parser.parse_args()
    grammar_dir = os.path.join('grammars', args.problem)
    main(args.out_dir, grammar_dir)
