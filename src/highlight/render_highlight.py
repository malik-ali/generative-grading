import os
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from src.datasets.student_programs import StudentPrograms
from src.rubricsampling.engineHighlight import InferenceNNHighlight
from src.rubricsampling.generatorUtils import fixWhitespace
from scripts.process_data import strip_comments
from src.utils.io_utils import load_json, save_json

from pprint import pprint
import re


def conv_sample(curr_node):
    nonterminal = curr_node['name']
    rvs = curr_node['rvs']
    children = curr_node['children']
    template = curr_node['template']
    
    if not isinstance(children, list):
        children = [children]
        # print(curr_node)
        # print()

    format_data = dict()
    rv_data = dict()
    
    for ch in children:
        ch_name, ch_render, ch_idxs = conv_sample(ch)
        format_data[ch_name] = ch_render
        rv_data[ch_name] = ch_idxs

    ret_idxs = []

    render = template
    render = render.replace('{{', '{')
    render = render.replace('}}', '}')

    all_nonterms = re.findall('{.*?}', render)

    for to_find in all_nonterms:
        key = to_find[1:-1]
        
        if to_find not in render:
            continue

        offset_idx = render.index(to_find)
        end_idx = offset_idx + len(to_find)
        render = render[:offset_idx] + format_data[key] + render[end_idx:]
        for rv_set, (idx, n) in rv_data[key]:
            ret_idxs.append((rv_set, (idx  + offset_idx, n)))


    if len(rvs) > 0:
        ret_idxs.append((rvs, (0, len(render))))
    return nonterminal, render, ret_idxs

def tagged_data(data):
    ret = []
    for d in tqdm(data):
        p = d[0]
        ret.append(conv_sample(d[1]))
    return ret

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_file', type=str, help='which results to process')
    args = arg_parser.parse_args()


    data = load_json(args.data_file)
    conv_data = tagged_data(data)

    save_json(conv_data, os.path.join('output.json'))
