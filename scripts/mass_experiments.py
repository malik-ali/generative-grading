import getpass
import argparse
import itertools
from dotmap import DotMap
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
import os, sys
from pprint import pprint

from src.utils.setup import process_config
from src.utils.io_utils import create_temp_copy
from src.agents import *

import numpy as np
import time
import subprocess
from io import BytesIO
import pandas as pd

EXP_BASE = '/mnt/fs5/{}/generative-grading'.format(getpass.getuser())

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU: #{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return int(idx)

def random_search_params():
   return {
       "encoder_kwargs.hidden_size": 2**np.random.randint(4, 10),
       "encoder_kwargs.embedding_size": 2**np.random.randint(4, 10),
       "encoder_kwargs.num_layers": np.random.randint(1, 9),
       "encoder_kwargs.hidden_dropout": np.random.uniform(0, 0.5),
       #"encoder_kwargs.word_dropout": np.random.uniform(0, 0.5),
       "inference_kwargs.hidden_size": 2**np.random.randint(4, 10),
       "inference_kwargs.embedding_size": 2**np.random.randint(4, 10),
       "inference_kwargs.num_attention_heads": 0,
       "inference_kwargs.hidden_dropout": 0,
       "inference_kwargs.use_batchnorm": False,
       "weight_decay": 10**np.random.randint(-9, -1)
    }

def flat_to_nested_dict(d):
    '''
    Converts flattened dictionary with keys of the form
    key1.key2.name to a nested dictionary of key1: {key2: name}
    '''
    ret = dict()
    for k, v in d.items():
        nested_keys = k.split('.')
        curr = ret
        for nest_key in nested_keys[:-1]:
            if nest_key not in curr:
                curr[nest_key] = dict()
            curr = curr[nest_key]
        curr[nested_keys[-1]] = v
    return ret


def run_agent(AgentClass, config, gpu_device):
    config.gpu_device = gpu_device

    print('======>  Running experiment for config:', gpu_device)
    print(config)

    agent = AgentClass(config)
    
    agent.run()
    agent.finalise()
    
    return config.summary_dir

def random_search(config_path, num_exps):
    gpu = get_free_gpu()
    print("=============== Acquired GPU: {} ===============)".format(gpu))
    for n in range(num_exps):
        params = random_search_params()
        nested_dict = flat_to_nested_dict(params)
        curr_config = process_config(config_path, override_dotmap=DotMap(nested_dict), exp_base=EXP_BASE)
        exp_dir = run_agent(globals()[curr_config.agent], curr_config, gpu)

        print('======> Finished: ', exp_dir)

    print('================================================================')
    print('*                  COMPLETED MASS EXPERIMENTS                  *')
    print('================================================================')    

def main(config_path, num_exps):
    # Make copy of config to tmp folder so modifications to config
    # don't mess up experiments mid-run
    temp_config_path = create_temp_copy(config_path)
    random_search(temp_config_path, num_exps)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'config',
        metavar='config-file',
        default='None')
    arg_parser.add_argument(
        'num_exps',
        metavar='num-exps',
        type=int,
        default=5,
        help='Number of experiments to try')

    args = arg_parser.parse_args()

    main(args.config, args.num_exps)

