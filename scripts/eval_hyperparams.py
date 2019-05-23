import argparse
import itertools
import numpy as np
import os, sys
from pprint import pprint

from src.utils.setup import load_config
from src.utils.io_utils import load_json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from sklearn.linear_model import LinearRegression

from scipy import stats

import time

SEARCH_PARAMS = [
   "encoder_kwargs.hidden_size",
   "encoder_kwargs.embedding_size",
   "encoder_kwargs.num_layers",
   "encoder_kwargs.hidden_dropout",
   "inference_kwargs.hidden_size",
   "inference_kwargs.embedding_size",
   "weight_decay"
]

def get_key_for_metric(keys, to_find):
    to_find = 'validation/loss/loss'
    cands = [k for k in keys if k.endswith(to_find)]
    if len(cands) > 1:
        raise ValueError("Duplicate keys: {}".format(cands))
    if not cands:
        return None

    return cands[0]

def nested_access(config, param):
    keys = param.split('.')
    curr = config
    for k in keys:
        curr = curr[k]
    return curr

def config_to_vec(config):
    ret = dict()
    for param in SEARCH_PARAMS:
        ret[param] = nested_access(config, param)
    return ret #np.array(ret)


def load_exp_data(all_exp_dir):
    params = []
    losses = []
    for exp_name in os.listdir(all_exp_dir):
        try:
            exp_dir = os.path.join(all_exp_dir, exp_name)
            config = load_config(os.path.join(exp_dir, 'config.json'))
            vec = config_to_vec(config)
            vec['exp_name'] = exp_name
            summaries = load_json(os.path.join(exp_dir, 'summaries', 'all_scalars.json'))
            k_loss = get_key_for_metric(summaries.keys(), 'validation/loss/loss')
            if k_loss is None:
                print('Metric not foud... skipping')
                continue 
            loss = np.average([x[2] for x in summaries[k_loss][-5:]])

            params.append(vec)
            losses.append(loss)
        except FileNotFoundError:
            print('File not found... skipping')
            continue


    return params, losses 



def main(exp_dir):
    params, losses = load_exp_data(exp_dir)
    K = 30
    zipped = zip(params, losses)

    print(SEARCH_PARAMS)
    sort = sorted(zipped, key=lambda x: x[1])
    print('Best:')
    for config, loss in sort:
        print(loss)
        pprint(config)
        print()
    
    # print('Worst:')
    # for config, loss in sort[-K :]:
        # print('\t', config, ' - {:3f}'.format(loss))


    # X = np.array(params)
    # y = np.array(losses)
    # # import pdb; pdb.set_trace()
    # reg = LinearRegression().fit(X, y)
    # print('Score: ', reg.score(X, y))
    # print('reg.coef_: ', reg.coef_)
    # print(SEARCH_PARAMS)





if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'experiment_dir',
        metavar='exp-dir',
        default='None',
        help='The path to the experiments with the hyperparameter search')

    args = arg_parser.parse_args()

    main(args.experiment_dir)

