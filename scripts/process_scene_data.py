import os
import re
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from PIL import Image

import src.utils.paths as paths
import src.utils.io_utils as io

import torch
import torchvision.models as models
from torchvision import transforms

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(CUR_DIR)
from process_data import (fix_labels, featurise_rv_order, create_rv_info)


def featurise_labels(labels, rv_info, all_rvs):
    '''
        - labels: shape N list. Each entry is a dict of all the rvs assignment
        - rv_info is a dict of rv2i, i2rv, and rv -> num_categories
        - all_rvs is a dict of rv -> list of categories for this rv

        returns N x (num_rvs) array where each the j-th column represents
        the value x takes at rv X_j
    '''
    num_rvs = len(all_rvs)
    w2i = rv_info['w2i']
    out_labels = np.zeros((len(labels), num_rvs), np.int32)
    for n, rv_assignments in enumerate(labels):
        for rv, val in rv_assignments.items():
            if rv not in w2i:
                print('WARNING: RV [{}] not in vocab. Ignoring'.format(rv))
                continue
            val = str(val)
            rv_idx = w2i[rv]
            # val = str(val)  # if we support chain, uncomment
            # [0] to grab the keys not the probabilities
            val_idx = all_rvs[rv][0].index(val)
            out_labels[n][rv_idx] = val_idx

    return out_labels


def load_raw_scene_graph_data(counts_path, labels_path, rv_order_path, 
                              images_path, tiers_path):
    counts = io.load_pickle(counts_path)
    labels = io.load_pickle(labels_path)
    images = io.load_pickle(images_path)
    tiers = io.load_pickle(tiers_path)
    rv_order = io.load_pickle(rv_order_path)
    scene_graphs = list(counts.keys())
    p_images = [images[p] for p in scene_graphs]
    p_labels = [fix_labels(labels[p]) for p in scene_graphs]
    p_rvorders = [rv_order[p] for p in scene_graphs]
    p_tiers = [tiers[p] for p in scene_graphs]
    return (scene_graphs, p_images, p_labels, p_rvorders, p_tiers, counts)


def featurise_images(images, model, device, image_transforms, batch_size=100):
    n = len(images)
    n_batches = n // batch_size
    
    features = []
    with torch.no_grad():
        for i in range(n_batches):
            x_i = []
            for j in range(batch_size*i, (batch_size+1)*i):
                image_ij = images[j]
                image_ij = Image.fromarray(image_ij)
                image_ij = image_transforms(image_ij)
                x_i.append(image_ij)
            x_i = torch.stack(x_i)
            x_i = x_i.to(device)
            out_i = get_resnet18_features(model, x_i)
            out_i = out_i.cpu().numpy()
            features.append(out_i)
    features = np.concatenate(features, axis=0)

    return features


def load_classification_model():
    model = models.resnet18(pretrained=True)
    input_size = 224
    for param in model.parameters():
        param.requires_grad = False
    
    return model, input_size


def get_resnet18_features(net, x):
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)

    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)

    x = net.avgpool(x)
    x = x.view(x.size(0), -1)

    return x


def load_data_transforms(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def make_scene_graph_data(device, problem, split, sampling_strategy='standard',
                          use_resnet=False):
    data_paths = paths.scene_graph_data_paths(problem, split, sampling_strategy)
    os.makedirs(data_paths['data_path'], exist_ok=True)

    (counts_paths, labels_paths, images_paths, rv_order_paths, 
     tiers_paths, all_rvs_path) = \
         paths.raw_scene_graph_data_paths(problem, split, sampling_strategy)
    n_shards = len(counts_paths)

    all_rvs = io.load_json(all_rvs_path)
    rv_info = create_rv_info(all_rvs)
    # save all_rvs into rv_info
    rv_info['values'] = all_rvs

    data_len = 0
    shard_size = 0

    if use_resnet:
        # load huge model :(
        print('loading deep net for feature extraction...')
        net, expected_input_dim = load_classification_model()
        net = net.to(device)
        image_transforms = load_data_transforms(expected_input_dim)

    for i in range(n_shards):
        scene_graphs_i, images_i, labels_i, rv_order_i, tiers_i, _ = load_raw_scene_graph_data(
            counts_paths[i], labels_paths[i], rv_order_paths[i], 
            images_paths[i], tiers_paths[i])

        n_items_i = len(scene_graphs_i)

        if use_resnet:
            feat_images_i = featurise_images(images_i, net, device, image_transforms)
        else:
            feat_images_i = images_i

        # assumes equally sized shards (except smaller remaining last one)
        shard_size = max(shard_size, n_items_i)
        data_len += n_items_i

        feat_labels_i = featurise_labels(labels_i, rv_info, all_rvs)
        feat_rv_order_i, rv_order_lengths_i = featurise_rv_order(rv_order_i, rv_info)

        image_mats_i = dict(images=feat_images_i, 
                            tiers=tiers_i)
        rv_order_mats_i = dict(rv_orders=feat_rv_order_i, 
                               lengths=rv_order_lengths_i)

        io.savemat(image_mats_i, data_paths['feat_images_path'].format(i))
        io.save_np(feat_labels_i, data_paths['feat_labels_path'].format(i))
        io.save_pickle(rv_order_i, data_paths['raw_rvOrder_path'].format(i))
        io.savemat(rv_order_mats_i, data_paths['feat_rvOrder_path'].format(i))

    io.save_json(rv_info, data_paths['rv_info_path'])

    metadata = dict(
        data_len=data_len,
        num_shards=n_shards,
        shard_size=shard_size
    )

    io.save_json(metadata, data_paths['metadata_path'])


def main(device, problem, split, sampling_strategy, use_resnet=False):
    make_scene_graph_data(device, problem, split, sampling_strategy, 
                          use_resnet=use_resnet)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to preprocess')
    arg_parser.add_argument(
        '--split',
        default='train',
        help='Which split (train|val|test) [default: train]')
    arg_parser.add_argument(
        '--sampling-strategy',
        default='standard',
        help='How we sample from the grammar (standard|uniform|tempered) [default: standard]') 
    arg_parser.add_argument(
        '--resnet',
        action='store_true',
        default=False,
        help='store ResNet embeddings instead of raw images')
    arg_parser.add_argument(
        '--cuda', 
        action='store_true', 
        default=False,
        help='enables CUDA training')
    args = arg_parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    main(device, args.problem, args.split, args.sampling_strategy, 
         use_resnet=args.resnet)
