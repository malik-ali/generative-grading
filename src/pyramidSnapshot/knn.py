r"""Given N training images...,

For each test image, define K nearest neighbors
as K training images with minimum sum-squared
difference (by pixels). Predict most common label
for each label out of the K set.
"""

import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from src.datasets.pyramid import PyramidImages
from scipy.spatial.distance import pdist, squareform

import getpass
USER = getpass.getuser()

DISTMAT_PATH = f'/mnt/fs5/{USER}/generative-grading/distmat.npy'


def get_images(dataset):
    n = len(dataset)
    images = []
    for i in tqdm(range(n)):
        img_i, _ = dataset.__getitem__(i)
        img_i = img_i.numpy()
        img_i = img_i[np.newaxis, ...]
        images.append(img_i)
    images = np.concatenate(images, axis=0)

    return images


def compute_distance_matrix(test_dataset):
    test_images  = get_images(test_dataset)
    test_images = test_images.reshape(len(test_images), -1)
    print('Building distance matrix')
    distances = pdist(test_images, 'euclidean')

    return distances


def load_labels_from_dataset(dataset):
    n = len(dataset)
    labels = []
    for i in tqdm(range(n)):
        _, (_, lab_i) = dataset.__getitem__(i)
        labels.append(lab_i)
    labels = np.array(labels)
    return labels


def load_counts_from_dataset(dataset):
    n = len(dataset)
    counts = []
    for i in tqdm(range(n)):
        _, (cnt_i, _) = dataset.__getitem__(i)
        counts.append(cnt_i)
    counts = np.array(counts)
    return counts


def run_nearest_neighbors(N, K):
    test_dataset = PyramidImages(N, split='test')

    print('Processing test dataset')
    test_labels = load_labels_from_dataset(test_dataset)
    test_counts = load_counts_from_dataset(test_dataset)
    train_labels = test_labels[:N]  # pick neighbors from N training pts

    n_test = len(test_dataset)

    if os.path.exists(DISTMAT_PATH):
        dist_mat = np.load(DISTMAT_PATH)
    else:
        test_dataset = PyramidImages(N, split='test')
        dist_mat = compute_distance_matrix(test_dataset)
        np.save(DISTMAT_PATH, dist_mat)
    dist_mat = squareform(dist_mat)

    is_correct = []
    correct_count = []
    for i in tqdm(range(n_test)):
        label = test_labels[i]
        count = test_counts[i]
        smallest = np.argsort(dist_mat[i][:N])[:K]
        preds = []
        for j in smallest:
            preds_j = train_labels[j]
            preds.append(preds_j)
        preds = np.array(preds)
        pred = np.argmax(np.bincount(preds))
        correct = int(pred == label)
        is_correct.append(correct)
        correct_count.append([correct, count])

    is_correct = np.array(is_correct)
    correct_count = np.array(correct_count)

    return np.mean(is_correct), correct_count


def main(N, K, out_dir):
    results, outputs = run_nearest_neighbors(N, K)
    print(results)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    np.save(os.path.join(out_dir, 'correct_count_N_{}_K_{}.npy'.format(N, K)), outputs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='number of datasets')
    parser.add_argument('--K', type=int, default=100, help='number of neighbors')
    parser.add_argument('--out-dir', type=str, default='./',
                        help='where to store stuff [default: ./]')
    args = parser.parse_args()

    main(args.N, args.K, args.out_dir)
