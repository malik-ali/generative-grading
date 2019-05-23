import os
import torch
import numpy as np
from glob import glob
from random import shuffle
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import src.utils.paths as paths
import src.utils.io_utils as io

from PIL import Image
from sklearn.utils import shuffle as sklearn_shuffle

GROUPING_TO_LABEL = {
    0: [0, 7, 14, 15],
    1: [1,2,3],
    2: [4,6],
    3: [5, 8, 9, 10],
    4: [11, 12, 13]
}
LABEL_TO_GROUPING = {}
for key, values in GROUPING_TO_LABEL.items():
    for value in values:
        LABEL_TO_GROUPING[value] = key


class PyramidGrammar(Dataset):
    def __init__(self, split='train', input_size=224, knowledge_states=False):
        assert split in ['train', 'test']

        self.split = split
        self.input_size = input_size
        self.knowledge_states = knowledge_states

        self._load_data()
        self.transforms = self._load_transforms()

    def _load_data(self):
        split_to_dir = {'train': 'train', 'test': 'cross_val'}
        data_dir = os.path.join(paths.PYRAMIDS_GRAMMAR_DATA_PATH, split_to_dir[self.split])
        label_dirs = glob(os.path.join(data_dir, '*'))
        label_dirs = sorted(label_dirs)

        image_paths, labels = [], []
        for label_dir in label_dirs:
            label = int(os.path.basename(label_dir).replace('samp', '')) - 1  # for 0-indexing
            image_files = glob(os.path.join(label_dir, '*'))
            n_files = len(image_files)
            label = [label for _ in range(n_files)]

            image_paths += image_files
            labels += label

        # shuffle these two lists together
        # image_paths, labels = sklearn_shuffle(image_paths, labels)
        self.data = list(zip(image_paths, labels))
        self.n_labels = 5 if self.knowledge_states else 13

    def _load_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filepath, label = self.data[index]
        if self.knowledge_states:
            label = LABEL_TO_GROUPING[label]
        img = self.transforms(Image.open(filepath).convert("RGB"))
        return img, (1, label)  # return 1 as count ... hack

    @property
    def labels_size(self):
        return self.n_labels


class PyramidImages(Dataset):
    def __init__(self, train_size, split='train', input_size=224, knowledge_states=False):
        assert split in ['train', 'test']
        if split == 'train':
            assert(isinstance(train_size, int) and train_size > 0)

        self.train_size = train_size
        self.knowledge_states = knowledge_states

        self.split = split
        self.input_size = input_size

        self._load_data()
        
        self.transforms = self._load_transforms()

    def _load_data(self):
        train_file = os.path.join(paths.PYRAMIDS_DATA_PATH, 'csvs', 'train.csv')
        labels_file = os.path.join(paths.PYRAMIDS_DATA_PATH, 'csvs', 'labels.csv')

        self.image_dir = os.path.join(paths.PYRAMIDS_DATA_PATH, 'images')
        images = [line.strip() for line in open(train_file).readlines()]
        labels_csv = [line.strip().split(',') for line in open(labels_file).readlines()][1:]
        
        all_labels = set([label for _,  _, label in labels_csv])
        self.labels_to_idx = {label: int(label) - 1 for label in all_labels}
        labels = {name: (int(count), self.labels_to_idx[label]) for name, count, label in labels_csv}
        self.data = sorted([(image, labels[image]) for image  in images], key=lambda x: x[1][0], reverse=True)   # sort by count
        # remove everything that is greater than 12 
        self.data = [d for d in self.data if d[1][1] < 13]
        self.n_labels = 13

        if self.split == 'train':
            self.data = self.data[:self.train_size]     # take first N most frequent for train
    
    def _load_transforms(self):
        return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, inp_idx):
        filename, (count, label) = self.data[inp_idx]
        if self.knowledge_states:
            label = LABEL_TO_GROUPING[label]
        img = self.transforms(Image.open(os.path.join(self.image_dir, filename)).convert("RGB"))
        return img, (count, label)
        
    @property
    def labels_size(self):
        return 5 if self.knowledge_states else self.n_labels


# test sizes and data loading
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = PyramidImages(10, split='train')
    y = dataset[0]
    
    # print(dataset.vocab_size)
    z = DataLoader(dataset, batch_size=64, shuffle=True)
    ctr = 0
    for img, (count, label) in z:
        print(count, label)
        print(img.size())
        
