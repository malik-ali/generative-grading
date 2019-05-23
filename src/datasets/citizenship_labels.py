r"""Small labeled dataset from the citizenship dataset."""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from collections import Counter, OrderedDict

from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_DIR, '..', 'rubricsampling', 'citizenshipRawData')


class CitizenshipLabels(Dataset):
    def __init__(self, problem, split='train', vocab=None):
        self.problem = problem
        self.split = split
        assert self.split in ['train', 'valid', 'test'], \
            "split {} not supported.".format(self.split)

        raw_inputs, labels = self._load_data()
        raw_inputs = self._clean_data(raw_inputs)
        self.raw_inputs = raw_inputs
        raw_tiers = self._get_tiers(self.raw_inputs)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocab(raw_inputs)
        self.w2i = self.vocab['w2i']
        self.i2w = self.vocab['i2w']
       
        self.inputs, self.lengths, self.tiers, self.max_length \
            = self._process_data(raw_inputs, raw_tiers)
        labels = labels[:, np.newaxis]
        self.labels = labels
        self.size = len(self.inputs)

    def _get_tiers(self, inputs):
        counter = Counter()
        for text in inputs:
            counter[text] += 1
        tiers = {}
        for text, val in counter.items():
            # head = 0, tail = 1
            tiers[text] = 0 if (val > 1) else 1
        return tiers

    def _clean_data(self, inputs):
        # lower case everything and remove extra whitespace
        # not sure if we want to fix typos?
        clean_inputs = []
        for text in inputs:
            text = text.lower()
            text = ' '.join(text.split())
            clean_inputs.append(text)
        return clean_inputs

    def _load_data(self):
        filepath_698 = os.path.join(DATA_DIR, 'studentanswers_grades_698.tsv')
        df_698 = pd.read_csv(filepath_698, sep='\t')
        df_698 = df_698[df_698['Q#'] == self.problem]

        if self.split == 'train':
            df = df_698[:50]
        elif self.split == 'valid':
            df = df_698[50:]
        elif self.split == 'test':
            df = df_698[50:]

        df = df.dropna()
        inputs = np.asarray(df['answer'])
        G1 = np.asarray(df['G1'])
        G2 = np.asarray(df['G2'])
        G3 = np.asarray(df['G3'])
        labels = np.vstack([G1, G2, G3]).T

        # get majority
        labels = np.sum(labels, axis=1) >= 2
        labels = labels.astype(np.int)

        return inputs, labels

    def _process_data(self, raw_inputs, raw_tiers):
        inputs, lengths, tiers = [], [], []

        n = len(raw_inputs)
        max_len = 0

        for i in range(n):
            seq = raw_inputs[i]
            seq_list = seq.split()
            tokens = [START_TOKEN] + seq_list + [END_TOKEN]
            length = len(tokens)
            tier = raw_tiers[seq]
            max_len = max(max_len, length)

            inputs.append(tokens)
            lengths.append(length)
            tiers.append(tier)

        for i in range(n):
            tokens = inputs[i]
            length = lengths[i]
            tokens.extend([PAD_TOKEN] * (max_len - length))
            tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) 
                      for token in tokens]
            inputs[i] = tokens
       
        inputs = np.array(inputs)
        lengths = np.array(lengths)
        tiers = np.array(tiers)

        return inputs, lengths, tiers, max_len

    def _build_vocab(self, texts):
        w2i = dict()
        i2w = dict()
        w2c = OrderedCounter()
        special_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for text in texts:
            tokens = text.split()
            w2c.update(tokens)

        for w, c in w2c.items():
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)

        return vocab

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        seq_src = self.inputs[index]
        seq_len = self.lengths[index]
        label = self.labels[index]
        tier = self.tiers[index]
        raw_string = self.raw_inputs[index]

        print("WARNING, this will break training. If we need to retrain, sad times")
        return seq_src, seq_len, raw_string, label, tier

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_id(self):
        return self.w2i[PAD_TOKEN]

    @property
    def sos_id(self):
        return self.w2i[START_TOKEN]

    @property
    def eos_id(self):
        return self.w2i[END_TOKEN]

    @property
    def unk_id(self):
        return self.w2i[UNK_TOKEN]


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
