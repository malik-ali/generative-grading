import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

import src.utils.paths as paths
import src.utils.io_utils as io

ALLOWED_STRATEGIES = ['standard', 'uniform', 'tempered']


class RubricSamples(Dataset):
    def __init__(self, problem, domain='education', sampling_strategy='standard', 
                 split='train', character_level=False, include_anonymized=False):
        
        if isinstance(sampling_strategy, str):
            assert sampling_strategy in ALLOWED_STRATEGIES
            # so sampling_strategy will be a list
            sampling_strategy_list = [sampling_strategy]
        else: 
            assert isinstance(sampling_strategy, list)
            assert set(sampling_strategy).issubset(set(ALLOWED_STRATEGIES))
            sampling_strategy_list = sampling_strategy
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.problem = problem
        self.domain = domain
        self.sampling_strategy_list = sampling_strategy_list
        self._load_data()
        self._load_shard(0)
        self.character_level = character_level
        self.include_anonymized = include_anonymized

    def _load_data(self):
        '''
            Loads all shard-independent data
        '''
        rv_info_list, metadata_dict, num_shards_list, shard_size_list, data_len_list = [], {}, [], [], []
        w2i_list, i2w_list = [], []
        char_w2i_list, char_i2w_list = [], []
        anon_w2i_list, anon_i2w_list = [], []
        anon_char_w2i_list, anon_char_i2w_list = [], []
        shard_num_to_sampling_strategy = []
        shard_num_to_sampling_shard_num = []
        max_len_list, char_max_len_list = [], []
        anon_max_len_list, anon_char_max_len_list = [], [] 

        for sampling_strategy in self.sampling_strategy_list:
            rnn_paths = paths.rnn_data_paths(
                self.problem, self.split, self.domain, sampling_strategy)
            vocab_paths = paths.vocab_paths(self.problem, self.domain)

            """
            for _, path in rnn_paths.items():
                if not os.path.exists(path) and not os.path.exists(path.format(0)):
                    if 'student' not in path:
                        raise RuntimeError("Data path does not exist: [{}]. Generate using preprocessing script".format(path))
            """

            # contains w2i, i2w, num_categories for all rvs
            rv_info = io.load_json(rnn_paths['rv_info_path'])
            metadata = io.load_json(rnn_paths['metadata_path'])
            num_shards = metadata['num_shards']
            shard_size = metadata['shard_size']
            data_len = metadata['data_len']
            max_len = metadata['max_len'] 
            char_max_len = metadata['char_max_len'] 
            anon_max_len = metadata['anon_max_len'] 
            anon_char_max_len = metadata['anon_char_max_len']

            vocab = io.load_json(vocab_paths['vocab_path'])
            w2i, i2w = vocab['w2i'], vocab['i2w']

            char_vocab = io.load_json(vocab_paths['char_vocab_path'])
            char_w2i, char_i2w = char_vocab['w2i'], char_vocab['i2w']

            assert char_w2i[PAD_TOKEN] == w2i[PAD_TOKEN]
            assert char_w2i[START_TOKEN] == w2i[START_TOKEN]
            assert char_w2i[END_TOKEN] == w2i[END_TOKEN]
            assert char_w2i[UNK_TOKEN] == w2i[UNK_TOKEN]

            anon_vocab = io.load_json(vocab_paths['anon_vocab_path'])
            anon_w2i, anon_i2w = anon_vocab['w2i'], anon_vocab['i2w']

            anon_char_vocab = io.load_json(vocab_paths['anon_char_vocab_path'])
            anon_char_w2i, anon_char_i2w = anon_char_vocab['w2i'], anon_char_vocab['i2w']

            assert anon_char_w2i[PAD_TOKEN] == anon_w2i[PAD_TOKEN]
            assert anon_char_w2i[START_TOKEN] == anon_w2i[START_TOKEN]
            assert anon_char_w2i[END_TOKEN] == anon_w2i[END_TOKEN]
            assert anon_char_w2i[UNK_TOKEN] == anon_w2i[UNK_TOKEN]

            rv_info_list.append(rv_info)
            metadata_dict[sampling_strategy] = metadata
            num_shards_list.append(num_shards)
            shard_num_to_sampling_strategy.extend([sampling_strategy]*num_shards)
            shard_num_to_sampling_shard_num.extend(range(num_shards))
            shard_size_list.append(shard_size)
            data_len_list.append(data_len)
            w2i_list.append(w2i)
            i2w_list.append(i2w)
            char_w2i_list.append(char_w2i)
            char_i2w_list.append(char_i2w)
            anon_w2i_list.append(anon_w2i)
            anon_i2w_list.append(anon_i2w)
            anon_char_w2i_list.append(anon_char_w2i)
            anon_char_i2w_list.append(anon_char_i2w)
            max_len_list.append(max_len)
            char_max_len_list.append(char_max_len)
            anon_max_len_list.append(anon_max_len) 
            anon_char_max_len_list.append(anon_char_max_len)

        self.rv_info = rv_info_list[0]  # assume all of these are the same
        self.metadata_dict = metadata_dict
        self.num_shards = sum(num_shards_list)  # consider all shards
        self.shard_size_list = shard_size_list
        self.data_len = sum(data_len_list)
        self.w2i = merge_dicts(*w2i_list)
        self.i2w = merge_dicts(*i2w_list)
        self.vocab = {'w2i': self.w2i, 'i2w': self.i2w}
        self.char_w2i = merge_dicts(*char_w2i_list)
        self.char_i2w = merge_dicts(*char_i2w_list)
        self.char_vocab = {'w2i': self.char_w2i, 'i2w': self.char_i2w}
        self.anon_w2i = merge_dicts(*anon_w2i_list)
        self.anon_i2w = merge_dicts(*anon_i2w_list)
        self.anon_vocab = {'w2i': self.anon_w2i, 'i2w': self.anon_i2w}
        self.anon_char_w2i = merge_dicts(*anon_char_w2i_list)
        self.anon_char_i2w = merge_dicts(*anon_char_i2w_list)
        self.anon_char_vocab = {'w2i': self.anon_char_w2i, 'i2w': self.anon_char_i2w}
        self.shard_num_to_sampling_strategy = shard_num_to_sampling_strategy
        self.shard_num_to_sampling_shard_num = shard_num_to_sampling_shard_num
        self.max_len_list = max_len_list
        self.char_max_len_list = char_max_len_list
        self.anon_max_len_list = anon_max_len_list
        self.anon_char_max_len_list = anon_char_max_len_list
        # take max and we will need to pad to this size
        self.max_len = max(max_len_list)
        self.char_max_len = max(char_max_len_list)
        self.anon_max_len = max(anon_max_len_list)
        self.anon_char_max_len = max(anon_char_max_len_list)

    def _pad_program(self, w2i, x, true_max_len):
        pad_id = w2i[PAD_TOKEN]
        batch_size, cur_max_len = x.shape
        if true_max_len > cur_max_len:
            pad = np.ones((batch_size, true_max_len - cur_max_len)) * pad_id
            pad = pad.astype(np.int)
            x = np.concatenate((x, pad), axis=1)
            x = x.astype(np.int)

        return x

    def _load_shard(self, _shard_num):
        self.curr_shard = _shard_num

        sampling_strategy = self.shard_num_to_sampling_strategy[_shard_num]
        # we need to recover the actual shard_num for a single sampling strategy
        shard_num = self.shard_num_to_sampling_shard_num[_shard_num]
        rnn_paths = paths.rnn_data_paths(
            self.problem, self.split, self.domain, sampling_strategy)

        self.raw_programs = io.load_pickle(rnn_paths['raw_programs_path'].format(shard_num))
        self.anon_raw_programs = io.load_pickle(rnn_paths['anon_raw_programs_path'].format(shard_num))
        self.raw_rvOrders = io.load_pickle(rnn_paths['raw_rvOrder_path'].format(shard_num))

        # Shape: (n x seq_len)
        programs_mat = io.loadmat(rnn_paths['feat_programs_path'].format(shard_num))
        char_programs_mat = io.loadmat(rnn_paths['char_feat_programs_path'].format(shard_num))

        anon_programs_mat = io.loadmat(rnn_paths['anon_feat_programs_path'].format(shard_num))
        anon_char_programs_mat = io.loadmat(rnn_paths['anon_char_feat_programs_path'].format(shard_num))

        self.programs = programs_mat['programs']
        self.lengths = programs_mat['lengths'].squeeze()
        self.tiers = programs_mat['tiers'][0]

        self.char_programs = char_programs_mat['programs']
        self.char_lengths = char_programs_mat['lengths'].squeeze()

        self.anon_programs = anon_programs_mat['programs']
        self.anon_lengths = anon_programs_mat['lengths'].squeeze()

        self.anon_char_programs = anon_char_programs_mat['programs']
        self.anon_char_lengths = anon_char_programs_mat['lengths'].squeeze()

        # pad programs to single shape
        self.programs = self._pad_program(self.w2i, self.programs, self.max_len)
        self.char_programs = self._pad_program(self.char_w2i, self.char_programs, self.char_max_len)
        self.anon_programs = self._pad_program(self.anon_w2i, self.anon_programs, self.anon_max_len)
        self.anon_char_programs = self._pad_program(self.anon_char_w2i, self.anon_char_programs, self.anon_char_max_len)

        # Shape: (n x num_labels).  1 if label, 0 otherwise
        self.labels = io.load_np(rnn_paths['feat_labels_path'].format(shard_num))

        rvOrders_mat = io.loadmat(rnn_paths['feat_rvOrder_path'].format(shard_num))
        self.rvOrders = rvOrders_mat['rv_orders']
        self.rvOrders_lengths = rvOrders_mat['lengths'].squeeze()

    def shard_num_idx_from_idx(self, idx):
        if idx > len(self):
            raise ValueError('Index out of bounds. Dataset size {}. Accessing index {}'.format(len(self), idx))

        if len(self.shard_size_list) == 1:
            shard_size = self.shard_size_list[0]
            shard_num =  idx // shard_size
            new_idx = idx % shard_size
        else:
            shard_sizes = np.array(self.shard_size_list)
            shard_cumsum = np.cumsum(shard_sizes)
            shard_num = np.where(idx < shard_cumsum)[0][0]
            if shard_num == 0:
                new_idx = idx
            else:
                new_idx = idx - shard_cumsum[shard_num - 1]

        return shard_num, new_idx

    def __len__(self):
        return self.data_len

    def __getitem__(self, inp_idx):
        shard_num, idx = self.shard_num_idx_from_idx(inp_idx)
        if shard_num != self.curr_shard:
            self._load_shard(shard_num)

        if self.character_level:
            if self.include_anonymized:
                return (self.programs[idx], self.lengths[idx],
                        self.char_programs[idx], self.char_lengths[idx],
                        self.anon_programs[idx], self.anon_lengths[idx],
                        self.anon_char_programs[idx], self.anon_char_lengths[idx],
                        self.labels[idx], self.tiers[idx],
                        self.rvOrders[idx], self.rvOrders_lengths[idx])
            else:
                return (self.programs[idx], self.lengths[idx],
                        self.char_programs[idx], self.char_lengths[idx],
                        self.labels[idx], self.tiers[idx],
                        self.rvOrders[idx], self.rvOrders_lengths[idx])
        else:
            if self.include_anonymized:
                return (self.programs[idx], self.lengths[idx],
                        self.anon_programs[idx], self.anon_lengths[idx],
                        self.labels[idx], self.tiers[idx],
                        self.rvOrders[idx], self.rvOrders_lengths[idx])
            else:
                return (self.programs[idx], self.lengths[idx],
                        self.labels[idx], self.tiers[idx],
                        self.rvOrders[idx], self.rvOrders_lengths[idx])

    def get_raw_program(self, inp_idx):
        shard_num, idx = self.shard_num_idx_from_idx(inp_idx)
        if shard_num != self.curr_shard:
            self._load_shard(shard_num)

        return self.raw_programs[idx]

    def get_anon_raw_program(self, inp_idx):
        shard_num, idx = self.shard_num_idx_from_idx(inp_idx)
        if shard_num != self.curr_shard:
            self._load_shard(shard_num)

        return self.anon_raw_programs[idx]

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def anon_vocab_size(self):
        return len(self.anon_w2i)

    @property
    def char_vocab_size(self):
        return len(self.char_w2i)

    @property
    def anon_char_vocab_size(self):
        return len(self.anon_char_w2i)

    @property
    def labels_size(self):
        return self.labels.shape[1]

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


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# test sizes and data loading
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = RubricSamples('liftoff', domain='education', 
                            sampling_strategy='standard')
    print(dataset.vocab_size)
    z = DataLoader(dataset, batch_size=4, shuffle=True)
    ctr = 0
    for b in z:
        for k in b:
            print(k.size())
        break
