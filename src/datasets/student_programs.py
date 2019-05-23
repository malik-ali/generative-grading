import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

import src.utils.paths as paths
import src.utils.io_utils as io


class StudentPrograms(Dataset):
    def __init__(self, problem, character_level=False, include_anonymized=False):
        
        self.problem = problem
        self._load_metadata()
        self._load_data()
        self.character_level = character_level
        self.include_anonymized = include_anonymized

    def _load_metadata(self):
        '''
            Loads all housekeeping data
        '''
        rnn_paths = paths.rnn_data_paths(self.problem, 'train', 'education', 'standard')
        vocab_paths = paths.vocab_paths(self.problem, 'education')

        for _, path in rnn_paths.items():
            if not os.path.exists(path) and not os.path.exists(path.format(0)):
                if 'student' not in path:
                    raise RuntimeError("Data path does not exist: [{}]. Generate using preprocessing script".format(path))

        metadata = io.load_json(rnn_paths['metadata_path'])
        self.max_len = metadata['max_len'] 
        self.char_max_len = metadata['char_max_len'] 
        self.anon_max_len = metadata['anon_max_len'] 
        self.anon_char_max_len = metadata['anon_char_max_len']

        self.vocab = io.load_json(vocab_paths['vocab_path'])
        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']

        self.char_vocab = io.load_json(vocab_paths['char_vocab_path'])
        self.char_w2i, self.char_i2w = self.char_vocab['w2i'], self.char_vocab['i2w']

        assert self.char_w2i[PAD_TOKEN] == self.w2i[PAD_TOKEN]
        assert self.char_w2i[START_TOKEN] == self.w2i[START_TOKEN]
        assert self.char_w2i[END_TOKEN] == self.w2i[END_TOKEN]
        assert self.char_w2i[UNK_TOKEN] == self.w2i[UNK_TOKEN]

        self.anon_vocab = io.load_json(vocab_paths['anon_vocab_path'])
        self.anon_w2i, self.anon_i2w = self.anon_vocab['w2i'], self.anon_vocab['i2w']

        self.anon_char_vocab = io.load_json(vocab_paths['anon_char_vocab_path'])
        self.anon_char_w2i, self.anon_char_i2w = self.anon_char_vocab['w2i'], self.anon_char_vocab['i2w']

        assert self.anon_char_w2i[PAD_TOKEN] == self.anon_w2i[PAD_TOKEN]
        assert self.anon_char_w2i[START_TOKEN] == self.anon_w2i[START_TOKEN]
        assert self.anon_char_w2i[END_TOKEN] == self.anon_w2i[END_TOKEN]
        assert self.anon_char_w2i[UNK_TOKEN] == self.anon_w2i[UNK_TOKEN]

      
    def _pad_program(self, w2i, x, true_max_len):
        pad_id = w2i[PAD_TOKEN]
        batch_size, cur_max_len = x.shape
        if true_max_len > cur_max_len:
            pad = np.ones((batch_size, true_max_len - cur_max_len)) * pad_id
            pad = pad.astype(np.int)
            x = np.concatenate((x, pad), axis=1)
            x = x.astype(np.int)

        return x

    def _load_data(self):

        rnn_paths = paths.rnn_data_paths(self.problem, 'train', 'education', 'standard')

        self.raw_programs = io.load_pickle(rnn_paths['raw_student_programs_path'])
        self.anon_raw_programs = io.load_pickle(rnn_paths['anon_raw_student_programs_path'])

        # Shape: (n x seq_len)
        programs_mat = io.loadmat(rnn_paths['student_programs_path'])
        char_programs_mat = io.loadmat(rnn_paths['student_char_programs_path'])

        anon_programs_mat = io.loadmat(rnn_paths['anon_student_programs_path'])
        anon_char_programs_mat = io.loadmat(rnn_paths['anon_student_char_programs_path'])

        self.programs = programs_mat['programs']
        self.lengths = programs_mat['lengths'].squeeze()
        #self.tiers = programs_mat['tiers'][0]

        self.char_programs = char_programs_mat['programs']
        self.char_lengths = char_programs_mat['lengths'].squeeze()

        self.anon_programs = anon_programs_mat['programs']
        self.anon_lengths = anon_programs_mat['lengths'].squeeze()

        self.anon_char_programs = anon_char_programs_mat['programs']
        self.anon_char_lengths = anon_char_programs_mat['lengths'].squeeze()

        # pad programs to single shape
        # TODO: do we need this for student programs?
        #self.programs = self._pad_program(self.w2i, self.programs, self.max_len)
        #self.char_programs = self._pad_program(self.char_w2i, self.char_programs, self.char_max_len)
        #self.anon_programs = self._pad_program(self.anon_w2i, self.anon_programs, self.anon_max_len)
        #self.anon_char_programs = self._pad_program(self.anon_char_w2i, self.anon_char_programs, self.anon_char_max_len)

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        if self.character_level:
            if self.include_anonymized:
                return (self.programs[idx], self.lengths[idx],
                        self.char_programs[idx], self.char_lengths[idx],
                        self.anon_programs[idx], self.anon_lengths[idx],
                        self.anon_char_programs[idx], self.anon_char_lengths[idx],
                        self.raw_programs[idx], self.anon_raw_programs[idx])
            else:
                return (self.programs[idx], self.lengths[idx],
                        self.char_programs[idx], self.char_lengths[idx],
                        self.raw_programs[idx])
        else:
            if self.include_anonymized:
                return (self.programs[idx], self.lengths[idx],
                        self.anon_programs[idx], self.anon_lengths[idx],
                        self.raw_programs[idx], self.anon_raw_programs[idx])
            else:
                return (self.programs[idx], self.lengths[idx], self.raw_programs[idx])
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


# test sizes and data loading
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = StudentPrograms('liftoff')
    z = DataLoader(dataset, batch_size=4, shuffle=True)
    ctr = 0
    for b in z:
        for k in b:
            print(k[0])
            print('=======')
        break


class CodeOrgStudentPrograms(StudentPrograms):
    def __init__(self, problem, character_level=False, include_anonymized=False):
        super().__init__(problem, character_level=character_level, 
                         include_anonymized=include_anonymized)
        
        rnn_paths = paths.rnn_data_paths(self.problem, 'train', 'education', 'standard')
        self.labels = io.load_np(rnn_paths['feat_labels_path'])
        self.zipfs = io.load_np(rnn_paths['feat_zipfs_path'])

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data = list(data)
        data.append(self.labels[idx])
        data.append(self.zipfs[idx])
        return tuple(data)
