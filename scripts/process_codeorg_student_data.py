import os
import sys
import argparse

import src.utils.paths as paths
import src.utils.io_utils as io

from scripts.process_data import load_vocabs

from tqdm import tqdm
import re
import numpy as np

from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

def raw_student_data(problem, account_for_counts=False):
    counts_path, labels_path, zipfs_path, anon_mapping_path = \
        paths.raw_codeorg_student_paths(problem)
    prog2counts = io.load_pickle(counts_path)
    prog2labels = io.load_pickle(labels_path)
    prog2zipfs = io.load_pickle(zipfs_path)
    prog2anons = io.load_pickle(anon_mapping_path)

    if account_for_counts:
        programs = []
        labels = []
        zipfs = []
        anon_programs = []
        for prog, count in prog2counts.iteritems():
            for i in range(count):
                programs.append(prog)
                labels.append(prog2labels[prog])
                zipfs.append(prog2zipfs[prog])
                anon_programs.append(prog2anons[prog])
    else:
        programs = list(prog2counts.keys())
        labels = [prog2labels[p] for p in programs]
        zipfs = [prog2zipfs[p] for p in programs]
        anon_programs = [prog2anons[p] for p in programs]
    
    return programs, labels, zipfs, anon_programs


def featurise_programs_rnn(programs, vocab, max_len, character_level=False):
    code_tokens = []
    raw_programs = []

    # keep track of max_len to pad the sequences
    lengths = np.zeros(len(programs), int)
    w2i = vocab['w2i']

    for i, prog in enumerate(tqdm(programs)):
        prog_tokens = prog.split()
   
        if character_level:
            # this will break a string of tokens into 
            # many individual characters.
            prog_tokens = list(" ".join(prog_tokens))
        
        enc = [START_TOKEN] + prog_tokens + [END_TOKEN]
        
        # student program lengths might be longer than anything seen in training
        if len(enc) > max_len:
            print("WARNING: Skipping prgoram. max_len ({}) exceeded. Found len {}"\
                .format(max_len, len(enc)))

        lengths[i] = len(enc)
        code_tokens.append(enc)
        raw_programs.append(prog)

    # Pad all sequences to max_len
    for enc in code_tokens:
        if len(enc) < max_len:
            enc += [PAD_TOKEN for _ in range(max_len - len(enc))]
        assert len(enc) == max_len

    # student programs might have unseen tokens
    unk_idx = w2i[UNK_TOKEN]
    encodings = [[w2i.get(t, unk_idx) for t in tokens] for tokens in code_tokens]

    return np.stack(encodings, axis=0), lengths, raw_programs


def process_student_data(problem, account_for_counts=False):
    rnn_paths = paths.rnn_data_paths(problem, 'train', 'education', 'standard')
    os.makedirs(rnn_paths['student_data_path'], exist_ok=True)
    vocab_paths = paths.vocab_paths(problem, 'education')

    if not os.path.isfile(vocab_paths['vocab_path']):
        raise ValueError('Run preprocessing script on rubric samples first to generate vocab file.')

    vocab, char_vocab, anon_vocab, anon_char_vocab = load_vocabs(vocab_paths)
    metadata = io.load_json(rnn_paths['metadata_path'])

    # load training max-lengths
    # max_len = metadata['max_len']
    # char_max_len = metadata['char_max_len']
    # anon_max_len = metadata['anon_max_len']
    # anon_char_max_len = metadata['anon_char_max_len']

    # we do not want to load these from metadata bc some programs may be longer than ones
    # seen in training. Instead we want to recompute the maximum length...
    programs, labels, zipfs, anon_programs = raw_student_data(problem, account_for_counts)
    
    # we +2 to include start and end tokens
    max_len = max(len(x.split()) for x in programs) + 2
    char_max_len = max(len(x) for x in programs) + 2
    anon_max_len = max(len(x.split()) for x in anon_programs) + 2
    anon_char_max_len = max(len(x) for x in anon_programs) + 2

    feat_programs, program_lengths, raw_programs = featurise_programs_rnn(programs, vocab, max_len)
    char_feat_programs, char_program_lengths, _ = featurise_programs_rnn(
        programs, char_vocab, char_max_len, character_level=True)

    anon_feat_programs, anon_program_lengths, anon_raw_programs = featurise_programs_rnn(
        anon_programs, anon_vocab, anon_max_len)
    anon_char_feat_programs, anon_char_program_lengths, _ = featurise_programs_rnn(
        anon_programs, anon_char_vocab, anon_char_max_len, character_level=True)

    program_mats = dict(programs=feat_programs, 
                        lengths=program_lengths)
    char_program_mats = dict(programs=char_feat_programs, 
                             lengths=char_program_lengths)
    anon_program_mats = dict(programs=anon_feat_programs, 
                             lengths=anon_program_lengths)
    anon_char_program_mats = dict(programs=anon_char_feat_programs, 
                                  lengths=anon_char_program_lengths)

    io.save_pickle(raw_programs, rnn_paths['raw_student_programs_path'])
    io.savemat(char_program_mats, rnn_paths['student_char_programs_path'])
    io.savemat(program_mats, rnn_paths['student_programs_path'])

    io.save_pickle(anon_raw_programs, rnn_paths['anon_raw_student_programs_path'])
    io.savemat(anon_char_program_mats, rnn_paths['anon_student_char_programs_path'])
    io.savemat(anon_program_mats, rnn_paths['anon_student_programs_path'])

    io.save_np(labels, rnn_paths['feat_labels_path'])
    io.save_np(zipfs, rnn_paths['feat_zipfs_path'])


def main(problem, account_for_counts):
    process_student_data(problem, account_for_counts)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to preprocess')
    arg_parser.add_argument(
        '--account-for-counts',
        action='store_true',
        default=False,
        help='duplicate inputs by counts')
    args = arg_parser.parse_args()

    main(args.problem, args.account_for_counts)
