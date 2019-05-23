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

def raw_student_data(problem):
    data_path, anon_mapping_path = paths.raw_student_paths(problem)
    progs = io.load_pickle(data_path)
    anon_mapping = io.load_pickle(anon_mapping_path)
    programs = list(progs.keys())
    anon_programs = [anon_mapping[p] for p in programs]
    return programs, anon_programs

def featurise_labels(labels):
    all_labels = set(l for s in labels for l in s)
    mapping = {l: i for i, l in enumerate(all_labels)}

    out_labels = np.zeros((len(labels), len(mapping)), np.float32)
    for n, s in enumerate(labels):
        indices = list({mapping[l] for l in s})
        out_labels[n][indices] = 1.

    return out_labels, mapping

def strip_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def tokenize_code(code_str):
    ret = []
    word = ""
    for ch in code_str:
        if word and (ch.isspace() or not ch.isalnum()):
            ret.append(word)
            word = ""
        if ch.isspace():
            continue
        elif not ch.isalnum():
            ret.append(ch)
        else:
            word += ch
    return ret


def featurise_programs_rnn(programs, vocab, max_len, character_level=False):
    code_tokens = []
    raw_programs = []
    lengths = np.zeros(len(programs), int)
    # keep track of max_len to pad the sequences
    w2i = vocab['w2i']

    for i, prog in enumerate(tqdm(programs)):
        prog_tokens = tokenize_code(strip_comments(prog))
   
        if character_level:
            # this will break a string of tokens into 
            # many individual characters.
            prog_tokens = list(" ".join(prog_tokens))
        
        enc = [START_TOKEN] + prog_tokens + [END_TOKEN]
        
        # student program lengths might be longer than anything seen in training
        if len(enc) > max_len:
            print("WARNING: Skipping prgoram. max_len ({}) exceeded. Found len {}".format(max_len, len(enc)))
            continue

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


def process_student_data(problem):
    # TODO: fix this, it's outdated

    rnn_paths = paths.rnn_data_paths(problem, 'train', 'education', 'standard')
    vocab_paths = paths.vocab_paths(problem, 'education')

    if not os.path.isfile(vocab_paths['vocab_path']):
        raise ValueError('Run preprocessing script on rubric samples first to generate vocab file.')

    vocab, char_vocab, anon_vocab, anon_char_vocab = load_vocabs(vocab_paths)
    metadata = io.load_json(rnn_paths['metadata_path'])

    # load training max-lengths
    max_len = metadata['max_len']
    char_max_len = metadata['char_max_len']
    anon_max_len = metadata['anon_max_len']
    anon_char_max_len = metadata['anon_char_max_len']

    programs, anon_programs = raw_student_data(problem)

    feat_programs, program_lengths, raw_programs = featurise_programs_rnn(programs, vocab, max_len)
    char_feat_programs, char_program_lengths, _ = featurise_programs_rnn(
        programs, char_vocab, char_max_len, character_level=True)

    anon_feat_programs, anon_program_lengths, anon_raw_programs = featurise_programs_rnn(
        anon_programs, anon_vocab, anon_max_len)
    anon_char_feat_programs, anon_char_program_lengths, _ = featurise_programs_rnn(
        anon_programs, anon_char_vocab, anon_char_max_len, character_level=True)

    program_mats = dict(programs=feat_programs, lengths=program_lengths)
    char_program_mats = dict(programs=char_feat_programs, lengths=char_program_lengths)
    anon_program_mats = dict(programs=anon_feat_programs, lengths=anon_program_lengths)
    anon_char_program_mats = dict(programs=anon_char_feat_programs, lengths=anon_char_program_lengths)

    io.save_pickle(raw_programs, rnn_paths['raw_student_programs_path'])
    io.savemat(char_program_mats, rnn_paths['student_char_programs_path'])
    io.savemat(program_mats, rnn_paths['student_programs_path'])

    io.save_pickle(anon_raw_programs, rnn_paths['anon_raw_student_programs_path'])
    io.savemat(anon_char_program_mats, rnn_paths['anon_student_char_programs_path'])
    io.savemat(anon_program_mats, rnn_paths['anon_student_programs_path'])

    # io.save_np(feat_labels, rnn_paths['feat_labels_path'])
    # io.save_json(vocab, rnn_paths['vocab_path'])
    # io.save_json(label_mapping, rnn_paths['label_mapping_path'])


def main(problem):
    process_student_data(problem)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to preprocess')

    args = arg_parser.parse_args()
    main(args.problem)
