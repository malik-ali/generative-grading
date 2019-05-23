import os
import re
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint

import src.utils.paths as paths
import src.utils.io_utils as io

from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN
from globals import HEAD, BODY, TAIL

def fix_labels(label):
    '''
    label - input dictionary of labels for a program

    returns - the same label dictionary with only valid fields
    such as only keeping True/False rubric labels
    '''
    fixed_label = dict()
    for key, value in label.items():
        # remove weird key/value decision that shouldnt even be in data dictionary
        if not isinstance(value, list):
            fixed_label[key] = value
 
    return fixed_label

def is_bool(value):
    if isinstance(value, bool):
        return True

    # handles mysterious edge case of numpy boolean as value type (debug?!)
    return isinstance(value, np.generic) and isinstance(value.item(), bool)
        
def load_raw_rubric_data(counts_path, labels_path, rv_order_path, tiers_path, anon_mapping_path):
    counts = io.load_pickle(counts_path)
    labels = io.load_pickle(labels_path)
    anon_mapping = io.load_pickle(anon_mapping_path)
    tiers = io.load_pickle(tiers_path)
    rv_order = io.load_pickle(rv_order_path)
    programs = list(counts.keys())
    anon_programs = [anon_mapping[p] for p in programs]
    p_labels = [fix_labels(labels[p]) for p in programs]
    p_rvorders = [rv_order[p] for p in programs]
    p_tiers = [tiers[p] for p in programs]
    return programs, anon_programs, p_labels, p_rvorders, p_tiers, counts

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

            rv_idx = w2i[rv]
            # val = str(val)  # if we support chain, uncomment 
            # [0] to grab the keys not the probabilities
            val_idx = all_rvs[rv][0].index(val)
            out_labels[n][rv_idx] = val_idx 
            
    return out_labels

def featurise_rv_order(rv_order_i, rv_info):
    '''
        - rv_order_i: shape N x t_i list. Each entry is a (variable length) list 
                      of the rv decisions made, in the order they were made

        - rv_info is a dict of rv2i, i2rv, and rv -> num_categories 

        returns N x (T) list where the decision names are replaced with
        the unique rv id from the vocab w2i. T is the max possible length
        of a decision sequence. Everything is padding to be this length.
        Also a start and end token are added 
    '''    

    N = len(rv_order_i)
    w2i = rv_info['w2i']

    # Maybe get max rv_order seq length from all data
    # Alternatively, max length is bounded by total num rvs:
    max_len = len(w2i)
    
    lengths = np.zeros(N, int)
    all_rv_orders = []

    for i, rv_order in enumerate(rv_order_i):
        rv_order_filt = [rv for rv in rv_order if rv in w2i]
        if len(rv_order) != len(rv_order_filt):
            print('WARNING: RVs [{}] not in vocab. Ignoring'.format(set(rv_order) - set(rv_order_filt)))
            
        enc = rv_order_filt
        lengths[i] = len(enc)
        all_rv_orders.append(enc)

    # Pad all sequences to max_len
    for enc in all_rv_orders:
        if len(enc) < max_len:
            enc += [PAD_TOKEN for _ in range(max_len - len(enc))]
        assert len(enc) == max_len

    encodings = [[w2i[rv] for rv in rv_order] for rv_order in all_rv_orders]
    return np.stack(encodings, axis=0), lengths

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
    if word:
        ret.append(word)
    return ret

def get_max_length(programs, character_level=False):
    max_len = 0
    for prog in programs:
        prog_tokens = tokenize_code(strip_comments(prog))
        if character_level:
            # this will break a string of tokens into 
            # many individual characters.
            prog_tokens = list(" ".join(prog_tokens))
        enc = [START_TOKEN] + prog_tokens + [END_TOKEN]
        max_len = max(max_len, len(enc))
    return max_len

def featurise_programs_rnn(programs, vocab, max_len, character_level=False):
    code_tokens = []
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
        lengths[i] = len(enc)
        code_tokens.append(enc)

    # Pad all sequences to max_len
    for enc in code_tokens:
        if len(enc) < max_len:
            enc += [PAD_TOKEN for _ in range(max_len - len(enc))]
        assert len(enc) == max_len

    encodings = [[w2i.get(t, w2i[UNK_TOKEN]) for t in tokens] for tokens in code_tokens]

    return np.stack(encodings, axis=0), lengths

def get_merged_info(counts_paths, labels_paths, rv_order_paths, tiers_paths, anon_mapping_paths):
    max_lens = []
    char_max_lens = []
    anon_max_lens = []
    anon_char_max_lens = []

    n_shards = len(counts_paths)
    for i in range(n_shards):
        programs_i, anon_programs_i, labels_i, _, _, _ = load_raw_rubric_data(
            counts_paths[i], labels_paths[i], rv_order_paths[i], 
            tiers_paths[i], anon_mapping_paths[i])

        max_len_i = get_max_length(programs_i)
        char_max_len_i = get_max_length(programs_i, character_level=True)
        anon_max_len_i = get_max_length(anon_programs_i)
        anon_char_max_len_i = get_max_length(anon_programs_i, character_level=True)
        
        labels_i = list(l for s in labels_i for l in s)

        max_lens.append(max_len_i)
        char_max_lens.append(char_max_len_i)
        anon_max_lens.append(anon_max_len_i)
        anon_char_max_lens.append(anon_char_max_len_i)

    max_lens = (max(max_lens), max(char_max_lens), max(anon_max_lens), max(anon_char_max_lens))
    
    return max_lens

def create_rv_info(all_rvs):
    '''
        - all_rvs:  A dictionary from rv name to tuple of (choices, ps)
                    where choices is a list of possible values this rv
                    can take and ps are the respective prior probabilities of
                    each value.
    '''
    # only padding
    special_tokens = [PAD_TOKEN]
    w2i, i2w, num_categories = dict(), dict(), dict()

    for rv, (vals, _) in all_rvs.items():
        id_num = len(w2i)
        w2i[rv] = id_num
        i2w[id_num] = rv

        num_categories[rv] = len(vals)

    # need to 0 index all the rv indexes, so add special tokens at end
    for t in special_tokens:
        id_num = len(w2i)
        w2i[t] = id_num
        i2w[id_num] = t        

    return dict(w2i=w2i, i2w=i2w, num_categories=num_categories)  

def load_vocabs(vocab_paths):
    vocab = io.load_json(vocab_paths['vocab_path'])
    char_vocab = io.load_json(vocab_paths['char_vocab_path'])
    anon_vocab = io.load_json(vocab_paths['anon_vocab_path'])
    anon_char_vocab = io.load_json(vocab_paths['anon_char_vocab_path'])

    return vocab, char_vocab, anon_vocab, anon_char_vocab

def make_rnn_data(problem, split, domain='education', sampling_strategy='standard'):
    rnn_paths = paths.rnn_data_paths(problem, split, domain, sampling_strategy)
    vocab_paths = paths.vocab_paths(problem, domain)
    os.makedirs(rnn_paths['data_path'], exist_ok=True)

    (counts_paths, labels_paths, rv_order_paths, 
     tiers_paths, anon_mapping_paths, all_rvs_path) = \
         paths.raw_data_paths(problem, split, domain, sampling_strategy)
    n_shards = len(counts_paths)

    # get info that has to be collected across all shards
    max_lens = get_merged_info(counts_paths, labels_paths, rv_order_paths, 
                               tiers_paths, anon_mapping_paths)
    vocab, char_vocab, anon_vocab, anon_char_vocab = load_vocabs(vocab_paths)
    max_len, char_max_len, anon_max_len, anon_char_max_len = max_lens

    all_rvs = io.load_json(all_rvs_path)
    rv_info = create_rv_info(all_rvs)
    # save all_rvs into rv_info
    rv_info['values'] = all_rvs

    data_len = 0
    shard_size = 0

    for i in range(n_shards):
        programs_i, anon_programs_i, labels_i, rv_order_i, tiers_i, _ = load_raw_rubric_data(
            counts_paths[i], labels_paths[i], rv_order_paths[i], tiers_paths[i], anon_mapping_paths[i])

        # assumes equally sized shards (except smaller remaining last one)
        shard_size = max(shard_size, len(programs_i))
        data_len += len(programs_i)

        feat_labels_i = featurise_labels(labels_i, rv_info, all_rvs)
        feat_rv_order_i, rv_order_lengths_i = featurise_rv_order(rv_order_i, rv_info)

        feat_programs_i, program_lengths_i = featurise_programs_rnn(programs_i, vocab, max_len)
        anon_feat_programs_i, anon_program_lengths_i = \
            featurise_programs_rnn(anon_programs_i, anon_vocab, anon_max_len)
        
        char_feat_programs_i, char_program_lengths_i = \
            featurise_programs_rnn(programs_i, char_vocab, char_max_len, character_level=True)

        anon_char_feat_programs_i, anon_char_program_lengths_i = \
            featurise_programs_rnn(anon_programs_i, anon_char_vocab, anon_char_max_len, character_level=True)

        program_mats_i = dict(  programs=feat_programs_i, 
                                lengths=program_lengths_i,
                                tiers=tiers_i)
        char_program_mats_i = dict( programs=char_feat_programs_i, 
                                    lengths=char_program_lengths_i,
                                    tiers=tiers_i)
        anon_program_mats_i = dict( programs=anon_feat_programs_i, 
                                    lengths=anon_program_lengths_i,
                                    tiers=tiers_i)
        anon_char_program_mats_i = dict(programs=anon_char_feat_programs_i, 
                                        lengths=anon_char_program_lengths_i,
                                        tiers=tiers_i)
        rv_order_mats_i = dict( rv_orders=feat_rv_order_i, 
                                lengths=rv_order_lengths_i)
 
        io.save_pickle(programs_i, rnn_paths['raw_programs_path'].format(i))
        io.savemat(program_mats_i, rnn_paths['feat_programs_path'].format(i))
        io.savemat(char_program_mats_i, rnn_paths['char_feat_programs_path'].format(i))
        
        # TODO: save raw labels in raw_labels_path
        io.save_np(feat_labels_i, rnn_paths['feat_labels_path'].format(i))
        io.save_pickle(anon_programs_i, rnn_paths['anon_raw_programs_path'].format(i))
        io.savemat(anon_program_mats_i, rnn_paths['anon_feat_programs_path'].format(i))
        io.savemat(anon_char_program_mats_i, rnn_paths['anon_char_feat_programs_path'].format(i))
        io.save_pickle(rv_order_i, rnn_paths['raw_rvOrder_path'].format(i))
        io.savemat(rv_order_mats_i, rnn_paths['feat_rvOrder_path'].format(i))
  
    io.save_json(rv_info, rnn_paths['rv_info_path'])
    
    metadata = dict(
        max_len=max_len,
        char_max_len=char_max_len,
        anon_max_len=anon_max_len,
        anon_char_max_len=anon_char_max_len,
        data_len=data_len,
        num_shards=n_shards,
        shard_size=shard_size
    )

    io.save_json(metadata, rnn_paths['metadata_path'])


def main(problem, split, domain, sampling_strategy):
    make_rnn_data(problem, split, domain, sampling_strategy)


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
        '--domain',
        default='education',
        help='education|postagging [default: education]')
    arg_parser.add_argument(
        '--sampling-strategy',
        default='standard',
        help='How we sample from the grammar (standard|uniform|tempered) [default: standard]')
    args = arg_parser.parse_args()
    main(args.problem, args.split, args.domain, args.sampling_strategy)
