import os
from tqdm import tqdm
from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

from scripts.process_data import tokenize_code, strip_comments, load_raw_rubric_data
import src.utils.paths as paths
import src.utils.io_utils as io


def build_vocab_rnn(programs, character_level=False):
    r"""If character_level is true, we will create a vocabulary 
    based on individual characters. Otherwise, it will be created
    on a token-level granularity.
    """
    special_tokens = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    w2i, i2w = dict(), dict()

    for t in special_tokens:
        id_num = len(w2i)
        w2i[t] = id_num
        i2w[id_num] = t

    for i, prog in enumerate(tqdm(programs)):
        prog_tokens = tokenize_code(strip_comments(prog))
   
        if character_level:
            # this will break a string of tokens into 
            # many individual characters.
            prog_tokens = list(" ".join(prog_tokens))
        
        for t in prog_tokens:
            if t not in w2i:
                id_num = len(w2i)
                w2i[t] = id_num
                i2w[id_num] = t

    vocab = dict(w2i=w2i, i2w=i2w)
    return vocab


def make_vocabs(problem, domain='education'):
    vocab_paths = paths.vocab_paths(problem, domain)
    os.makedirs(vocab_paths['data_path'], exist_ok=True)

    all_programs, all_anon_programs = [], []

    for sampling_strategy in ['standard', 'uniform', 'tempered']:
        (counts_paths, labels_paths,rv_order_paths, 
         tiers_paths, anon_mapping_paths, all_rvs_path) = \
            paths.raw_data_paths(problem, 'train', domain, sampling_strategy)
        n_shards = len(counts_paths)

        for i in range(n_shards):
            programs_i, anon_programs_i, _, _, _, _ = load_raw_rubric_data(
                counts_paths[i], labels_paths[i], rv_order_paths[i],
                tiers_paths[i], anon_mapping_paths[i])

            all_programs.extend(programs_i)
            all_anon_programs.extend(anon_programs_i)

    vocab = build_vocab_rnn(all_programs, character_level=False)
    char_vocab = build_vocab_rnn(all_programs, character_level=True)
    anon_vocab = build_vocab_rnn(all_anon_programs, character_level=False)
    anon_char_vocab = build_vocab_rnn(all_anon_programs, character_level=True)

    io.save_json(vocab, vocab_paths['vocab_path'])
    io.save_json(char_vocab, vocab_paths['char_vocab_path'])
    io.save_json(anon_vocab, vocab_paths['anon_vocab_path'])
    io.save_json(anon_char_vocab, vocab_paths['anon_char_vocab_path'])


def main(problem, domain):
    make_vocabs(problem, domain)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to preprocess')
    arg_parser.add_argument(
        '--domain',
        default='education',
        help='education|postagging [default: education]')
    args = arg_parser.parse_args()
    main(args.problem, args.domain)
