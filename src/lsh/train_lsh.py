import os
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from datasketch import MinHashLSHForest, MinHash
from src.utils.codeorg_utils import labelInfo
from src.rubricsampling.engineConditioned import EngineConditioned
from sklearn.metrics import f1_score


def most_common(array):
    data = Counter(array)
    return data.most_common()[0][0]


def vectorize_labels(labels, n_labels, label2ix):
    vector = np.zeros(n_labels)
    for label in labels:
        if label in label2ix.keys():
            vector[label2ix[label]] = 1

    return vector


def get_citizenship_tiers(inputs):
    counter = Counter()
    for text in inputs:
        counter[text] += 1
    tiers = []
    for text in inputs:
        val = counter[text]
        # head = 0, tail = 1
        lab = 0 if (val > 1) else 1
        tiers.append(lab)
    tiers = np.array(tiers)
    return tiers


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset', type=str, help='codeorg|citizenship')
    arg_parser.add_argument('--k', type=int, default=5)
    args = arg_parser.parse_args()

    import getpass
    USER = getpass.getuser()


    if args.dataset == 'codeorg':
        fake_data_file = f'/mnt/fs5/{USER}/generative-grading/data/raw/codeorg9_ability_100k/tempered/labels_shard_0.pkl'
        real_data_file = f'/mnt/fs5/{USER}/generative-grading/data/real/education/codeorg9_ability_100k/raw/labels.pkl'
        real_zipf_file = f'/mnt/fs5/{USER}/generative-grading/data/real/education/codeorg9_ability_100k/raw/zipfs.pkl'
        with open(real_data_file, 'rb') as fp:
            real_data = pickle.load(fp)
        with open(real_zipf_file, 'rb') as fp:
            real_zipf = pickle.load(fp)
        ix2label = labelInfo.CODEORG9_IX_TO_LABEL
        label2ix = labelInfo.CODEORG9_LABEL_TO_IX
        n_labels = labelInfo.CODEORG9_N_LABELS
    elif args.dataset == 'citizenship':
        fake_data_file = f'/mnt/fs5/{USER}/generative-grading/data/raw/citizenship13_100k/tempered/labels_shard_0.pkl'
        real_data_file = f'/home/{USER}/generative-grading/src/rubricsampling/citizenshipRawData/studentanswers_grades_698.tsv'
        df = pd.read_csv(real_data_file, sep='\t')
        df = df[df['Q#'] == 13]
        real_ans = df['answer']
        real_ans = np.asarray(real_ans)
        real_zipf = get_citizenship_tiers(real_ans)
        G1 = np.asarray(df['G1'])
        G2 = np.asarray(df['G2'])
        G3 = np.asarray(df['G3'])
        real_lab = np.vstack([G1, G2, G3]).T
        real_lab = np.sum(real_lab, axis=1) >= 2
        real_lab = real_lab.astype(np.int)
        real_data = dict(zip(real_ans, real_lab))
        real_zipf = dict(zip(real_ans, real_zipf))

    with open(fake_data_file, 'rb') as fp:
        string2label = pickle.load(fp)

    string2hash = {}
    pbar = tqdm(total=len(string2label))
    for program, _ in string2label.items():
        tokens = program.split()
        minhash = MinHash()
        for token in tokens:
            minhash.update(token.encode('utf-8'))
        string2hash[program] = minhash
        pbar.update()
    pbar.close()

    forest = MinHashLSHForest()
    pbar = tqdm(total=len(string2hash))
    for program, minhash in string2hash.items():
        forest.add(program, minhash)
        pbar.update()
    pbar.close()
    forest.index()

    true_labels = []
    pred_labels = []
    zipf_labels = []
    for program, label in real_data.items():
        zipf = real_zipf[program]
        try:
            tokens = program.split()
        except:
            continue
        minhash = MinHash()
        for token in tokens:
            minhash.update(token.encode('utf-8'))
        result = forest.query(minhash, args.k)
        if len(result) == 0:
            continue
        lset = []
        for r in result:
            l = string2label[r]
            if args.dataset == 'codeorg':
                GRAMMAR_DIR = 'src/rubricsampling/grammars/codeorg9_ability'
                inf_e = EngineConditioned(GRAMMAR_DIR, l, choice_style='standard')
                _program, _labels, _decisions, _, _ = inf_e.renderProgram()
                _labels = vectorize_labels(_labels, n_labels, label2ix)
                lset.append(_labels)
            elif args.dataset == 'citizenship':
                l = int(l['correctStrategy'])
                lset.append(l)
        if args.dataset == 'codeorg':
            lst = np.vstack(lset)
            pred = [most_common(lst[:, i]) for i in range(n_labels)]
            pred = np.array(pred)
        else:
            lset = np.array(lset)
            pred = most_common(lset)
        true_labels.append(label)
        pred_labels.append(pred)
        zipf_labels.append(zipf)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.astype(np.int)
    zipf_labels = np.array(zipf_labels)

    if args.dataset == 'codeorg':
        body_f1 = np.zeros(n_labels)
        tail_f1 = np.zeros(n_labels)
        for i in range(n_labels):
            body_f1_i = f1_score(true_labels[zipf_labels == 1, i],
                                 pred_labels[zipf_labels == 1, i])
            tail_f1_i = f1_score(true_labels[zipf_labels == 2, i],
                                 pred_labels[zipf_labels == 2, i])
            body_f1[i] = body_f1_i
            tail_f1[i] = tail_f1_i
        print(np.mean(body_f1), np.mean(tail_f1))
    else:
        tail_acc = np.mean(true_labels[zipf_labels == 1] ==
                           pred_labels[zipf_labels == 1])
        avg_f1 = f1_score(true_labels, pred_labels)
        print(avg_f1, tail_acc)

