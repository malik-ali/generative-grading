r"""Clone of eval_nn_agents.py but we compute accuracy for Codeorg."""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.datasets.student_programs import CodeOrgStudentPrograms
import src.utils.paths as paths
from src.utils.codeorg_utils import labelInfo
from src.utils.io_utils import save_json

from inference_nn import InferenceNN
from sklearn.metrics import f1_score


class NNPredictionEval:
    def __init__(self, problem, exp_dir, strategy='map'):
        self.grammar_dir = paths.grammar_path(problem)
        # map = take most likely prediction
        self.inference_engine = InferenceNN(self.grammar_dir, exp_dir, strategy='map')
        self.config = self.inference_engine.getModelConfig()
        self.student_data = CodeOrgStudentPrograms(
            self.config['problem'],
            # load from the trained model to see if we need this
            character_level=self.config['character_level'], 
            include_anonymized=self.config['include_anonymized'])
        self.dataloader = DataLoader(self.student_data, batch_size=1, shuffle=False)
        
        if 'codeorg1' in problem:
            self.isCodeOrg1 = True
            self.isCodeOrg9 = False
        else:
            self.isCodeOrg1 = False
            self.isCodeOrg9 = True

    def evaluateAgent(self):
        tqdm_batch = tqdm(self.dataloader, total=len(self.student_data))

        true_labels_list = []
        pred_labels_list = []
        zipf_list = []

        nearest_neighbours = dict()

        if self.isCodeOrg1:
            ix2label = labelInfo.CODEORG1_IX_TO_LABEL
            label2ix = labelInfo.CODEORG1_LABEL_TO_IX
            n_labels = labelInfo.CODEORG1_N_LABELS
            scores = np.zeros((n_labels, 3))  # head,body,tail
            counts = np.zeros(3)
        else:
            ix2label = labelInfo.CODEORG9_IX_TO_LABEL
            label2ix = labelInfo.CODEORG9_LABEL_TO_IX
            n_labels = labelInfo.CODEORG9_N_LABELS
            scores = np.zeros((n_labels, 3))  # head,body,tail
            counts = np.zeros(3)

        def vectorize_labels(labels):
            vector = np.zeros(n_labels)
            for label in labels:
                if label in label2ix.keys():
                    vector[label2ix[label]] = 1
            
            return vector

        for i, data_list in enumerate(tqdm_batch):
            if self.config['include_anonymized']:
                program_args, raw_program, _, true_labels, zipf_labels = \
                    data_list[:-4], data_list[-4], data_list[-3], data_list[-2], data_list[-1]
            else:
                program_args, raw_program, true_labels, zipf_labels = \
                    data_list[:-3], data_list[-3], data_list[-2], data_list[-1]

            assert len(raw_program) == 1
            raw_program = raw_program[0]
            true_labels = true_labels[0]
            zipf_labels = zipf_labels[0]
            true_labels = true_labels.cpu().numpy()
            zipf_labels = zipf_labels.item()  # should just be one value
            zipf_labels = int(zipf_labels)

            try:
                nn, pred_labels, _ = self.inference_engine.guided_sample(
                    program_args, return_labels=True)
            except Exception as e:
                # its possible that really long/strange programs will require more 
                # depth in random variable recursion than we support. 
                # -- in such cases, ignore them.
                if 'Choice name' in str(e):
                    continue
                raise e

            nearest_neighbours[raw_program] = nn

            pred_labels = vectorize_labels(pred_labels)
            pred_labels = pred_labels.astype(np.int)
            
            accuracy = pred_labels == true_labels
            accuracy = accuracy.astype(np.int)

            scores[:, zipf_labels] = scores[:, zipf_labels] + accuracy
            counts[zipf_labels] = counts[zipf_labels] + 1

            true_labels_list.append(true_labels)
            pred_labels_list.append(pred_labels)
            zipf_list.append(zipf_labels)

        true_labels_list = np.vstack(true_labels_list)
        pred_labels_list = np.vstack(pred_labels_list)
        zipf_list = np.array(zipf_list)

        head_accuracy = scores[:, 0] / counts[0]
        body_accuracy = scores[:, 1] / counts[1]
        tail_accuracy = scores[:, 2] / counts[2]
        avg_accuracy  = np.sum(scores, axis=1) / np.sum(counts)

        def compute_f1_scores(predArr, trueArr):
            f1_scores = np.zeros(n_labels)
            for i in range(n_labels):
                f1_scores[i] = f1_score(trueArr[:, i], predArr[:, i])
            return f1_scores

        head_f1 = compute_f1_scores(pred_labels_list[zipf_list == 0],
                                    true_labels_list[zipf_list == 0])
        body_f1 = compute_f1_scores(pred_labels_list[zipf_list == 1],
                                    true_labels_list[zipf_list == 1])
        tail_f1 = compute_f1_scores(pred_labels_list[zipf_list == 2],
                                    true_labels_list[zipf_list == 2])
        avg_f1 = compute_f1_scores(pred_labels_list, true_labels_list)

        head_acc_dict, head_f1_dict = {}, {}
        body_acc_dict, body_f1_dict = {}, {}
        tail_acc_dict, tail_f1_dict = {}, {}
        avg_acc_dict, avg_f1_dict   = {}, {}
        
        for i in range(n_labels):
            head_acc_dict[ix2label[i]] = float(head_accuracy[i])
            body_acc_dict[ix2label[i]] = float(body_accuracy[i])
            tail_acc_dict[ix2label[i]] = float(tail_accuracy[i])
            avg_acc_dict[ix2label[i]]  = float(avg_accuracy[i])

            head_f1_dict[ix2label[i]] = float(head_f1[i])
            body_f1_dict[ix2label[i]] = float(body_f1[i])
            tail_f1_dict[ix2label[i]] = float(tail_f1[i])
            avg_f1_dict[ix2label[i]]  = float(avg_f1[i])

        return {
            'head_acc': head_acc_dict,
            'body_acc': body_acc_dict,
            'tail_acc': tail_acc_dict,
            'avg_acc': avg_acc_dict,

            'head_f1': head_f1_dict,
            'body_f1': body_f1_dict,
            'tail_f1': tail_f1_dict,
            'avg_f1': avg_f1_dict,

            'nns': nearest_neighbours
        }


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to evaluate e.g. codeorg1')
    arg_parser.add_argument(
        'exp_dir',
        type=str,
        help='where model is saved')
    args = arg_parser.parse_args()

    agent = NNPredictionEval(args.problem, args.exp_dir)
    scores = agent.evaluateAgent()

    results_dir = os.path.join(args.exp_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir, 'student_accuracy_dict.pickle')

    with open(results_file, 'wb') as fp:
        pickle.dump(scores, fp)

