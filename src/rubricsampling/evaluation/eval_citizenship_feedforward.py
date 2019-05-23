r"""Clone of eval_nn_agents.py but we compute accuracy for Citizenship13."""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.agents.feedforward_nn import FeedforwardNN
from src.datasets.citizenship_labels import CitizenshipLabels
import src.utils.paths as paths
from src.utils.io_utils import save_json
from src.utils.setup import load_config

from inference_nn import InferenceNN
from sklearn.metrics import f1_score


class NNPredictionEval:
    def __init__(self, problem, exp_dir, strategy='map'):
        self.config = load_config(os.path.join(exp_dir, 'config.json'))
        self._load_model(exp_dir)

        self.strategy = strategy

        self.student_data = CitizenshipLabels(
            13,
            split='test',
            vocab=self.agent.train_dataset.vocab)
        self.dataloader = DataLoader(self.student_data, batch_size=1, shuffle=False)

    def _load_model(self, exp_dir):
        config = load_config(os.path.join(exp_dir, 'config.json'))
        # config['cuda'] = False
        config['gpu_device'] = 9
        self.agent = FeedforwardNN(config)
        self.agent.load_checkpoint('checkpoint.pth.tar')
        self.model = self.agent.model
        self.model.eval()
        self.config = config        
        # self.agent.validate()

    def _chooseOutputs(self, rvOutputList, rvOrders, rvOrders_lengths):
        label_info = rvOrders[:, 1:]
        outputs = [] 

        for n in range(label_info.shape[0]):
            true_len = rvOrders_lengths[n] - 1
            curr_order = label_info[n][:true_len]
            curr_order_npy = curr_order.cpu().numpy()

            outputs_n = [rvOutputList[i][n] for i in curr_order_npy]
            outputs.append(outputs_n)

        return outputs        

    def evaluateAgent(self):
        tqdm_batch = tqdm(self.dataloader, total=len(self.student_data))
        
        scores = np.zeros(3)  # head,body,tail
        counts = np.zeros(3)

        true_label_list = []
        pred_label_list = []

        nearest_neighbours = dict()

        def vectorize_labels(outputs):
            idx = self.agent.train_dataset.rv_info['w2i']['correctStrategy']    # same as incorrect answer
            output_i = outputs[idx][0].cpu().numpy()
            # TODO: add map vs not map
            rv_var = np.argmax(output_i)
            ret = self.agent.train_dataset.rv_info['values']['correctStrategy'][0][rv_var]
            return int(ret)

        with torch.no_grad():
            for i, data_list in enumerate(tqdm_batch):
                program_args, raw_program, true_labels, zipf_labels = \
                    data_list[:-3], data_list[-3], data_list[-2], data_list[-1]

                
                outputs = self.model(program_args)

                pred_labels = vectorize_labels(outputs)
                

                true_labels = true_labels[0]
                raw_program = raw_program[0]
                zipf_labels = zipf_labels[0]
                true_labels = true_labels.cpu().numpy()[0]
                zipf_labels = zipf_labels.item()  # should just be one value
                # nearest_neighbours[raw_program] = nn

                accuracy = pred_labels == true_labels
                scores[zipf_labels] = scores[zipf_labels] + accuracy
                counts[zipf_labels] = counts[zipf_labels] + 1

                true_label_list.append(true_labels)
                pred_label_list.append(pred_labels)
            
            head_accuracy = scores[0] / counts[0]
            body_accuracy = scores[1] / counts[1]
            tail_accuracy = scores[2] / counts[2]
            avg_accuracy  = float(np.sum(scores) / np.sum(counts))

            true_label_list = np.array(true_label_list)
            pred_label_list = np.array(pred_label_list)
            avg_f1 = f1_score(true_label_list, pred_label_list)

            print('AVERAGE f1: ', avg_f1)            

            return {
                'head_acc': head_accuracy,
                'body_acc': body_accuracy,
                'tail_acc': tail_accuracy,
                'avg_acc': avg_accuracy,
                'avg_f1': avg_f1,
                'nns': nearest_neighbours
            }


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'exp_dir',
        type=str,
        help='where model is saved')
    args = arg_parser.parse_args()

    agent = NNPredictionEval('citizenship13', args.exp_dir)
    scores = agent.evaluateAgent()

    results_dir = os.path.join(args.exp_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir, 'student_accuracy_dict.pickle')

    with open(results_file, 'wb') as fp:
        pickle.dump(scores, fp)

    print(scores)
