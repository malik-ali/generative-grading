r"""Clone of eval_nn_agents.py but we compute accuracy for Citizenship13."""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.datasets.pyramid import PyramidImages, PyramidGrammar
from src.models.conv_encoder import ConvImageEncoder, ConvImageEncoder_Old, ImageEncoder
from src.agents.pyramid_convnet import PyramidConvNet
import src.utils.paths as paths
from src.utils.io_utils import save_json
from src.utils.metrics import AverageMeter

from inference_nn import InferenceNN
from sklearn.metrics import f1_score



from src.utils.setup import load_config


GROUPING = {
    0: [0, 7, 14, 15],
    1: [1,2,3],
    2: [4,6],
    3: [5, 8, 9, 10],
    4: [11, 12, 13]
}


class NNPredictionEval:
    def __init__(self, exp_dir):
        self.config = load_config(os.path.join(exp_dir, 'config.json'))
        self._load_model(exp_dir)

        # self.test_dataset = PyramidImages(500, input_size=self.config.encoder_kwargs.input_size, split='train')
        self.test_dataset = PyramidImages(None, input_size=self.config.encoder_kwargs.input_size, split='test',
                                          knowledge_states=self.config.knowledge_states)
        self.group_lookup = self._group_lookup_array(GROUPING)
        self.dataloader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        self.n_labels = 5 if self.config.knowledge_states else 13

    def _group_lookup_array(self, grouping):
        arr = np.zeros(16, dtype=int)
        for group, items in grouping.items():
            for i in items:
                arr[i] = group
        return arr

    def _load_model(self, exp_dir):
        config = load_config(os.path.join(exp_dir, 'config.json'))
        # config['cuda'] = False
        config['gpu_device'] = 9
        self.agent = PyramidConvNet(config)
        self.agent.load_checkpoint('checkpoint.pth.tar')
        self.model = self.agent.model
        self.model.eval()
        self.config = config
        # self.agent.validate()

    def evaluateAgent(self):
        num_batches = len(self.test_dataset) // self.config.batch_size
        tqdm_batch = tqdm(self.dataloader, total=num_batches)

        loss_meter = AverageMeter()
        avg_acc_meter = AverageMeter()

        cm = np.zeros((self.n_labels, self.n_labels))
        if self.n_labels == 13:
            group_acc_meter = AverageMeter()
            cm_group = np.zeros((5, 5))

        accuracy_data = []

        for _, data_list in enumerate(tqdm_batch):
            X, (counts, y) = data_list

            X = X.to(device=self.agent.device, dtype=torch.float32)
            y = y.to(device=self.agent.device, dtype=torch.long)

            if self.n_labels == 13:
                valid_mask = y < 13     # not predcting labels after samp13
                X = X[valid_mask]
                y = y[valid_mask]
                counts = counts[valid_mask]

            batch_size = len(y)

            scores = self.model(X)
            loss = F.cross_entropy(scores, y)
            ll = torch.softmax(scores, 1)
            preds = torch.argmax(ll, 1)
            accuracy = torch.sum(preds == y).float().cpu().numpy()/y.size(0)

            # import pdb; pdb.set_trace()
            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            counts_np = counts.cpu().numpy()
            correct = preds_np == y_np

            if self.n_labels == 13:
                y_group_np = self.group_lookup[y_np]
                preds_group_np = self.group_lookup[preds_np]
                correct_group = preds_group_np == y_group_np
                group_accuracy = np.sum(preds_group_np == y_group_np).astype(float)/y_group_np.shape[0]

            if self.n_labels == 13:
                accuracy_data.extend(zip(correct, correct_group, counts_np))
            else:
                accuracy_data.extend(zip(correct, counts_np))

            cm[y_np, preds_np] += 1
            if self.n_labels == 13:
                cm_group[y_group_np, preds_group_np] += 1

            # write data and summaries
            loss_meter.update(loss.item(), n=batch_size)
            avg_acc_meter.update(accuracy, n=batch_size)

            if self.n_labels == 13:
                group_acc_meter.update(group_accuracy, n=batch_size)
                tqdm_batch.set_postfix({"Group acc": group_acc_meter.avg,
                                        "Avg acc": avg_acc_meter.avg})
            else:
                tqdm_batch.set_postfix({"Avg acc": avg_acc_meter.avg})

        print('AVERAGE ACCURACY: {}'.format(avg_acc_meter.avg))

        results = {
            'avg_acc': avg_acc_meter.avg,
            'confusion_matrix': cm,
            'accuracy_data': accuracy_data
        }
        if self.n_labels == 13:
            results['confusion_matrix_group'] = cm_group

        return results


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'exp_dir',
        type=str,
        help='where model is saved')
    args = arg_parser.parse_args()

    agent = NNPredictionEval(args.exp_dir)
    scores = agent.evaluateAgent()

    results_dir = os.path.join(args.exp_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir, 'pyramid_accuracy_dict.pickle')

    with open(results_file, 'wb') as fp:
        pickle.dump(scores, fp)

    print(scores)
