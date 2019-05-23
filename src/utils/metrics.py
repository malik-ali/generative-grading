import torch
import numpy as np

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count

    @property
    def val(self):
        return self.value

    @property
    def avg(self):
        return self.average


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def update(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    @property
    def stop(self):
        return self.early_stop


def bc_accuracy(preds, target, counts_only=False):
    """
    Binary classification accuracy for multiple independent labels

    Inputs:
        - preds: Tensor of shape (N x num_labels)
        - target: Tensor of same shape as preds with 1./0. in each position

    Returns:
        - accuracy: Tensor of shape (num_labels) with average accuracy per label

    """
    batch_size = target.size(0)
    correct = (preds == target)

    # accuracy for each independent label
    accuracy = torch.sum(correct, dim=0).to(dtype=torch.float)
    if not counts_only:
        accuracy /= batch_size

    return accuracy


def tier_bc_counts(preds, targets, tiers):
    """
    Binary classification accuracy for mutiple independent labels
    split by (head, body, tail).
    """
    preds_npy = preds.cpu().detach().numpy()
    targets_npy = targets.cpu().detach().numpy()
    tiers_npy = tiers.cpu().detach().numpy()
    n_targets = targets_npy.shape[1]
    cnt_tier = np.zeros((3, n_targets))
    for tier in [0, 1, 2]:
        selection = tiers_npy == tier
        if np.sum(selection) == 0:
            continue
        preds_iter = torch.from_numpy(preds_npy[selection])
        targets_iter = torch.from_numpy(targets_npy[selection])
        acc = bc_accuracy(preds_iter, targets_iter)
        cnt = acc * len(preds_iter)
        cnt_tier[tier] = cnt
    
    cnt_tier = np.round(cnt_tier).astype(np.int)
    return cnt_tier

