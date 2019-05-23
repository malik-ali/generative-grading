import torch
import torch.nn as nn


class BinaryCrossEntropy(nn.Module):
    def __init__(self, logits=False):
        super().__init__()
        self.logits = logits

        if logits:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.BCELoss()


    def forward(self, recon_label, labels):
        return self.loss(recon_label, labels)