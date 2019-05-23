import torch
import torch.nn as nn


class SoftmaxCrossEntropy(nn.Module):
    '''
        Used to compute softmax 
    '''
    def __init__(self):
        super().__init__()
        
        # expects log probabilities
        self.loss = nn.CrossEntropyLoss
        

    def forward(self, recon_label, labels):
        '''
            recon_labels: Shape N x C
            labels: Shape N
        '''
        return self.loss(recon_label, labels)
