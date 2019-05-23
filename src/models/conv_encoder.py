from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torchvision import models, transforms

import numpy as np
from src.models.resnet_models import ResNet18
from src.models.vgg_models import vgg11, vgg19


class ConvImageEncoder(nn.Module):
    def __init__(self, num_classes, input_size=64, dropout=0):
        super().__init__()
        self.model = vgg11(pretrained=False, num_classes=num_classes)

    def forward(self, X):
        return self.model(X)

    
# -----------


class ConvImageEncoder_Old(nn.Module):
    def __init__(self, num_classes, input_size=64, dropout=0):
        super().__init__()
        in_channels = 3
        out_channels = 8
        #self.max1 = nn.MaxPool2d(4,stride = 4)
        self.max1 = nn.MaxPool2d(8, stride = 8, padding = 4)
        self.max2 = nn.MaxPool2d(4, stride=4, padding = 2)
        self.cov1 = nn.Conv2d(3,8,4,stride = 1, padding = 2)
        self.cov2 = nn.Conv2d(8, 16, 2 , stride = 1, padding=0)
        self.fc1 = nn.Linear(144, num_classes)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init 
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.cov1.weight)
        nn.init.xavier_uniform_(self.cov2.weight)

    def flatten(self, X):
        N = X.shape[0] # read in N, C, H, W
        return X.view(N, -1) # "flatten" the C * H * W values into a single vector per image

    def forward(self, X):
        L1 = self.max1(F.relu(self.cov1(X)))
        L2 = self.max2(F.relu(self.cov2(L1))) 
        flattened = self.flatten(L2)
        return self.fc1(flattened)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size=256, fix_pretrained=True, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.fix_pretrained = fix_pretrained
        self.device = device if device else torch.device('cpu')

        self.model = self._create_model()
        self.to(device=device)

    def params_to_optimise(self):
        if self.fix_pretrained:
            return [param for name, param in self.model.named_parameters() if param.requires_grad == True]
        else:
            return self.model.parameters()
        
    def _create_model(self):
        model = models.resnet101(pretrained=True)
        self.set_parameter_requires_grad(model, self.fix_pretrained)
        
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, self.hidden_size)

        self.input_size = 224
        return model
 

    def set_parameter_requires_grad(self, model, fix_pretrained):
        if fix_pretrained:
            for param in model.parameters():
                param.requires_grad = False
        

    def forward(self, inputs, return_hiddens=None):
        '''
          inputs:   Input images of shape  n x (3 x self.input_size x sel.input_size)
        '''
        return self.model(inputs)


if __name__ == "__main__":
    device = torch.device("cuda")
    torch.cuda.set_device(8)

    def make_stub_data(num_batches):
        stub_x = torch.from_numpy(
            np.random.random(size=(num_batches, 3, 224, 224))
        ).to(device=device, dtype=torch.float)
        
        return stub_x
    

    output_size = 256
    num_batches = 15

    enc = ImageEncoder(output_size, device=device)
    
    X = make_stub_data(num_batches)
    y = enc(X)

    print(y.size())
    assert len(enc.params_to_optimise()) == 2
    assert y.size() == torch.Size((num_batches, output_size))
