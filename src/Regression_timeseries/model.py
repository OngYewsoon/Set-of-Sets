import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import segmentation_models_pytorch as smp
from collections import OrderedDict
import numpy as np
import copy

import os
from collections import OrderedDict
import json
import time

import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
from torchinfo import summary

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.act1 = nn.Tanh()
        self.out_layer = nn.Linear(256, 1)


    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(4, batch_size, 128).requires_grad_().cuda()
        c0 = torch.zeros(4, batch_size, 128).requires_grad_().cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))

        return self.out_layer(self.act1(output))

    def apply_mask(self, mask, sizing):
        start = 0
        copy_state = copy.deepcopy(self.state_dict())
        segments = {}
        for i in sizing:
            end = start + sizing[i]
            segment = np.round(mask[start:end])
            index = np.where(segment == 0)
            copy_state[i].data[index] = 0
            start = end
        self.load_state_dict(copy_state)

    def revert_weights(self):
        self.load_state_dict(self.weights_backup)
        for name, param in self.named_parameters():
            param.requires_grad = True

    def update_backup(self):
        self.weights_backup = copy.deepcopy(self.state_dict())
