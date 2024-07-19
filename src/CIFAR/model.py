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


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.weights_backup = copy.deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)

    def apply_mask(self, mask, sizing):
        start = 0
        copy_state = copy.deepcopy(self.model.state_dict())
        segments = {}
        for i in copy_state:
            if i in sizing:
                end = start + sizing[i]
                segment = np.round(mask[start:end])
                index = np.where(segment == 0)
                copy_state[i].data[index] = 0
                segments.update({i:index})
                start = end
        self.model.load_state_dict(copy_state)
        for name, param in self.model.named_parameters():
            if name in segments:
                param.data[segments[name]].requires_grad = False
                start = end

    def return_model(self):
        return self.model

    def return_model_state(self):
        return self.model.state_dict()

    def revert_weights(self):
        self.model.load_state_dict(self.weights_backup)
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def update_backup(self):
        self.weights_backup = copy.deepcopy(self.model.state_dict())
