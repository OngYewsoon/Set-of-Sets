import numpy as np
import math
import torch
import torch.nn as nn
import config as C
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    #overwrite the forward function for pruning
    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
    #overwrite the forward function for pruning
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
