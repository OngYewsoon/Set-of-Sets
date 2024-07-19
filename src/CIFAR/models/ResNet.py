import numpy as np
import math
import random
import torch
import torch.nn as nn
import compression as R
import torch.nn.functional as F
import copy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = R.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 =  R.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                 R.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 =  R.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear =  R.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def apply_mask(self, mask, sizing):
        start = 0
        copy_state = copy.deepcopy(self.state_dict())
        segments = {}
        for i in sizing:
            base, seg_idx = i.split('_')[0], int(i.split('_')[1])
            quart_size = int(copy_state[base].shape[1]/4)
            end = start + sizing[i]
            segment = np.round(mask[start:end])
            index = np.where(segment == 0)
            if seg_idx == 1:
                copy_state[base].data[index, 0:quart_size] = 0
                # print("------------")
                # print(copy_state[base].data.shape)
                # print(copy_state[base].data[index].shape)
                # print(copy_state[base].data[index, 0:quart_size].shape)
            elif seg_idx == 2:
                copy_state[base].data[index, quart_size:quart_size*2] = 0
            elif seg_idx == 3:
                copy_state[base].data[index, quart_size*2:quart_size*3] = 0

            elif seg_idx == 4:
                copy_state[base].data[index, quart_size*3:] = 0
            if base not in segments:
                segments.update({base:{seg_idx:index}})
            else:
                curr = segments[base]
                curr.update({seg_idx:index})
                segments.update({base:curr})
            # else:

            start = end

        self.load_state_dict(copy_state)
        self.segments = segments

    def half(self):

        for name, param in self.named_parameters():
            param.data = param.data.half()
            # if name in self.segments:
            #     quart_size = int(param.shape[1]/4)
            #     param.data[self.segments[name][1], 0:quart_size].grad.zero()
            #     param.data[self.segments[name][2], quart_size:quart_size*2].grad.zero()
            #     param.data[self.segments[name][3], quart_size*2:quart_size*3].grad.zero()
            #     param.data[self.segments[name][4], quart_size*3:].grad.zero()

    # def set_train(self):
    #     for name, param in self.named_parameters():
    #         param.data.requires_grad = True
    #         if name in self.segments:
    #             quart_size = int(param.shape[1]/4)
    #             param.data[self.segments[name][1], 0:quart_size].requires_grad = False
    #             param.data[self.segments[name][2], quart_size:quart_size*2].requires_grad = False
    #             param.data[self.segments[name][3], quart_size*2:quart_size*3].requires_grad = False
    #             param.data[self.segments[name][4], quart_size*3:].requires_grad = False
                # start = end

    def return_model_state(self):
        return self.state_dict()

    def revert_weights(self):
        self.load_state_dict(self.weights_backup)
        for name, param in self.named_parameters():
            param.requires_grad = True

    def update_backup(self):
        self.weights_backup = copy.deepcopy(self.state_dict())


def Resnet(type_id, num_classes):
    if(type_id==18):  net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    if(type_id==34):  net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    if(type_id==50):  net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    if(type_id==101):  net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    if(type_id==152):  net = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    return net
