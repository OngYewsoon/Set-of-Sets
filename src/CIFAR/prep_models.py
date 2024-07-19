import gc
from multitask import multitaskMFEA
import pickle
import sys
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve
import config as C
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from model import CNN
import torchvision.models as models
from torchvision import datasets, transforms
import models.ResNet as ResNet
import traceback
import sys
from chromosome import Chromosome
from compare_solutions import *
from operators import *
from torch.autograd import Variable
from datetime import datetime
DEVICE_1 = torch.device('cuda:0')

def preload_batches(dataloader):
    batches = []
    for i, data in enumerate(dataloader):
        x, y = data[0].half().cuda(0), data[1].cuda(0)
        batches.append((x, y))
    return batches

def test(model, extloader):
    model.eval()
    avg_loss = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE_1)
    with torch.no_grad():
        for i, data in enumerate(extloader):
            x, y = data[0].cuda(0), data[1].cuda(0)
            x, y=Variable(x), Variable(y)
            fx = model(x)
            loss = criterion(fx.squeeze(), y)
            avg_loss += loss.detach().item()
    avg_loss = avg_loss/len(extloader)
    return avg_loss

def test_and_update(model, mask, mask_sizing, extloader, save_path, stats_dict, j):

    base_lr = 0.05
    criterion = nn.CrossEntropyLoss().to(DEVICE_1)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    lmbda = lambda epoch: 0.95
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lmbda)
    best_acc = 0
    average_acc = 0
    epochs = 0
    best_acc = test(model, extloader)
    best_acc = best_acc
    size = count_active_params(model.state_dict()) * 16 / (1024*8)
    stats_dict.update({j:[best_acc, size]})

def size_mask(state_dict):
    total = 0
    mask_sizing = OrderedDict()
    for i in list(state_dict.keys()):
        shape = state_dict[i].shape
        if len(shape) > 1:
            if 'linear' not in i and 'layer' in i:
                size1 = shape[0]
                total += size1*4
                mask_sizing.update({i+'_1':int(size1)})
                mask_sizing.update({i+'_2':int(size1)})
                mask_sizing.update({i+'_3':int(size1)})
                mask_sizing.update({i+'_4':int(size1)})
    return mask_sizing

def size_mask_for_calculation(state_dict):
    total = 0
    mask_sizing = OrderedDict()
    for i in list(state_dict.keys()):
        shape = state_dict[i].shape
        if len(shape) > 1:
            if 'linear' not in i and 'layer' in i:
                size1 = shape[0]
                total += size1*4
                mask_sizing.update({i+'_1':[shape[0], int(shape[1]/4), shape[2], shape[3]]})
                mask_sizing.update({i+'_2':[shape[0], int(shape[1]/4), shape[2], shape[3]]})
                mask_sizing.update({i+'_3':[shape[0], int(shape[1]/4), shape[2], shape[3]]})
                mask_sizing.update({i+'_4':[shape[0], int(shape[1]/4), shape[2], shape[3]]})
    return mask_sizing


task_names = ['animals', 'vehicles']
model_arr = []
train_data_arr = []
#
for task_name in task_names:
    ext_path = './CIFAR/data/finetune/train/'+task_name+'/'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = datasets.ImageFolder(ext_path, transform=transform)
    train_data_arr.append(preload_batches(DataLoader(dataset, batch_size=128, shuffle=True)))


task_indices = {'animals':[3, 4, 5, 7], 'vehicles':[0, 1, 8, 9]}

for task_name in task_names:
    model = ResNet.Resnet(18, 4)
    checkpoint = torch.load('./CIFAR/models/base.pth')
    state = checkpoint['network']
    base_state = copy.deepcopy(model.state_dict())
    idx_arr = task_indices[task_name]
    for key in state:
        if 'linear.' not in key:
            base_state[key] = state[key]
        elif 'linear.' in key and 'weight' in key:
            base_params = base_state[key]
            for i, idx in enumerate(idx_arr):
                base_params[i] = state[key][idx]
            base_state[key] = base_params
    model.load_state_dict(base_state)
    model.update_backup()
    model = model.cuda(0)
    model_arr.append(model)

def count_active_params(state_dict):
    total = 0
    for i in state_dict:
        flattened = torch.flatten(state_dict[i])
        total += torch.count_nonzero(flattened)
    return total

a = datetime.now()

initial_generation = int(sys.argv[1])
C.set_is_quantized(0)
mask_dict = {}
for i, task_name in enumerate(task_names):
    trainloader = train_data_arr[i]
    model = model_arr[i]
    mask_sizing = size_mask(model.state_dict())
    stats_dict = {}
    save_path = './CIFAR/prepared_model_info/' + task_name + '/'
    if task_name not in mask_dict:
        mask_dict.update({task_name:{}})
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for j in tqdm(range(initial_generation)):
        temp = Chromosome()
        temp.initialize(18944, 0, 0.9)
        temp.collapse_prevention(mask_sizing)
        mask = temp.rnvec

        model.apply_mask(mask, mask_sizing)
        model.half()

        test_and_update(model, mask, mask_sizing, trainloader, save_path + str(j) + '.pth', stats_dict, j)
        model.revert_weights()

        curr = mask_dict[task_name]
        curr.update({j:mask})
        mask_dict.update({task_name:curr})


    with open(save_path + "stats_dict.pkl", 'wb') as f:
        pickle.dump(stats_dict, f)

with open('./CIFAR/prepared_model_info/' + "mask_dict.pkl", 'wb') as f:
    pickle.dump(mask_dict, f)
print('Time Elapsed')
print((datetime.now() - a).seconds)
