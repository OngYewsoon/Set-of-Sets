import gc
import pickle
import sys
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from model import Model
from torch.utils.data import TensorDataset, DataLoader
import traceback
import sys
from chromosome import Chromosome
from compare_solutions import *
from operators import *
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm

DEVICE_1 = torch.device('cuda:0')

def preload_batches(dataloader):
    batches = []
    for i, data in enumerate(dataloader):
        x, y = data[0].float().cuda(0), data[1].cuda(0)
        batches.append((x, y))
    return batches

def test_finetune(model, extloader):
    model.eval()
    avg_acc = 0
    criterion = nn.MSELoss().to(DEVICE_1)
    with torch.no_grad():
        for i, data in enumerate(extloader):
            x, y = data[0], data[1]
            x, y=Variable(x), Variable(y)
            fx = model(x)
            loss = torch.sqrt(criterion(fx.squeeze(), y))
            avg_acc += loss.detach().item()
    avg_acc = avg_acc/len(extloader)
    return avg_acc

def finetune(model, mask, mask_sizing, extloader, save_path, stats_dict, j):
    best_acc = 0
    average_acc = 0
    epochs = 0
    best_acc = test_finetune(model, extloader)
    best_acc = best_acc
    size = count_active_params(model.state_dict()) * 32 / (1024*8)
    stats_dict.update({j:[best_acc, size]})

def count_active_params(state_dict):
    total = 0
    for i in state_dict:
        flattened = torch.flatten(state_dict[i])
        total += torch.count_nonzero(flattened)
    return total

def size_mask(state_dict):
    total = 0
    mask_sizing = OrderedDict()
    for i in list(state_dict.keys()):
        shape = state_dict[i].shape
        if len(shape) > 1:
            if 'out' not in i:
                size1 = shape[0]
                total += size1
                mask_sizing.update({i:int(size1)})
    print(total)
    return mask_sizing

def load(data_dict, place, batchsize):
    x = torch.tensor(data_dict[place]['x'])
    y = torch.tensor(data_dict[place]['y'])
    data_gen = TensorDataset(x, y)
    dataloader = DataLoader(data_gen, batch_size=batchsize, shuffle=True)
    return dataloader

with open('./Regression_timeseries/data/air_train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

task_names = ['Aotizhongxin', 'Changping', 'Nongzhanguan', 'Dingling']
train_data_arr = []
#
for task_name in task_names:
    train_data_arr.append(preload_batches(load(train_data, task_name, 1024)))

a = datetime.now()
initial_generation = int(sys.argv[1])
mask_dict = {}
for i, task_name in enumerate(task_names):
    trainloader = train_data_arr[i]
    model = Model()
    model = model.cuda(0)
    model.load_state_dict(torch.load('./Regression_timeseries/models/base.pth'))
    model.update_backup()
    mask_sizing = size_mask(model.state_dict())
    stats_dict = {}
    save_path = './Regression_timeseries/prepared_model_info/' + task_name + '/'
    if task_name not in mask_dict:
        mask_dict.update({task_name:{}})
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(task_name)
    print('------------------------------------')
    for j in tqdm(range(initial_generation)):
        temp = Chromosome()
        temp.initialize(4096, 0.2, 0.99)
        temp.collapse_prevention(mask_sizing)
        mask = temp.rnvec
        model.apply_mask(mask, mask_sizing)
        finetune(model, mask, mask_sizing, trainloader, save_path + str(j) + '.pth', stats_dict, j)
        model.revert_weights()
        curr = mask_dict[task_name]
        curr.update({j:mask})
        mask_dict.update({task_name:curr})

    with open(save_path + "stats_dict.pkl", 'wb') as f:
        pickle.dump(stats_dict, f)

with open('./Regression_timeseries/prepared_model_info/' + "mask_dict.pkl", 'wb') as f:
    pickle.dump(mask_dict, f)
print('Time Elapsed')
print((datetime.now() - a).seconds)
