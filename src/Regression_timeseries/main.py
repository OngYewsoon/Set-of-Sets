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
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, DataLoader

from collections import OrderedDict
from model import Model
import traceback
import sys
from chromosome import Chromosome

DEVICE_1 = torch.device('cuda:0')

run_name = sys.argv[1]
pretrained = int(sys.argv[2])
rmp = float(sys.argv[3])
final_finetune_switch = float(sys.argv[4])
MFEA_II = int(sys.argv[5])
finetune_epochs = int(sys.argv[6])

def preload_batches(dataloader):
    batches = []
    for i, data in enumerate(dataloader):
        x, y = data[0].float().cuda(0), data[1].cuda(0)
        batches.append((x, y))
    return batches

def is_pareto_efficient(lim1, processed):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """

    costs = np.array(processed)
    # print(p)
    indices = np.array([i for i in range(len(costs))])
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        indices = indices[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    final_x = [i[0] for i in costs]
    final_y = [i[1] for i in costs]
    return final_x, final_y, indices

train_data_arr = []
train_data_arr_unloaded = []
test_data_arr = []
model_arr = []

def load(data_dict, place, batchsize):
    x = torch.tensor(data_dict[place]['x'])
    y = torch.tensor(data_dict[place]['y'])
    data_gen = TensorDataset(x, y)
    dataloader = DataLoader(data_gen, batch_size=batchsize, shuffle=True)
    return dataloader

with open('./Regression_timeseries/data/air_train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('./Regression_timeseries/data/air_test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

task_names = ['Aotizhongxin', 'Changping', 'Nongzhanguan', 'Dingling']

train_data_arr = []
#
for task_name in task_names:
    train_data_arr.append(preload_batches(load(train_data, task_name, 1024)))

for task_name in task_names:
    train_data_arr_unloaded.append(load(train_data, task_name, 1024))

for task_name in task_names:
    test_data_arr.append(preload_batches(load(test_data, task_name, 1024)))

hyperparams = {'gen':120, 'subpop':60, 'sbxdi':10, 'pmdi':10}

for task_name in task_names:
    model = Model()
    model = model.cuda(0)
    model.load_state_dict(torch.load('./Regression_timeseries/models/base.pth'))
    model.update_backup()
    model_arr.append(model)


all_progenitors = []


if pretrained == 1:
    print('Using pre-prepared progenitors')
    with open('./Regression_timeseries/prepared_model_info/mask_dict.pkl', 'rb') as f:
        progenitor_masks = pickle.load(f)
    for i, task_name in enumerate(task_names):
        with open('./Regression_timeseries/prepared_model_info/' +task_name + '/stats_dict.pkl', 'rb') as f:
            stats_dict = pickle.load(f)
        accuracies = [stats_dict[i][0] for i in stats_dict]
        sizes = [stats_dict[i][1] for i in stats_dict]
        processed = [(accuracies[i], sizes[i].cpu()) for i,item in enumerate(accuracies)]
        _, _, indices = is_pareto_efficient(100, processed)
        print(indices)
        progenitors = [progenitor_masks[task_name][j] for j in indices][0:hyperparams['subpop']]
        while len(progenitors) < hyperparams['subpop']:
            temp = Chromosome()
            temp.initialize(4096, 0.2, 0.99)
            progenitors.append(temp.rnvec)
        all_progenitors.append(progenitors)
        print(len(progenitors))
else:
    print('Not using pre-prepared progenitors')
    for i, task_name in enumerate(task_names):
        progenitors = []
        while len(progenitors) < hyperparams['subpop']:
            temp = Chromosome()
            temp.initialize(4096, 0.2, 0.99)
            progenitors.append(temp.rnvec)
        all_progenitors.append(progenitors)

try:
    multitaskMFEA('./Regression_timeseries/results/MT/',
                run_name,
                all_progenitors,
                model_arr,
                train_data_arr,
                train_data_arr_unloaded,
                test_data_arr,
                hyperparams,
                pretrained,
                rmp,
                final_finetune_switch, MFEA_II, finetune_epochs)
except Exception:
    print(traceback.format_exc())
