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
import traceback
import sys


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from model import Model, Custom_Dataloader
import torchvision.models as models
from torchvision import datasets, transforms
from chromosome import Chromosome
from compare_solutions import *
from operators import *


DEVICE_1 = torch.device('cuda:0')
DEVICE_2 = torch.device('cpu')

run_name = sys.argv[1]
pretrained = int(sys.argv[2])
rmp = float(sys.argv[3])
final_finetune_switch = float(sys.argv[4])
MFEA_II = int(sys.argv[5])
finetune_epochs = int(sys.argv[6])

def is_pareto_efficient(lim1, processed):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    # processed = []
    # for i in results[task]:
    #     processed.append([lim1-results[task][i][0], results[task][i][1]])
    costs = np.array(processed)
    indices = np.array([i for i in range(len(costs))])

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
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


def size_mask(state_dict):
    total = 0
    mask_sizing = OrderedDict()
    total_params = 0
    total_considered = 0

    size = 0
    for i in list(state_dict.keys()):
        shape = torch.tensor(state_dict[i].shape)
        total_params += torch.prod(shape)
        if 'bias' not in i and 'embed' not in i and 'norm' not in i and 'shared' not in i and 'head' not in i:
            if shape[0] == 4096:
                total += 1024
                mask_sizing.update({i:1024})
            else:
                total += 256
                mask_sizing.update({i:256})
            total_considered += torch.prod(shape)
    print(total_params)
    print(total_considered)
    print(total)
    return mask_sizing

def dual_sample(ds1_batch, ds2_batch):
    perm = torch.randperm(len(ds1_batch['x']['input_ids'])*2)
    x_input_ids = torch.cat((ds1_batch['x']['input_ids'], ds2_batch['x']['input_ids']), 0)
    x_attention_masks = torch.cat((ds1_batch['x']['attention_mask'], ds2_batch['x']['attention_mask']), 0)
    y = torch.cat((ds1_batch['y'], ds2_batch['y']))

    return {'x':{'input_ids': x_input_ids[perm],
                 'attention_mask': x_attention_masks[perm]},
            'y':y[perm]}

def test_loss(model, test_dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            loss += model.forward_train(batch).loss
    return loss/len(test_dataloader)

def prep_dataloader(path, batch_size):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return Custom_Dataloader(data, batch_size)

def preload_batches(dataloader):
    batches = []
    for i in range(dataloader.length):
        batch = dataloader.sample_batch()
        batches.append(batch)
    print("==========")
    print(dataloader.length)
    return batches

def load_model():
    model = Model()
    model.load_state_dict(torch.load('./NLP_M2M/MFEA/models/base.pth'))
    model.update_backup()
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            param.requires_grad =  True
    model = model.cuda()
    return model

data_paths = [
            './NLP_M2M/MFEA/data/cs_evo.pkl',
            './NLP_M2M/MFEA/data/de_evo.pkl',
            ]

train_data_arr = [preload_batches(prep_dataloader(data_paths[i], 1000)) for i in range(len(data_paths))]

data_paths = [
            './NLP_M2M/MFEA/data/cs_evo_test.pkl',
            './NLP_M2M/MFEA/data/de_evo_test.pkl',
            ]

# train_data_arr_unloaded = [prep_dataloader(data_paths[i], 64) for i in range(len(data_paths))]
train_data_arr_unloaded = []
data_paths = [
            './NLP_M2M/MFEA/data/cs_evo_test.pkl',
            './NLP_M2M/MFEA/data/de_evo_test.pkl',
            ]

# test_data_arr = [preload_batches(prep_dataloader(data_paths[i], 64)) for i in range(len(data_paths))]
test_data_arr = []
all_progenitors = []

task_names = ['cs_only', 'de_only']
with open('./NLP_M2M/MFEA/prepared_model_info/mask_dict.pkl', 'rb') as f:
    progenitor_masks = pickle.load(f)

if pretrained == 1:
    hyperparams = {'gen':120, 'subpop':20, 'sbxdi':10, 'pmdi':10}
    print('using pretrained')
    for i, task_name in enumerate(task_names):
        with open('./NLP_M2M/MFEA/prepared_model_info/' +task_name + '/stats_dict.pkl', 'rb') as f:
            stats_dict = pickle.load(f)
        accuracies = [stats_dict[i][0] for i in stats_dict]
        sizes = [stats_dict[i][1]for i in stats_dict]
        processed = [(accuracies[i].cpu(), sizes[i].cpu()) for i,item in enumerate(accuracies)]
        _, _, indices = is_pareto_efficient(100, processed)
        # print([(accuracies[i].cpu(), sizes[i].cpu()) for i,item in enumerate(accuracies)])
        print(indices)
        progenitors = [progenitor_masks[task_name][j] for j in indices][0:hyperparams['subpop']]
        print(progenitors)
        while len(progenitors) < hyperparams['subpop']:
            temp = Chromosome()
            temp.initialize(67584, 0.25, 0.95)
            progenitors.append(temp.rnvec)
        all_progenitors.append(progenitors)
        print(len(progenitors))
else:
    hyperparams = {'gen':2, 'subpop':10, 'sbxdi':10, 'pmdi':10}
    print('Not using pretrained')
    for i, task_name in enumerate(task_names):
        progenitors = []
        while len(progenitors) < hyperparams['subpop']:
            temp = Chromosome()
            temp.initialize(67584, 0.3, 0.95)
            progenitors.append(temp.rnvec)
        all_progenitors.append(progenitors)
        print(len(progenitors))
print(sys.argv)
print(pretrained)
print(len(all_progenitors))

model = load_model()
model.update_backup()

try:
    multitaskMFEA('./NLP_M2M/MFEA/results/MT/',
                run_name,
                model,
                train_data_arr,
                train_data_arr_unloaded,
                test_data_arr,
                hyperparams,
                all_progenitors,
                rmp,
                pretrained,
                final_finetune_switch, MFEA_II, finetune_epochs)

except Exception as e:
    print(traceback.format_exc())
