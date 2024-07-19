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

DEVICE_1 = torch.device('cuda:0')

# Get Hyperparams
run_name = sys.argv[1]
pretrained = int(sys.argv[2])
rmp = float(sys.argv[3])
final_finetune_switch = float(sys.argv[4])
MFEA_II = int(sys.argv[5])
finetune_epochs = int(sys.argv[6])

def preload_batches(dataloader):
    batches = []
    for i, data in enumerate(dataloader):
        x, y = data[0].half().cuda(0), data[1].cuda(0)
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

task_names = ['animals', 'vehicles']
train_data_arr = []
train_data_arr_unloaded = []
test_data_arr = []
model_arr = []
for task_name in task_names:
    ext_path = './CIFAR/data/train/'+task_name+'/'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p=0.5),

                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = datasets.ImageFolder(ext_path, transform=transform)
    train_data_arr_unloaded.append(DataLoader(dataset, batch_size=128, shuffle=True))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = datasets.ImageFolder(ext_path, transform=transform)

    train_data_arr.append(preload_batches(DataLoader(dataset, batch_size=512, shuffle=True)))

for task_name in task_names:
    ext_path = './CIFAR/data/test/'+task_name+'/'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = datasets.ImageFolder(ext_path, transform=transform)
    dataloader = preload_batches(DataLoader(dataset, batch_size=512, shuffle=True))
    test_data_arr.append(dataloader)

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

all_progenitors = []

if pretrained == 1:
    with open('./CIFAR/prepared_model_info/mask_dict.pkl', 'rb') as f:
        progenitor_masks = pickle.load(f)
    hyperparams = {'gen':120, 'subpop':60, 'sbxdi':10, 'pmdi':10}
    print('using pretrained')
    for i, task_name in enumerate(task_names):
        save_path = './CIFAR/selected_prepared_model_info/' + task_name + '/'
        with open('./CIFAR/prepared_model_info/' +task_name + '/stats_dict.pkl', 'rb') as f:
            stats_dict = pickle.load(f)

        accuracies = [stats_dict[i][0] for i in stats_dict]
        sizes = [stats_dict[i][1] for i in stats_dict]
        processed = [(accuracies[i], sizes[i]) for i,item in enumerate(accuracies)]
        _, _, indices = is_pareto_efficient(100, processed)
        progenitors = [progenitor_masks[task_name][j] for j in indices][0:hyperparams['subpop']]
        while len(progenitors) < hyperparams['subpop']:
            temp = Chromosome()
            temp.initialize(18944, 0, 0.15)
            progenitors.append(temp.rnvec)
        all_progenitors.append(progenitors)
else:
    print('No pretrain')
    hyperparams = {'gen':120, 'subpop':60, 'sbxdi':10, 'pmdi':10}
    for i, task_name in enumerate(task_names):
        progenitors = []
        while len(progenitors) < hyperparams['subpop']:
            temp = Chromosome()
            temp.initialize(18944, 0, 0.15)

            progenitors.append(temp.rnvec)
        all_progenitors.append(progenitors)

try:
        multitaskMFEA('./CIFAR/results/MT/',
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
