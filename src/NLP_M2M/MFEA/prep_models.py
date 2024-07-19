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
from datetime import datetime
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
from chromosome import Chromosome
from compare_solutions import *
from operators import *
from segmentation_models_pytorch.losses import FocalLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import datasets
import transformers
from transformers.optimization import Adafactor, AdafactorSchedule
import random
DEVICE_1 = torch.device('cuda:0')

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
    return batches

def load_model():
    model = Model()
    model.load_state_dict(torch.load('./NLP_M2M/MFEA/models/base.pth'))
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

def count_active_params(state_dict):
    total = 0
    for i in state_dict:
        flattened = torch.flatten(state_dict[i])
        total += torch.count_nonzero(flattened)
    return total

a = datetime.now()

initial_generation = int(sys.argv[1])

mask_dict = {}

task_names = ['cs_only', 'de_only']
for i, task_name in enumerate(task_names):
    trainloader = train_data_arr[i]

    model = load_model()

    model.update_backup()
    model.cuda(0)
    mask_sizing = size_mask(model.return_model_state())
    stats_dict = {}
    save_path = './NLP_M2M/MFEA/prepared_model_info/' + task_name + '/'
    if task_name not in mask_dict:
        mask_dict.update({task_name:{}})
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(task_name)
    print('------------------------------------')
    for j in tqdm(range(initial_generation)):
        temp = Chromosome()
        temp.initialize(67584, 0.25, 0.95)
        temp.collapse_prevention(mask_sizing)
        mask = temp.rnvec

        model.apply_mask(mask, mask_sizing)
        loss = test_loss(model, trainloader)
        size = count_active_params(model.return_model_state())
        stats_dict.update({j:[loss, size]})
        model.revert_weights()
        curr = mask_dict[task_name]
        curr.update({j:mask})
        mask_dict.update({task_name:curr})

    with open(save_path + "stats_dict.pkl", 'wb') as f:
        pickle.dump(stats_dict, f)

with open('./NLP_M2M/MFEA/prepared_model_info/' + "mask_dict.pkl", 'wb') as f:
    pickle.dump(mask_dict, f)
print('Time Elapsed')
print((datetime.now() - a).seconds)
