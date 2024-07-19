from __future__ import division
import os
import shutil
import argparse
import copy
import pickle
import json
import math
import random
import numpy as np
import pdb
import operator
import time
from datetime import datetime
from timeit import default_timer as timer
from collections import OrderedDict, Counter
from threading import Thread
from hyp import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchinfo import summary
from torch.autograd import Variable
from chromosome import Chromosome
from compare_solutions import *
from operators import *
from model import Model

DEVICE_1 = torch.device('cuda:0')
DEVICE_2 = torch.device('cpu')

# Helper functions
def count_active_params(state_dict):
    # Count number of non zero parameters in model
    # state dictionary, and convert to number of bytes.
    total = 0
    for i in state_dict:
        flattened = torch.flatten(state_dict[i])
        total += torch.count_nonzero(flattened)
    return total.item() * 32 /(1024*8)

def test_models(model, extloader):
    model.eval()
    avg_loss = 0
    criterion = nn.MSELoss().to(DEVICE_1)
    with torch.no_grad():
        for i, data in enumerate(extloader):
            x, y = data[0], data[1]
            x, y=Variable(x), Variable(y)
            fx = model(x)
            loss = torch.sqrt(criterion(fx.squeeze(), y))
            avg_loss += loss.detach().item()
    avg_loss = avg_loss/len(extloader)
    return avg_loss

def finetune(model, mask, mask_sizing, extloader, evalloader, epochs, task_name, ind):
    criterion = nn.MSELoss().to(DEVICE_1)
    lmbda = lambda epoch: 0.99
    best_loss = 10000
    average_loss = 0
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lmbda)

    for epoch in range(epochs):
        model.train()
        loss_per_epoch = 10101010

        for i, batch in enumerate(extloader):
            optim.zero_grad()
            x, y = batch[0].float().cuda(0), batch[1].float().cuda(0)
            x, y = Variable(x), Variable(y)
            fx = model(x)
            loss = torch.sqrt(criterion(fx.squeeze(), y))
            loss.backward()
            optim.step()
            x = x.cpu()
            y = y.cpu()
            model.apply_mask(mask, mask_sizing)
        loss_per_epoch = test_models(model, evalloader)
        model = model.float()
        scheduler.step()
        if loss_per_epoch < best_loss:
            best_loss = loss_per_epoch
    return best_loss, count_active_params(model.state_dict())

def finetune_evo(model, mask, mask_sizing, extloader, epochs):
    loss_per_epoch = test_models(model, extloader)
    return loss_per_epoch, count_active_params(model.state_dict())

def reset(population, pop):
    for i in range(pop):
        population[i].dominationcount = 0
        population[i].dominatedset = []
        population[i].dominatedsetlength = 0
    return population

def initialize_and_evaluate_parents(progenitor_arr, model_arr, data_arr, dims, pop, subpop_size, mask_sizing, pretrained):
    population = []
    for i in range(pop):
        temp = Chromosome()
        if i < subpop_size:
            temp.rnvec = progenitor_arr[0][i]
            temp.skill_factor = 1
        elif subpop_size <= i < subpop_size*2:
            temp.rnvec = progenitor_arr[1][i-subpop_size]
            temp.skill_factor = 2
        elif subpop_size <= i < subpop_size*3:
            temp.rnvec = progenitor_arr[1][i-subpop_size*2]
            temp.skill_factor = 3
        elif subpop_size <= i < subpop_size*4:
            temp.rnvec = progenitor_arr[1][i-subpop_size*3]
            temp.skill_factor = 4
        population.append(temp)
    print(" - Evaluating Progenitor Population")
    progenitor_results = []
    for i in range(len(population)):
        mask = population[i].rnvec
        model = model_arr[population[i].skill_factor - 1]
        model.apply_mask(mask, mask_sizing)
        loss = test_models(model, data_arr[population[i].skill_factor - 1])
        mem = count_active_params(model.state_dict())
        model.revert_weights()
        population[i].Evaluate([loss, mem])
        progenitor_results.append((loss, mem))
    print("--------------- Progenitor Evaluation Done -------------")
    return population, progenitor_results

def evaluate_offspring(model_arr, train_data_arr, pop, mask_sizing, offspring):
    results = {}
    for i in range(pop):
        mask = offspring[i].rnvec
        model = model_arr[offspring[i].skill_factor - 1]
        model.apply_mask(mask, mask_sizing)
        loss, mem = finetune_evo(model,
                            mask,
                            mask_sizing,
                            train_data_arr[offspring[i].skill_factor - 1],
                            1)
        offspring[i].Evaluate([loss, mem])
        model.revert_weights()
    return offspring


def is_pareto_efficient(lim1, processed):
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

def final_finetune(model_arr, fitnesses, train_data_arr, test_data_arr, pop, mask_sizing, offspring, subpop_size, finetune_epochs):

    results = {}
    task_names = ['Aotizhongxin', 'Changping', 'Nongzhanguan', 'Dingling']

    pareto_indices = []

    losses = []
    sizes = []
    for fit_arr in fitnesses:
        print("--------")
        losses = [i[0] for i in fit_arr]
        sizes = [i[1] for i in fit_arr]
        processed = [(losses[i], sizes[i]) for i,item in enumerate(losses)]
        _, _, indices = is_pareto_efficient(100, processed)
        pareto_indices.append(indices)

    print(pareto_indices)
    for i in range(pop):
        if i < subpop_size:
            ind = i
        elif subpop_size <= i < subpop_size*2:
            ind = i-subpop_size
        elif subpop_size*2 <= i < subpop_size*3:
            ind = i-subpop_size*2
        elif subpop_size*3 <= i < subpop_size*4:
            ind = i-subpop_size*3

        if ind in pareto_indices[offspring[i].skill_factor - 1]:
            task_name = task_names[offspring[i].skill_factor - 1]
            print('Task: ', task_name, ', Model: ', ind)
            mask = offspring[i].rnvec

            model = Model()
            model = model.cuda(0)
            model.load_state_dict(torch.load('./Regression_timeseries/models/base.pth'))
            model.update_backup()

            model.apply_mask(mask, mask_sizing)

            loss, mem = finetune(model,
                                mask,
                                mask_sizing,
                                train_data_arr[offspring[i].skill_factor - 1],
                                test_data_arr[offspring[i].skill_factor - 1],
                                finetune_epochs, task_names[offspring[i].skill_factor - 1], i)
            model.revert_weights()
            print('Loss: ', loss, ', Memory:', mem)
            if task_name not in results:
                results.update({task_name:[(loss, mem)]})
            else:
                curr = results[task_name]
                curr.append((loss, mem))
                results.update({task_name:curr})
    return results

def get_final_population(list_of_pops, pop_sizes):
    final_pop = []
    for i, pop in enumerate(list_of_pops):
        for j in range(pop_sizes[i]):
            final_pop.append(pop[j])
    return final_pop

def crossover_and_mutate(p1, p2, sbxdi, pmdi):
    child1, child2 = sbx_crossover(p1, p2, sbxdi)
    child1 = mutate_poly(child1, pmdi)
    child2 = mutate_poly(child2, pmdi)
    return child1, child2

def mutate_only(p1, p2, pmdi):
    child1 = mutate_poly(p1, pmdi)
    child2 = mutate_poly(p2, pmdi)
    return child1, child2

def intra_task_crossover_and_mutate_one_child(p1, p2, sbxdi, pmdi):
    child1rnvec, child2rnvec = crossover_and_mutate(p1, p2, sbxdi, pmdi)
    roll = np.random.uniform(0, 1, 1)[0]
    if roll > 0.5:
        return child1rnvec
    else:
        return child2rnvec

def clone_prevention(child1, child2, p1, p2):
    new_child1 = np.copy(child1)
    new_child2 = np.copy(child2)

    if np.array_equal(child1, p1) or np.array_equal(child1, p2):
        print("Child 1 is a clone. Inserting new material.")
        choice = np.random.choice(range(0, len(child1)), 1)[0]
        if child1[choice] == 1:
            new_child1[choice] = 0
        else:
            new_child1[choice] = 1
    if np.array_equal(child2, p1) or np.array_equal(child2, p2):
        print("Child 2 is a clone. Inserting new material.")
        choice = np.random.choice(range(0, len(child1)), 1)[0]
        if child2[choice] == 1:
            new_child2[choice] = 0
        else:
            new_child2[choice] = 1
    return new_child1, new_child2

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

def multitaskMFEA(target_dir,
                run_name,
                progenitor_arr,
                model_arr,
                train_data_arr,
                train_data_arr_unloaded,
                test_data_arr,
                hyperparams,
                pretrained,
                rmp_setting,
                final_finetune_switch, MFEA_II, finetune_epochs):

    dims = 4096
    print('Filter Dimensions ', dims)

    state = model_arr[0].state_dict()
    mask_sizing = size_mask(state)
    print("Testing Generalists")
    for i, model in enumerate(model_arr):
        mask = np.ones(dims)
        loss = test_models(model, test_data_arr[i])
        mem = count_active_params(model.state_dict())
        print('Generalist: ', i, ', Loss: ', loss, ', Size: ',mem)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    target_dir = target_dir + run_name + '/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    print('Result Save Path ', target_dir)

    a = datetime.now()

    total_time = datetime.now()
    print("Evolution Commencing ... ")
    path_save = target_dir

    print("Defining hyperparameters")
    gen = hyperparams['gen']
    pop1 = hyperparams['subpop']  # population size of task 1
    pop2 = hyperparams['subpop'] # population size of task 2
    pop3 = hyperparams['subpop']  # task 3
    pop4 = hyperparams['subpop']

    pop = pop1 + pop2 + pop3 + pop4 # has to be even sum
    if pop % 2 != 0:
        pop1 = pop1 + 1
        pop = pop1 + pop2 + pop3 + pop4

    sbxdi = hyperparams['sbxdi']
    pmdi = hyperparams['pmdi']
    print("Done. Chromosome Dimension: ", dims, "SBXDI", sbxdi, "PMDI", pmdi)
    print("Commencing Runs ... ")

    startall = timer()

    rep_fit1 = []  # pareto front objective values
    rep_fit2 = []
    rep_fit3 = []
    rep_fit4 = []

    rep_NDfit1 = []  # pareto front non-dominated objective values
    rep_NDfit2 = []
    rep_NDfit3 = []
    rep_NDfit4 = []

    rep_hv1 = []
    rep_hv2 = []
    rep_hv3 = []
    rep_hv4 = []
    rep_time = []

    print('Initializing Progenitor Population and assigning skill factors')
    population, progenitor_results = initialize_and_evaluate_parents(progenitor_arr, model_arr, train_data_arr, dims, pop, pop1, mask_sizing, pretrained)
    print('Progenitor Population Evaluated')
    print('Nondominated Sorting and Diversity Checks Commencing')
    popT1 = []
    popT2 = []
    popT3 = []
    popT4 = []

    for x in population:
        if x.skill_factor == 1:
            popT1.append(x)
        elif x.skill_factor == 2:
            popT2.append(x)
        elif x.skill_factor == 3:
            popT3.append(x)
        elif x.skill_factor == 4:
            popT4.append(x)


    obj = 2


    popT1, frontnumbers = nondominatedsort(popT1, pop1, obj)
    popT1, minimums = diversity(popT1, frontnumbers, pop1, obj)
    print("T1 Sorting and Diversity Checks Done")
    popT2, frontnumbers = nondominatedsort(popT2, pop2, obj)
    popT2, minimums = diversity(popT2, frontnumbers, pop2, obj)
    print("T2 Sorting and Diversity Checks Done")
    popT3, frontnumbers = nondominatedsort(popT3, pop3, obj)
    popT3, minimums = diversity(popT3, frontnumbers, pop3, obj)
    print("T3 Sorting and Diversity Checks Done")
    popT4, frontnumbers = nondominatedsort(popT4, pop4, obj)
    popT4, minimums = diversity(popT4, frontnumbers, pop4, obj)
    print("T4 Sorting and Diversity Checks Done")
    print('Sorts and Checks complete, merging...')

    merge = []
    merge.extend(popT1)
    merge.extend(popT2)
    merge.extend(popT3)
    merge.extend(popT4)
    population = merge

    print("Merge Complete")

    gen_fit = []
    gen_population = []
    all_gen_masks = []

    for generation in range(gen):  # Main evolutionary loop
        print('Gen Number: ', generation," - Crossover and Mutation commencing ... ")
        startall_gen = timer()
        randlist = np.random.permutation(pop)
        shuffled_pop = []
        for i in range(pop):
            shuffled_pop.append(population[randlist[i]])
        population = shuffled_pop
        parent = []
        for i in range(pop):
            temp = Chromosome()
            p1 = random.randint(0, pop - 1)
            p2 = random.randint(0, pop - 1)
            if population[p1].rank < population[p2].rank:
                temp.rnvec = population[p1].rnvec
                temp.skill_factor = population[p1].skill_factor
            elif population[p1].rank == population[p2].rank:
                if random.random() <= 0.5:
                    temp.rnvec = population[p1].rnvec
                    temp.skill_factor = population[p1].skill_factor
                else:
                    temp.rnvec = population[p2].rnvec
                    temp.skill_factor = population[p2].skill_factor
            else:
                temp.rnvec = population[p2].rnvec
                temp.skill_factor = population[p2].skill_factor
            temp.collapse_prevention(mask_sizing)
            parent.append(temp)

        offspring = []
        subpops = []
        subpopT1 = []
        subpopT2 = []
        subpopT3 = []
        subpopT4 = []

        for x in parent:
            if x.skill_factor == 1:
                subpopT1.append(x.rnvec)
            elif x.skill_factor == 2:
                subpopT2.append(x.rnvec)
            elif x.skill_factor == 3:
                subpopT3.append(x.rnvec)
            elif x.skill_factor == 4:
                subpopT4.append(x.rnvec)

        subpops.append(np.asarray(subpopT1))
        subpops.append(np.asarray(subpopT2))
        subpops.append(np.asarray(subpopT3))
        subpops.append(np.asarray(subpopT4))
        if MFEA_II:
            RMP = learn_rmp(np.array(subpops), dims)
            print("============================= Multitask =====================================")
            print('Gen Number: ', generation," RMP: ", RMP)
        else:
            rmp = rmp_setting
            print("============================= Multitask =====================================")
            print('Gen Number: ', generation," RMP: ", rmp)

        for i in range(0, pop, 2):
            p1 = i
            p2 = i + 1
            child1 = Chromosome()
            child2 = Chromosome()
            if MFEA_II:
                rmp = RMP[parent[p1].skill_factor - 1, parent[p2].skill_factor - 1]
                child1rnvec, child2rnvec = crossover_and_mutate(parent[p1].rnvec, parent[p2].rnvec, sbxdi, pmdi)
                child1rnvec, child2rnvec = clone_prevention(child1rnvec, child2rnvec, parent[p1].rnvec, parent[p2].rnvec)
                child1rnvec, child2rnvec = variable_swap(child1rnvec, child2rnvec, 0.5)
                child1.rnvec = child1rnvec
                child2.rnvec = child2rnvec
                child1.skill_factor = parent[p1].skill_factor
                child2.skill_factor = parent[p2].skill_factor
            elif np.random.rand() < rmp:
                child1rnvec, child2rnvec = crossover_and_mutate(parent[p1].rnvec, parent[p2].rnvec, sbxdi, pmdi)
                child1rnvec, child2rnvec = clone_prevention(child1rnvec, child2rnvec, parent[p1].rnvec, parent[p2].rnvec)
                child1rnvec, child2rnvec = variable_swap(child1rnvec, child2rnvec, 0.5)
                child1.rnvec = child1rnvec
                child2.rnvec = child2rnvec
                roll = np.random.uniform(0, 1, 1)[0]
                if roll > 0.5:
                    child1.skill_factor = parent[p1].skill_factor
                    child2.skill_factor = parent[p2].skill_factor
                else:
                    child1.skill_factor = parent[p2].skill_factor
                    child2.skill_factor = parent[p1].skill_factor
            else:
                p11 = None
                p22 = None
                for j, _ in enumerate(parent):
                    if parent[j].skill_factor == parent[p1].skill_factor and j != p1:
                        p11 = j
                    if parent[j].skill_factor == parent[p2].skill_factor and j != p2:
                        p22 = j
                    if p11 is not None and p22 is not None:
                        break

                if p11 is None:
                    print("Dreadfulness at line 534: Lack of choice for parentage given small population of ", pop)
                    p11 = p1
                if p22 is None:
                    print("Dreadfulness at line 534: Lack of choice for parentage given small population of ", pop)
                    p22 = p2

                child1rnvec = intra_task_crossover_and_mutate_one_child(parent[p1].rnvec, parent[p11].rnvec, sbxdi, pmdi)
                temp_child1rnvec = intra_task_crossover_and_mutate_one_child(parent[p11].rnvec, parent[p1].rnvec, sbxdi, pmdi)

                child2rnvec = intra_task_crossover_and_mutate_one_child(parent[p2].rnvec, parent[p22].rnvec, sbxdi, pmdi)
                temp_child2rnvec = intra_task_crossover_and_mutate_one_child(parent[p22].rnvec, parent[p2].rnvec, sbxdi, pmdi)

                child1rnvec, child2rnvec = clone_prevention(child1rnvec, child2rnvec, parent[p1].rnvec, parent[p2].rnvec)
                child1rnvec, child2rnvec = clone_prevention(child1rnvec, child2rnvec, parent[p11].rnvec, parent[p22].rnvec)

                temp_child1rnvec, temp_child2rnvec = clone_prevention(child1rnvec, child2rnvec, parent[p1].rnvec, parent[p2].rnvec)
                temp_child1rnvec, temp_child2rnvec = clone_prevention(child1rnvec, child2rnvec, parent[p11].rnvec, parent[p22].rnvec)

                child1rnvec, temp_child1rnvec = variable_swap(child1rnvec, temp_child1rnvec, 0.5)
                child2rnvec, temp_child2rnvec = variable_swap(child2rnvec, temp_child2rnvec, 0.5)

                child1.rnvec = child1rnvec
                child2.rnvec = child2rnvec

                child1.skill_factor = parent[p11].skill_factor
                child2.skill_factor = parent[p22].skill_factor

            child1.collapse_prevention(mask_sizing)
            child2.collapse_prevention(mask_sizing)
            offspring.append(child1)
            offspring.append(child2)

        print('Gen Number: ', generation, " - Crossover and Mutation complete. Evaluating Offspring Population ... ")
        offspring = evaluate_offspring(model_arr,
                                    train_data_arr,
                                    pop,
                                    mask_sizing,
                                    offspring)
        print('Gen Number: ', generation, " - Offspring Evaluated. Merging Parents and Offspring")
        population = reset(population, pop)
        intpopulation = []
        for x in population:
            intpopulation.append(x)
        for x in offspring:
            intpopulation.append(x)
        IntpopT1 = []
        IntpopT2 = []
        IntpopT3 = []
        IntpopT4 = []

        for x in intpopulation:
            if x.skill_factor == 1:
                IntpopT1.append(x)
            elif x.skill_factor == 2:
                IntpopT2.append(x)
            elif x.skill_factor == 3:
                IntpopT3.append(x)
            elif x.skill_factor == 4:
                IntpopT4.append(x)


        T1_pop = IntpopT1.__len__()
        T2_pop = IntpopT2.__len__()
        T3_pop = IntpopT3.__len__()
        T4_pop = IntpopT4.__len__()


        IntpopT1, frontnumbers = nondominatedsort(IntpopT1, T1_pop, obj)
        IntpopT1, minimums = diversity(IntpopT1, frontnumbers, T1_pop, obj)

        IntpopT2, frontnumbers = nondominatedsort(IntpopT2, T2_pop, obj)
        IntpopT2, minimums = diversity(IntpopT2, frontnumbers, T2_pop, obj)

        IntpopT3, frontnumbers = nondominatedsort(IntpopT3, T3_pop, obj)
        IntpopT3, minimums = diversity(IntpopT3, frontnumbers, T3_pop, obj)

        IntpopT4, frontnumbers = nondominatedsort(IntpopT4, T4_pop, obj)
        IntpopT4, minimums = diversity(IntpopT4, frontnumbers, T4_pop, obj)


        print('Gen Number: ', generation, " - Acquiring unique parents and offspring for merging ... ")

        population = get_final_population([IntpopT1, IntpopT2, IntpopT3, IntpopT4], [pop1, pop2, pop3, pop4])

        print('Gen Number: ', generation, " - Merge Complete, ND Sorting for Hypervolume Calculation ... ")

        fit1 = []
        fit2 = []
        fit3 = []
        fit4 = []

        nd_fit1 = [] #for hv
        nd_fit2 = []
        nd_fit3 = []
        nd_fit4 = []

        for x in population:
            if x.skill_factor == 1:
                fit1.append(x.objs_T1)
                if x.front == 1: #non-dominated front solutions
                    nd_fit1.append(x.objs_T1)
            elif x.skill_factor == 2:
                fit2.append(x.objs_T2)
                if x.front == 1:
                    nd_fit2.append(x.objs_T2)
            elif x.skill_factor == 3:
                fit3.append(x.objs_T3)
                if x.front == 1:
                    nd_fit3.append(x.objs_T3)
            elif x.skill_factor == 4:
                fit4.append(x.objs_T4)
                if x.front == 1:
                    nd_fit4.append(x.objs_T4)


        nd_fit1 = np.array(nd_fit1)
        ref_point = np.array([35,2121]) * 1.1
        hv_fit1 = hypervolume(nd_fit1, ref_point)
        # hv_fit1 = hv.compute(ref_point)
        hv_fit1 /= (2121*35)

        nd_fit2 = np.array(nd_fit2)
        ref_point = np.array([35,2121]) * 1.1
        hv_fit2 = hypervolume(nd_fit2, ref_point)
        # hv_fit2 = hv.compute(ref_point)
        hv_fit2 /= (2121*35)

        nd_fit3 = np.array(nd_fit3)
        ref_point = np.array([35,2121]) * 1.1
        hv_fit3 = hypervolume(nd_fit3, ref_point)
        # hv_fit3 = hv.compute(ref_point)
        hv_fit3 /= (2121*35)

        nd_fit4 = np.array(nd_fit4)
        ref_point = np.array([35,2121]) * 1.1
        hv_fit4 = hypervolume(nd_fit4, ref_point)
        # hv_fit3 = hv.compute(ref_point)
        hv_fit4 /= (2121*35)

        gen_masks = []
        for x in population:
            gen_masks.append(x.rnvec)
        all_gen_masks.append(gen_masks)
        print('Gen Number: ', generation, " - Complete. Saving results.")

        rep_fit1.append(fit1)  # all fitness values ---
        rep_fit2.append(fit2)  # (IGD1)
        rep_fit3.append(fit3)  # (IGD1)
        rep_fit4.append(fit4)
        rep_NDfit1.append(nd_fit1)
        rep_NDfit2.append(nd_fit2)
        rep_NDfit3.append(nd_fit3)
        rep_NDfit4.append(nd_fit4)
        rep_hv1.append(hv_fit1)
        rep_hv2.append(hv_fit2)
        rep_hv3.append(hv_fit3)
        rep_hv4.append(hv_fit4)
        end_gen = timer()
        time_gen = end_gen - startall_gen

        rep_time.append(time_gen)

        print("============================= Multitask =====================================")
        print("---------- Hypervolumes ------------")
        print("Task 1 hypervolume: ", hv_fit1, "Task 2 hypervolume: ", hv_fit2, "Task 3 hypervolume: ", hv_fit3)
        print("---------- Task 1 Fitness ------------")
        print(fit1[0:20])
        print("Generation", generation, 'Average Fitness Task 1', np.mean(np.array(fit1), axis=0)[0], 'Average Memory Task 1', np.mean(np.array(fit1), axis=0)[1])
        print("---------- Task 2 Fitness ------------")
        print(fit2[0:20])
        print("Generation", generation, 'Average Fitness Task 2', np.mean(np.array(fit2), axis=0)[0], 'Average Memory Task 2', np.mean(np.array(fit2), axis=0)[1])
        print("---------- Task 3 Fitness ------------")
        print(fit3[0:20])
        print("Generation", generation, 'Average Fitness Task 3', np.mean(np.array(fit3), axis=0)[0], 'Average Memory Task 3', np.mean(np.array(fit3), axis=0)[1])
        print("---------- Task 4 Fitness ------------")
        print(fit4[0:20])
        print("Generation", generation, 'Average Fitness Task 3', np.mean(np.array(fit4), axis=0)[0], 'Average Memory Task 4', np.mean(np.array(fit4), axis=0)[1])

    if final_finetune_switch == 1:
        finetuned_results = final_finetune(model_arr,
                                    [fit1, fit2, fit3, fit4],
                                    train_data_arr_unloaded,
                                    test_data_arr,
                                    pop,
                                    mask_sizing,
                                    population,
                                    hyperparams['subpop'],
                                    finetune_epochs)

        print(finetuned_results)
    temp_masks = []
    for i in range(pop):
        temp_masks.append(population[i].rnvec)

    endall = timer()
    time_all = endall - startall

    with open(path_save + "population.pkl", 'wb') as f:
        pickle.dump(population, f)

    if final_finetune_switch == 1:
        with open(path_save + "finetuned_results.pkl", 'wb') as f:
            pickle.dump(finetuned_results, f)

    with open(path_save + "progenitor_results.pkl", 'wb') as f:
        pickle.dump(progenitor_results, f)

    with open(path_save + "rep_fit1.pkl", 'wb') as f:
        pickle.dump(rep_fit1, f)
    with open(path_save + "rep_fit2.pkl", 'wb') as f:
        pickle.dump(rep_fit2, f)
    with open(path_save + "rep_fit3.pkl", 'wb') as f:
        pickle.dump(rep_fit3, f)
    with open(path_save + "rep_fit4.pkl", 'wb') as f:
        pickle.dump(rep_fit4, f)

    with open(path_save + "rep_NDfit1.pkl", 'wb') as f:
        pickle.dump(rep_NDfit1, f)
    with open(path_save + "rep_NDfit2.pkl", 'wb') as f:
        pickle.dump(rep_NDfit2, f)
    with open(path_save + "rep_NDfit3.pkl", 'wb') as f:
        pickle.dump(rep_NDfit3, f)
    with open(path_save + "rep_NDfit4.pkl", 'wb') as f:
        pickle.dump(rep_NDfit4, f)


    with open(path_save + "rep_hv1.pkl", 'wb') as f:
        pickle.dump(rep_hv1, f)
    with open(path_save + "rep_hv2.pkl", 'wb') as f:
        pickle.dump(rep_hv2, f)
    with open(path_save + "rep_hv3.pkl", 'wb') as f:
        pickle.dump(rep_hv3, f)
    with open(path_save + "rep_hv4.pkl", 'wb') as f:
        pickle.dump(rep_hv4, f)


    with open(path_save + "rep_time.pkl", 'wb') as f:
        pickle.dump(rep_time, f)

    with open(path_save + "time_all.pkl", 'wb') as f:
        pickle.dump(time_all, f)

    with open(path_save + "all_gen_masks.pkl", 'wb') as f:
        pickle.dump(all_gen_masks, f)

    print("Results saved")
    print((datetime.now() - a).seconds)
    # # ----------------------------------------------------------------------------------------------------------------------
