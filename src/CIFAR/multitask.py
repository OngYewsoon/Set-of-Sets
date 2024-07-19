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
import config as C
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
import models.ResNet as ResNet

DEVICE_1 = torch.device('cuda:0')
DEVICE_2 = torch.device('cpu')


def count_active_params(state_dict):
    total = 0
    for i in state_dict:
        flattened = torch.flatten(state_dict[i])
        total += torch.count_nonzero(flattened)
    return total.item() * 16 /(1024*8)

def test_only(model, mask, mask_sizing, evalloader):
    model.eval()
    model.apply_mask(mask, mask_sizing)
    model.half()
    total_correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(evalloader):
            images, labels=images, labels
            images, labels=Variable(images), Variable(labels)
            C.set_is_quantized(0)
            output = model(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()/len(images)
    model.revert_weights()
    return 100-(100*(float(total_correct) / len(evalloader))), count_active_params(model.state_dict())

def test_without_size(model, evalloader):
    model.eval()
    avg_acc = 0

    criterion = nn.CrossEntropyLoss().cuda(0)

    with torch.no_grad():
        for i, data in enumerate(evalloader):
            x, y = data[0], data[1]
            x, y=Variable(x), Variable(y)
            fx = model(x)
            loss = criterion(fx.squeeze(), y)
            avg_acc += loss.detach().item()

    avg_acc = avg_acc/len(evalloader)
    return avg_acc

def test_acc(model, evalloader):
    model.eval()
    avg_acc = 0
    model.half()
    with torch.no_grad():
        for i, data in enumerate(evalloader):

            x, y = data[0].cuda(0), data[1].cuda(0)
            x, y=Variable(x), Variable(y)
            fx = model(x)
            _, predicted = fx.max(1)
            acc_per_batch = 100. * predicted.eq(y).sum().item() / y.size(0)
            avg_acc += acc_per_batch
    avg_acc = avg_acc/len(evalloader)
    model = model.float()
    return avg_acc


def finetune(model, mask, mask_sizing, extloader, evalloader, epochs):
    criterion = nn.CrossEntropyLoss().to(DEVICE_1)
    lmbda = lambda epoch: 0.95
    best_acc = 0
    average_acc = 0
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lmbda)

    for epoch in range(epochs):
        model.train()
        acc_per_epoch = 0

        for i, batch in enumerate(extloader):
            optim.zero_grad()
            x, y = batch[0].cuda(0), batch[1].cuda(0)
            x, y = Variable(x), Variable(y)
            fx = model(x)
            loss = criterion(fx.squeeze(), y)
            loss.backward()
            optim.step()
            x = x.cpu()
            y = y.cpu()
        acc_per_epoch = test_acc(model, evalloader)
        scheduler.step()
        if acc_per_epoch > best_acc:
            best_acc = acc_per_epoch

    return best_acc, count_active_params(model.state_dict())

def test_offspring(model, mask, mask_sizing, extloader, epochs):
    acc_per_epoch = test_without_size(model, extloader)
    return acc_per_epoch, count_active_params(model.state_dict())

def reset(population, pop):
    for i in range(pop):
        population[i].dominationcount = 0
        population[i].dominatedset = []
        population[i].dominatedsetlength = 0
    return population

def initialize_and_evaluate_parents(progenitor_arr, model_arr, data_arr, dims, pop, subpop_size, mask_sizing, pretrained):
    population = []
    counter = 0
    counter2 = 0.005
    for i in range(pop):
        temp = Chromosome()

        if i < subpop_size:
            temp.rnvec = progenitor_arr[0][i]
            temp.skill_factor = 1
        elif subpop_size <= i < subpop_size*2:

            temp.rnvec = progenitor_arr[1][i-subpop_size]
            temp.skill_factor = 2

        population.append(temp)

    print(" - Evaluating Progenitor Population")
    progenitor_results = []
    for i in range(len(population)):

        mask = population[i].rnvec
        acc, mem = test_only(model_arr[population[i].skill_factor - 1], mask, mask_sizing, data_arr[population[i].skill_factor - 1])
        population[i].Evaluate([acc, mem])
        progenitor_results.append((acc, mem))
        # print(i)
    print("--------------- Progenitor Evaluation Done -------------")

    return population, progenitor_results

def evaluate_offspring(model_arr, train_data_arr, pop, mask_sizing, offspring):

    results = {}

    for i in range(pop):

        mask = offspring[i].rnvec
        model = model_arr[offspring[i].skill_factor - 1]
        model.apply_mask(mask, mask_sizing)
        model.half()
        acc, mem = test_offspring(model,
                            mask,
                            mask_sizing,
                            train_data_arr[offspring[i].skill_factor - 1],
                            1)
        offspring[i].Evaluate([acc, mem])
        model.revert_weights()

    return offspring


def is_pareto_efficient(processed):
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
    task_names = ['animals', 'vehicles']

    pareto_indices = []

    accuracies = []
    sizes = []
    for fit_arr in fitnesses:
        print("--------")
        accuracies = [i[0] for i in fit_arr]
        sizes = [i[1] for i in fit_arr]
        processed = [(accuracies[i], sizes[i]) for i,item in enumerate(accuracies)]
        _, _, indices = is_pareto_efficient(processed)
        pareto_indices.append(indices)

    print(pareto_indices)
    for i in range(pop):
        if i < subpop_size:
            ind = i
        elif subpop_size <= i < subpop_size*2:
            ind = i-subpop_size

        if ind in pareto_indices[offspring[i].skill_factor - 1]:
            task_name = task_names[offspring[i].skill_factor - 1]
            print('Task: ', task_name, ', Model: ', ind)
            mask = offspring[i].rnvec
            task_name = task_names[offspring[i].skill_factor - 1]
            task_indices = {'animals':[3, 4, 5, 7], 'vehicles':[0, 1, 8, 9]}

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

            model.apply_mask(mask, mask_sizing)

            acc, mem = finetune(model,
                                mask,
                                mask_sizing,
                                train_data_arr[offspring[i].skill_factor - 1],
                                test_data_arr[offspring[i].skill_factor - 1],
                                finetune_epochs)
            print('Accuracy: ', acc, ', Memory:', mem)
            model.revert_weights()

            if task_name not in results:
                results.update({task_name:[(acc, mem)]})
            else:
                curr = results[task_name]
                curr.append((acc, mem))
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
        # print(i)
        # print(state_dict[i].shape)
        # if 'conv' in i and 'weight' in i:
        shape = state_dict[i].shape
        if len(shape) > 1:
            if 'linear' not in i and 'layer' in i:
                size1 = shape[0]
                total += size1*4
                mask_sizing.update({i+'_1':int(size1)})
                mask_sizing.update({i+'_2':int(size1)})
                mask_sizing.update({i+'_3':int(size1)})
                mask_sizing.update({i+'_4':int(size1)})
                    # print(list(shape))
    print(total)
    return mask_sizing

def multitaskMFEA(target_dir,
                run_number,
                progenitor_arr,
                model_arr,
                train_data_arr,
                train_data_arr_unloaded,
                test_data_arr,
                hyperparams,
                pretrained,
                rmp_setting,
                final_finetune_switch,
                MFEA_II, finetune_epochs):

    dims = 18944
    print('Filter Dimensions ', dims)

    state = model_arr[0].return_model_state()
    mask_sizing = size_mask(state)
    print("Testing Generalists")
    for i, model in enumerate(model_arr):
        mask = np.ones(dims)
        acc, mem = test_only(model, mask, mask_sizing, test_data_arr[i])
        print('Generalist: ', i, ', Accuracy: ', 100-acc, ', Size: ',mem)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    target_dir = target_dir + run_number + '/'
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

    pop = pop1 + pop2# has to be even sum
    if pop % 2 != 0:
        pop1 = pop1 + 1
        pop = pop1 + pop2

    sbxdi = hyperparams['sbxdi']
    pmdi = hyperparams['pmdi']
    print("Done. Chromosome Dimension: ", dims, "SBXDI", sbxdi, "PMDI", pmdi)
    print("Commencing Runs ... ")

    startall = timer()

    rep_fit1 = []  # pareto front objective values
    rep_fit2 = []

    rep_NDfit1 = []  # pareto front non-dominated objective values
    rep_NDfit2 = []

    rep_hv1 = []
    rep_hv2 = []
    rep_time = []

    print('Initializing Progenitor Population and assigning skill factors')
    population, progenitor_results = initialize_and_evaluate_parents(progenitor_arr, model_arr, train_data_arr, dims, pop, pop1, mask_sizing, pretrained)
    print('Progenitor Population Evaluated')
    print('Nondominated Sorting and Diversity Checks Commencing')
    popT1 = []
    popT2 = []

    for x in population:
        if x.skill_factor == 1:
            popT1.append(x)
        elif x.skill_factor == 2:
            popT2.append(x)

    obj = 2

    popT1, frontnumbers = nondominatedsort(popT1, pop1, obj)
    popT1, minimums = diversity(popT1, frontnumbers, pop1, obj)
    print("T1 Sorting and Diversity Checks Done")
    popT2, frontnumbers = nondominatedsort(popT2, pop2, obj)
    popT2, minimums = diversity(popT2, frontnumbers, pop2, obj)
    print("T2 Sorting and Diversity Checks Done")
    print('Sorts and Checks complete, merging...')

    merge = []
    merge.extend(popT1)
    merge.extend(popT2)
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

        for x in parent:
            if x.skill_factor == 1:
                subpopT1.append(x.rnvec)
            elif x.skill_factor == 2:
                subpopT2.append(x.rnvec)

        subpops.append(np.asarray(subpopT1))
        subpops.append(np.asarray(subpopT2))

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
            if parent[p1].skill_factor == parent[p2].skill_factor:
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
                                    test_data_arr,
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

        for x in intpopulation:
            if x.skill_factor == 1:
                IntpopT1.append(x)
            elif x.skill_factor == 2:
                IntpopT2.append(x)


        T1_pop = IntpopT1.__len__()
        T2_pop = IntpopT2.__len__()

        IntpopT1, frontnumbers = nondominatedsort(IntpopT1, T1_pop, obj)
        IntpopT1, minimums = diversity(IntpopT1, frontnumbers, T1_pop, obj)

        IntpopT2, frontnumbers = nondominatedsort(IntpopT2, T2_pop, obj)
        IntpopT2, minimums = diversity(IntpopT2, frontnumbers, T2_pop, obj)

        print('Gen Number: ', generation, " - Acquiring unique parents and offspring for merging ... ")

        population = get_final_population([IntpopT1, IntpopT2], [pop1, pop2])

        print('Gen Number: ', generation, " - Merge Complete, ND Sorting for Hypervolume Calculation ... ")

        fit1 = []
        fit2 = []

        nd_fit1 = [] #for hv
        nd_fit2 = []

        for x in population:
            if x.skill_factor == 1:
                fit1.append(x.objs_T1)
                if x.front == 1: #non-dominated front solutions
                    nd_fit1.append(x.objs_T1)
            elif x.skill_factor == 2:
                fit2.append(x.objs_T2)
                if x.front == 1:
                    nd_fit2.append(x.objs_T2)


        nd_fit1 = np.array(nd_fit1)
        ref_point = np.array([5,21834.787109375]) * 1.1
        hv_fit1 = hypervolume(nd_fit1, ref_point)
        # hv_fit1 = hv.compute(ref_point)
        hv_fit1 /= (21834.787109375*5)

        nd_fit2 = np.array(nd_fit2)
        ref_point = np.array([5,21834.787109375]) * 1.1
        hv_fit2 = hypervolume(nd_fit2, ref_point)
        # hv_fit2 = hv.compute(ref_point)
        hv_fit2 /= (21834.787109375*5)

        gen_masks = []
        for x in population:
            gen_masks.append(x.rnvec)
        all_gen_masks.append(gen_masks)
        print('Gen Number: ', generation, " - Complete. Saving results.")

        rep_fit1.append(fit1)  # all fitness values ---
        rep_fit2.append(fit2)  # (IGD1)

        rep_NDfit1.append(nd_fit1)
        rep_NDfit2.append(nd_fit2)

        rep_hv1.append(hv_fit1)
        rep_hv2.append(hv_fit2)

        end_gen = timer()
        time_gen = end_gen - startall_gen

        rep_time.append(time_gen)

        print("============================= Multitask =====================================")
        print("---------- Hypervolumes ------------")
        print("Task 1 hypervolume: ", hv_fit1, "Task 2 hypervolume: ", hv_fit2)
        print("---------- Task 1 Fitness ------------")
        print(fit1[0:20])
        print("Generation", generation, 'Average Fitness Task 1', np.mean(np.array(fit1), axis=0)[0], 'Average Memory Task 1', np.mean(np.array(fit1), axis=0)[1])
        print("---------- Task 2 Fitness ------------")
        print(fit2[0:20])
        print("Generation", generation, 'Average Fitness Task 2', np.mean(np.array(fit2), axis=0)[0], 'Average Memory Task 2', np.mean(np.array(fit2), axis=0)[1])

    if final_finetune_switch == 1:
        finetuned_results = final_finetune(model_arr,
                                    [fit1, fit2],
                                    train_data_arr_unloaded,
                                    test_data_arr,
                                    pop,
                                    mask_sizing,
                                    population,
                                    hyperparams['subpop'],
                                    finetune_epochs)
        with open(path_save + "finetuned_results.pkl", 'wb') as f:
            pickle.dump(finetuned_results, f)

    temp_masks = []
    for i in range(pop):
        temp_masks.append(population[i].rnvec)

    endall = timer()
    time_all = endall - startall

    with open(path_save + "population.pkl", 'wb') as f:
        pickle.dump(population, f)

    with open(path_save + "progenitor_results.pkl", 'wb') as f:
        pickle.dump(progenitor_results, f)

    with open(path_save + "rep_fit1.pkl", 'wb') as f:
        pickle.dump(rep_fit1, f)
    with open(path_save + "rep_fit2.pkl", 'wb') as f:
        pickle.dump(rep_fit2, f)


    with open(path_save + "rep_NDfit1.pkl", 'wb') as f:
        pickle.dump(rep_NDfit1, f)
    with open(path_save + "rep_NDfit2.pkl", 'wb') as f:
        pickle.dump(rep_NDfit2, f)



    with open(path_save + "rep_hv1.pkl", 'wb') as f:
        pickle.dump(rep_hv1, f)
    with open(path_save + "rep_hv2.pkl", 'wb') as f:
        pickle.dump(rep_hv2, f)


    with open(path_save + "rep_time.pkl", 'wb') as f:
        pickle.dump(rep_time, f)

    with open(path_save + "time_all.pkl", 'wb') as f:
        pickle.dump(time_all, f)

    with open(path_save + "all_gen_masks.pkl", 'wb') as f:
        pickle.dump(all_gen_masks, f)

    # with open(path_save + "all_gen_RMPS.pkl", 'wb') as f:
    #     pickle.dump(all_gen_RMPS, f)

    print("Results saved")
    print("Ciao!")
    print((datetime.now() - a).seconds)
    # # ----------------------------------------------------------------------------------------------------------------------
