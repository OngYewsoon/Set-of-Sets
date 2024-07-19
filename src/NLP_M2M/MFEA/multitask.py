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
from model import Model, Custom_Dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sacrebleu.metrics import BLEU
from torchinfo import summary

from chromosome import Chromosome
from compare_solutions import *
from operators import *

import datasets
import transformers
from transformers.optimization import Adafactor, AdafactorSchedule
import random

DEVICE_1 = torch.device('cuda:0')
DEVICE_2 = torch.device('cpu')

def count_active_params(state_dict):
    total = 0
    for i in state_dict:
        flattened = torch.flatten(state_dict[i])
        total += torch.count_nonzero(flattened)
    return total.detach().item()

def test_with_size(model, mask, mask_sizing, test_dataloader):
    model.eval()
    model.apply_mask(mask, mask_sizing)
    loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            loss += model.forward_train(batch).loss.detach().item()
    size = count_active_params(model.state_dict())
    model.revert_weights()
    return loss/len(test_dataloader), size

def test_without_size(model, test_dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            loss += model.forward_train(batch).loss.detach().item()
    return loss/len(test_dataloader)

def convert_to_string(tokens, test_code):
    prohib = [1, 2, 128022, 128020, 128017]
    if test_code:
        return [' '.join([str(i) for i in tokens.tolist() if i not in prohib])]
    else:
        return ' '.join([str(i) for i in tokens.tolist() if i not in prohib])

def stringify(tokens, test_code):
    return [convert_to_string(tokens[i], test_code) for i in range(tokens.shape[0])]

def test_BLEU(model, test_dataloader, metric, lang_code):
    start = time.time()
    model.eval()
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for batch in test_dataloader:
            preds = model.forward_eval(batch, lang_code)
            all_preds += stringify(preds[:, 0:16], False)
            all_refs += stringify(batch['y'], True)
        score = metric.compute(predictions = all_preds, references = all_refs)['score']
    return score

def train_loop(model,
                mask, mask_sizing,
              extloader,
              evalloader,
              epochs,
              steps=200):
    optim= Adafactor(model.parameters(),
                    scale_parameter=True,
                    relative_step=True,
                    warmup_init=True,
                    lr=None)

    best_loss = np.inf
    saved_state = None
    for epoch in range(epochs):
        model.train()
        for name, param in model.named_parameters():
            if param.requires_grad is False:
                param.requires_grad =  True

        train(model,
            mask, mask_sizing,
              extloader,
              optim,
              steps)

        torch.cuda.empty_cache()

        loss = test_without_size(model, evalloader)
        torch.cuda.empty_cache()

        if loss < best_loss:
            best_loss = loss
            saved_state = copy.deepcopy(model.state_dict())
        return saved_state

def train(model,
            mask, mask_sizing,
          extloader,
          optim,
          steps):
    optim.zero_grad()
    for i in range(steps):
        batch = extloader.sample_batch()
        loss = model.forward_train(batch).loss
        loss.backward()
        if (i+1)%10 == 0:
            optim.step()
            optim.zero_grad()
            model.apply_mask(mask, mask_sizing)

def finetune(model, mask, mask_sizing, metric, lang_code, extloader, evalloader, epochs):
    saved_state = train_loop(model,
                            mask, mask_sizing,
                              extloader,
                              evalloader,
                              epochs)
    model.load_state_dict(saved_state)
    bleu = test_BLEU(model, evalloader, metric, lang_code)
    size = count_active_params(model.state_dict())
    return bleu, size

def final_finetune(model, fitnesses, train_data_arr, test_data_arr, pop, mask_sizing, offspring, subpop_size, finetune_epochs):

    results = {}
    task_names = ['cs_only', 'de_only']
    metric = datasets.load_metric('sacrebleu')
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
            model.apply_mask(mask, mask_sizing)
            lang_code = task_name.split('_')[0]
            bleu, size = finetune(model, mask, mask_sizing,
                                metric,
                                lang_code,
                                train_data_arr[offspring[i].skill_factor - 1],
                                test_data_arr[offspring[i].skill_factor - 1],
                                finetune_epochs)
            print('BLEU: ', bleu_score, ', Memory:', size)

            model.revert_weights()
            if task_name not in results:
                results.update({task_name:[(bleu_score, mem)]})
            else:
                curr = results[task_name]
                curr.append((bleu_score, mem))
                results.update({task_name:curr})
    return results


def reset(population, pop):
    for i in range(pop):
        population[i].dominationcount = 0
        population[i].dominatedset = []
        population[i].dominatedsetlength = 0
    return population

def initialize_and_evaluate_parents(progenitor_arr, model, data_arr, dims, pop, subpop_size, mask_sizing, pretrained):
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
        model.apply_mask(mask, mask_sizing)
        acc = test_without_size(model, data_arr[population[i].skill_factor - 1])
        mem = count_active_params(model.state_dict())
        population[i].Evaluate([acc, mem])
        model.revert_weights()
        progenitor_results.append((acc, mem))
    print("--------------- Progenitor Evaluation Done -------------")

    return population, progenitor_results

def evaluate_offspring(model, train_data_arr, pop, mask_sizing, offspring):
    results = {}
    for i in range(pop):
        mask = offspring[i].rnvec
        acc, mem = test_with_size(model,
                            mask,
                            mask_sizing,
                            train_data_arr[offspring[i].skill_factor - 1])
        offspring[i].Evaluate([acc, mem])
    return offspring

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

def multitaskMFEA(target_dir,
                run_number,
                model,
                train_data_arr,
                finetune_data_arr,
                test_data_arr,
                hyperparams,
                progenitor_arr,
                rmp_setting, pretrained, final_finetune_switch, MFEA_II, finetune_epochs):

    dims = 67584

    mask_sizing = size_mask(model.return_model_state())
    # print("Testing Generalists")
    # for i,_ in enumerate(train_data_arr):
    #     mask = np.ones(dims)
    #     loss = test_without_size(model, test_data_arr[i])
    #     mem = count_active_params(model.state_dict())
    #     print('Generalist: ', i, ', Loss: ', loss, ', Size: ',mem)

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

    pop = pop1 + pop2 # has to be even sum
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
    population, progenitor_results = initialize_and_evaluate_parents(progenitor_arr, model, train_data_arr, dims, pop, pop1, mask_sizing, pretrained)

    # population, progenitor_results = initialize_and_evaluate_parents(model, train_data_arr, dims, pop, pop1, mask_sizing)
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
        offspring = evaluate_offspring(model,
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

        gen_masks = {}
        for x in population:
            if x.skill_factor not in gen_masks:
                gen_masks.update({x.skill_factor:[]})

            curr = gen_masks[x.skill_factor]
            curr.append({'rnvec':x.rnvec, 'objs_T1':x.objs_T1, 'objs_T2':x.objs_T2})
            gen_masks.update({x.skill_factor:curr})

            if x.skill_factor == 1:
                fit1.append(x.objs_T1)
                if x.front == 1: #non-dominated front solutions
                    nd_fit1.append(x.objs_T1)

            elif x.skill_factor == 2:
                fit2.append(x.objs_T2)
                if x.front == 1:
                    nd_fit2.append(x.objs_T2)

        all_gen_masks.append(gen_masks)

        with open(path_save + "mask_checkpoint.pkl", 'wb') as f:
            pickle.dump(gen_masks, f)


        nd_fit1 = np.array(nd_fit1)
        ref_point = np.array([50,879563776]) * 1.1
        hv_fit1 = hypervolume(nd_fit1, ref_point)
        # hv_fit1 = hv.compute(ref_point)
        hv_fit1 /= (879563776*50)

        nd_fit2 = np.array(nd_fit2)
        ref_point = np.array([50,879563776]) * 1.1
        hv_fit2 = hypervolume(nd_fit2, ref_point)
        # hv_fit2 = hv.compute(ref_point)
        hv_fit2 /= (879563776*50)


        # gen_masks = []
        # for x in population:
        #     gen_masks.append(x.rnvec)
        # all_gen_masks.append(gen_masks)
        print('Gen Number: ', generation, " - Complete. Saving results.")

        rep_fit1.append(fit1)
        rep_fit2.append(fit2)

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
        finetuned_results = final_finetune(model,
                                    [fit1, fit2],
                                    finetune_data_arr,
                                    test_data_arr,
                                    pop,
                                    mask_sizing,
                                    population,
                                    hyperparams['subpop'],
                                    finetune_epochs)
        with open(path_save + "finetuned_results.pkl", 'wb') as f:
            pickle.dump(finetuned_results, f)
        print(finetuned_results)
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

    print("Results saved")
    print((datetime.now() - a).seconds)
    # # ----------------------------------------------------------------------------------------------------------------------
