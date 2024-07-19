from __future__ import division
import random
import numpy as np
import math
import numpy as np
import pickle


'''
testdata = torch.zeros([1761, 1])
var = np.random.random(1761)
for i in range(testdata.size(0)):
    testdata[i]=var[i]

'''

class Chromosome:

    def __init__(self):
        self.rnvec = None
        #self.factorial_costs = [math.inf, math.inf]  #2 objective
        #self.factorial_ranks = None
        #self.scalar_fitness = None
        self.skill_factor = None
        self.objs_T1 = [None, None]  #2 objective..
        self.objs_T2 = [None, None]
        self.objs_T3 = [None, None]
        self.objs_T4 = [None, None]
        self.convio = None
        self.front = None
        self.CD = None
        self.rank = None
        self.dominationcount = 0
        self.dominatedset = []
        self.dominatedsetlength = 0

    def initialize(self, dim, lim1, lim2):
        np.random.seed()

        sample = np.random.uniform(lim1, lim2, 1)[0]
        num_indices = np.int_(sample*dim)
        indices = np.random.choice(range(0, dim), num_indices, replace=False)
        base = np.random.uniform(0, 0.5, dim)
        base[indices] += 0.5
        self.rnvec = base

    def collapse_prevention(self, sizing):
        start = 0
        for key, value in sizing.items():
            end = start + value
            if sum(np.round(self.rnvec[start:end])) == 0:
                choice = np.random.choice(range(start, end), 1)[0]
                self.rnvec[choice] = 1
            start = end

    def Evaluate(self, fit):
        if self.skill_factor == 1:
            self.objs_T1[0] = fit[0]
            self.objs_T1[1] = fit[1]
        elif self.skill_factor == 2:
            self.objs_T2[0] = fit[0]
            self.objs_T2[1] = fit[1]
        elif self.skill_factor == 3:
            self.objs_T3[0] = fit[0]
            self.objs_T3[1] = fit[1]
        elif self.skill_factor == 4:
            self.objs_T4[0] = fit[0]
            self.objs_T4[1] = fit[1]

    def print_objs(self):
        print(self.objs_T1)
        print(self.objs_T2)
        print(self.objs_T3)
        print(self.objs_T4)
    # def crossover(self, p1, p2, dims, muc):
    #
    #     child = np.zeros(dims)
    #     randlist = np.random.rand(dims)
    #     for i in range(dims):
    #         if randlist[i] <= 0.5:
    #             k = (2 * randlist[i]) ** (1 / (muc + 1))
    #         else:
    #             k = (1 / (2 * (1 - randlist[i]))) ** (1 / (muc + 1))
    #         child[i] = 0.5 * (((1 + k) * p1[i]) + (1 - k) * p2[i])
    #
    #         ## TODO -------========
    #         if child[i] > 1:
    #             child[i] = 1
    #
    #         if child[i] < 0:
    #             child[i] = 0
    #     self.rnvec = np.round(child)
    #
    # def mutate(self, mum, dims):
    #     p = self.rnvec
    #     child = self.rnvec
    #     for i in range(dims):
    #         if np.random.rand() < 1 / dims:
    #             u = np.random.rand()
    #             if u <= 0.5:
    #                 delta = (2 * u) ** (1 / (1 + mum)) - 1
    #                 child[i] = p[i] + delta * p[i]
    #             else:
    #                 delta = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
    #                 child[i] = p[i] + delta * (1 - p[i])
    #     self.rnvec = child

#
# def func1(var):
#     # dim = 10
#     L1 = -100 * np.ones(var.__len__())
#     U1 = 100 * np.ones(var.__len__())
#     L1[0] = 0
#     U1[0] = 1
#
#     #print("unmapped var---", var)
#     xtemp = var
#     var = L1 + np.multiply(xtemp, (U1 - L1))
#
#     #print("mapped var", var)
#     y = var[1:]
#     #print(y)
#     gx = 1+sum(np.multiply(y, y))
#     ob1 = gx * math.cos(0.5 * math.pi * var[0])
#     ob2 = gx * math.sin(0.5 * math.pi * var[0])
#
#     conv = 0 # np.random.rand(1)
#     return ob1, ob2, conv
#
# def func2(self):
#     #tag = 1
#     ob1 = 100
#     ob2 = 1000
#     conv = 1
#
#     return ob1*2,ob2*2,conv*2
