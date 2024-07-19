import numpy as np
import math

def nondominatedsort(population, pop, no_objs):
    #minimums = np.zeros(no_objs) # dummy store
    count = 0
    frontnumbers = []
    for i in range(pop):
       # print("iloop", i)
        for j in range(pop):
            if i == j:
                continue
            better=0
            worse = 0
            for k in range(no_objs):
                if population[0].skill_factor == 1:
                    if population[i].objs_T1[k] < population[j].objs_T1[k]:
                        better = 1
                    elif population[i].objs_T1[k] > population[j].objs_T1[k]:
                        worse = 1
                elif population[0].skill_factor == 2:  #skill_factor ==2
                    if population[i].objs_T2[k] < population[j].objs_T2[k]:
                        better = 1
                    elif population[i].objs_T2[k] > population[j].objs_T2[k]:
                        worse = 1
                elif population[0].skill_factor == 3:
                    if population[i].objs_T3[k] < population[j].objs_T3[k]:
                        better = 1
                    elif population[i].objs_T3[k] > population[j].objs_T3[k]:
                        worse = 1
                else:
                    if population[i].objs_T4[k] < population[j].objs_T4[k]:
                        better = 1
                    elif population[i].objs_T4[k] > population[j].objs_T4[k]:
                        worse = 1
            if worse==0 and better>0:
                # print(population[i].dominatedset)
                population[i].dominatedset.append(j)
                population[i].dominatedsetlength = population[i].dominatedsetlength + 1
                population[j].dominationcount = population[j].dominationcount + 1
            elif better==0 and worse>0:
                # print(population[i].dominatedset)
                population[j].dominatedset.append(i)
                population[j].dominatedsetlength = population[j].dominatedsetlength + 1
                population[i].dominationcount = population[i].dominationcount + 1

        if population[i].dominationcount==0:
            population[i].front=1
            count = count+1
    #print("for loop done")
    frontnumbers.append(count)
    front = 0
    while count>0:
        count = 0
        front = front+1
        for i in range(pop):
            if population[i].front == front:
                for j in range(population[i].dominatedsetlength):
                    ind = population[i].dominatedset[j]
                    population[ind].dominationcount=population[ind].dominationcount-1
                    if population[ind].dominationcount == 0:
                        population[ind].front = front + 1
                        count = count + 1
        frontnumbers.append(count)

    #tag = 100000
    #return tag
    return population, frontnumbers

def diversity (population, frontnumbers, pop, no_objs):
    minimums = np.zeros(no_objs) # dummy store
    for i in range(pop):
        population[i].CD = 0

    #sort according to front
    front_val = []
    for x in population:
        front_val.append(x.front)

    index = np.argsort(front_val)
    sorted_pop = []
    for i in range(pop):
        sorted_pop.append(population[index[i]])

    population = sorted_pop
    currentind = 0
    for i in range(population[pop-1].front):
        subpopulation = population[currentind:currentind+frontnumbers[i]] #currentind+1
        minima = np.zeros(no_objs)  #[1,
        x = np.zeros(frontnumbers[i]) #[1,frontnumbers[i]]
        for j in range(no_objs):
            for k in range(frontnumbers[i]):
                if population[0].skill_factor ==1:
                    x[k] = subpopulation[k].objs_T1[j]  #IndexError: list index out of range
                elif population[0].skill_factor ==2:
                    x[k] = subpopulation[k].objs_T2[j]
                elif population[0].skill_factor ==3:
                    x[k] = subpopulation[k].objs_T3[j]
                else: # population[0].skill_factor ==3:
                    x[k] = subpopulation[k].objs_T4[j]

            index = np.argsort(x)
            temp_x = []
            for t in range(x.__len__()):
                temp_x.append(x[index[t]])
            x = temp_x
            #print("sorted x", x)
            #sort subpop
            sorted_subpop = []
            for t in range(subpopulation.__len__()):
                sorted_subpop.append(subpopulation[index[t]])
            subpopulation = sorted_subpop

            if population[0].skill_factor == 1:
                minima[j] = subpopulation[0].objs_T1[j]
                max = subpopulation[frontnumbers[i]-1].objs_T1[j]
            elif population[0].skill_factor == 2:
                minima[j] = subpopulation[0].objs_T2[j]
                max = subpopulation[frontnumbers[i]-1].objs_T2[j]
            elif population[0].skill_factor == 3:
                minima[j] = subpopulation[0].objs_T3[j]
                max = subpopulation[frontnumbers[i] - 1].objs_T3[j]
            else:
                minima[j] = subpopulation[0].objs_T4[j]
                max = subpopulation[frontnumbers[i] - 1].objs_T4[j]

            subpopulation[0].CD = math.inf
            subpopulation[frontnumbers[i]-1].CD = math.inf
            #print("x", x)
            a = (max - minima[j])
            if a == 0:
                a = 1
            normobj = np.divide((x - minima[j]), a) ##TODO----- RuntimeWarning: invalid value encountered in true_divide normobj = np.divide((x - minima[j]), (max - minima[j]))
            #print("mini[j]", minima[j], "max", max, "(x - minima[j]", x - minima[j], "max - minima[j]", max - minima[j], "normobj", normobj)  #nan - (div by 0 -just one sol in the front??) handled or ignored in sorting

            #if normobj[0] == math.nan:
             #   print("nan identified")

            for k in range(1,frontnumbers[i]-2): #2:frontnumbers(i) - 1
                subpopulation[k].CD = subpopulation[k].CD + (normobj[k + 1] - normobj[k - 1])

        #print("obj over...next obj", currentind, frontnumbers[i])
        if i==1:
            minimums = minima

        #sort via CD
        # sort according to front
        CD_val = []
        for x in subpopulation:
            CD_val.append(x.CD)

        CD_val = np.multiply(-1, CD_val) #descending order search
        index = np.argsort(CD_val)
        sorted_pop = []
        for t in range(subpopulation.__len__()):
            sorted_pop.append(subpopulation[index[t]])
        subpopulation = sorted_pop

       # print("current index before update", currentind, "frontnumbers[i]", frontnumbers[i], "currentind+frontnum", currentind+frontnumbers[i])
        population[currentind:currentind+frontnumbers[i]] = subpopulation #TODO  IndexError: list index out of range
        currentind = currentind + frontnumbers[i]
       # print("current index after update", currentind)
    for i in range(pop):
        population[i].rank = i

    return population , minimums
