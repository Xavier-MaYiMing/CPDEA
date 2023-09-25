#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/22 10:18
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : ARMOEA.py
# @Statement :
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist


def cal_obj(pop, nobj):
    # DTLZ1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def cal_dis(objs, refs):
    # calculate the distance between each solution to each adjusted reference point
    npop = objs.shape[0]
    (nref, nobj) = refs.shape
    objs = np.where(objs > 1e-6, objs, 1e-6)
    refs = np.where(refs > 1e-6, refs, 1e-6)

    # adjust the location of each reference point
    cosine = 1 - cdist(objs, refs, 'cosine')
    normP = np.sqrt(np.sum(objs ** 2, axis=1))
    normR = np.sqrt(np.sum(refs ** 2, axis=1))
    d1 = np.tile(normP.reshape((npop, 1)), (1, nref)) * cosine
    d2 = np.tile(normP.reshape((npop, 1)), (1, nref)) * np.sqrt(1 - cosine ** 2)
    nearest = np.argmin(d2, axis=0)
    refs = refs * np.tile((d1[nearest, np.arange(nref)] / normR).reshape((nref, 1)), (1, nobj))
    return cdist(objs, refs)


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    return rank


def update_ref(arch, W, Range):
    # update reference points
    # Step 1. Delete duplicated and dominated solutions
    (narch, nobj) = arch.shape
    nref = W.shape[0]
    ind = nd_sort(arch) == 1
    arch = arch[ind]
    (arch, ind) = np.unique(arch, axis=0, return_index=True)

    # Step 2. Update the ideal point
    if np.any(Range):
        Range[0] = np.min((Range[0], np.min(arch, axis=0)), axis=0)
    elif np.any(arch):
        Range = np.zeros((2, nobj))
        Range[0] = np.min(arch, axis=0)
        Range[1] = np.max(arch, axis=0)

    # Step 3. Update archive and reference points
    if arch.shape[0] <= 1:
        refs = W
    else:
        # Step 3.1. Find contributing solutions and valid weight vectors
        tarch = arch - Range[0]
        W *= (Range[1] - Range[0])
        dis = cal_dis(tarch, W)
        nearest1 = np.argmin(dis, axis=0)
        con_sols = np.unique(nearest1)  # contributing solutions
        nearest2 = np.argmin(dis, axis=1)
        valid_W = np.unique(nearest2[con_sols])  # valid reference points

        # Step 3.2. Update archive
        choose = np.full(tarch.shape[0], False)
        choose[con_sols] = True
        cosine = 1 - cdist(tarch, tarch, 'cosine')
        np.fill_diagonal(cosine, 0)
        while np.sum(choose) < min(3 * nref, tarch.shape[0]):
            unselected = np.where(~choose)[0]
            best = np.argmin(np.max(cosine[~choose][:, choose], axis=1))
            choose[unselected[best]] = True
        arch = arch[choose]
        tarch = tarch[choose]

        # Step 3.3. Update reference points
        refs = np.concatenate((W[valid_W], tarch), axis=0)
        choose = np.concatenate((np.full(valid_W.shape[0], True), np.full(tarch.shape[0], False)))
        cosine = 1 - cdist(refs, refs, 'cosine')
        np.fill_diagonal(cosine, 0)
        while np.sum(choose) < min(nref, refs.shape[0]):
            unselected = np.where(~choose)[0]
            best = np.argmin(np.max(cosine[~choose][:, choose], axis=1))
            choose[unselected[best]] = True
        refs = refs[choose]
    return arch, refs, Range


def mating_selection(pop, objs, refs, Range):
    # mating selection
    (npop, nvar) = pop.shape
    dis = cal_dis(objs - Range[0], refs)
    convergence = np.min(dis, axis=1)
    rank = np.argsort(dis, axis=0)
    dis = np.sort(dis, axis=0)

    # Step 1. Calculate the fitness of noncontributing solutions
    noncontributing = np.full(npop, True)
    noncontributing[rank[0]] = False
    metric = np.sum(dis[0]) + np.sum(convergence[noncontributing])
    fitness = np.full(npop, np.inf)
    fitness[noncontributing] = metric - convergence[noncontributing]

    # Step 2. Calculate the fitness of contributing solutions

    # Step 3. Binary tournament selection
    nm = npop if npop % 2 == 0 else npop + 1  # mating pool size
    mating_pool = np.zeros((nm, nvar))


def environmental_selection(pop, objs, refs, Range, num):
    # environmental selection
    pass


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < 1 / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def main(npop, iter, lb, ub, nobj=3, eta_c=20, eta_m=20):
    """
    The main loop
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of the objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    W = reference_points(npop, nvar)  # original reference points
    arch, refs, Range = update_ref(objs, W, [])  # archive, reference points, ideal and nadir points

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 100 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation

        # Step 2.2. Environmental selection


if __name__ == '__main__':
    import scipy.io as scio
