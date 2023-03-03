import copy
import math
import os
import warnings

import networkx as nx
import numpy as np
from copy import deepcopy

import pandas as pd

import DataVectors
import functions_copy as fc
import parameters
from parameters import ITERATIONS, n, p, q, tf, gSize, pQueryOracle, C, path, nCommunities, mu_bad
from DataVectors import initialMedian, median, medianOther, times, firstStepValues, secondStepValues

# Suppress future warnings, comment if code not running due to  this
warnings.simplefilter(action='ignore', category=FutureWarning)

"Clean the output folder"
if os.path.exists(path):
    fc.clean_fldr()
else:
    os.makedirs(path)

"Create the files to save the data to"
DataVectors.create_files()

"Set up the graph"

# G is generated through a stochastic block model
probs = [[p[0], q], [q, p[1]]]  # TODO make it robust to multiple communities
groundG = nx.stochastic_block_model(gSize, probs, seed=65)  # Generate graph
mc_edges, mc, minDegree, rValues, excessInnerDegree = fc.find_cut(groundG)  # Obtain the edges of the inter-blocks cut
listOfNodes = list(groundG)

print(f'n = {n}\n'
      f'n1 = n2 = n\n'
      f'a = n1/(2n)\n'
      f'r = Pr[query oracle]: {round(pQueryOracle, 5)} > {round(p[0]*tf/(0.5*2*(2*n-1)), 5)} = p*|F|/((1-a)*2*(2n-1))\n'
      f'It satisfies p*n*|F| < 2n(2n-1)r(1-a) (E[edges from nodes on the other community] > E[edges to faulty nodes])\n'
      f'However, it cannot satisfy r*2*n < 1 (r*2*n = {round(pQueryOracle*2*n, 5)})\n'
      f'c = {C}\n'
      f'p = {round(parameters.p[0], 4)} -- c * log(n)/sqrt(n)\n'
      f'q = {round(parameters.q, 4)} -- c * ((1 - delta) * (n - sqrt(n)) * log(n)) / (n ** (5 / 2))\n'
      f'C = {{e in E: e is xCommunities}} -- F = {{i in V: i faulty}}\n'
      f'E[|C|] = {round(parameters.q * n ** 2, 4)} -- q * n^2\n'
      f'Sampled cut |C|: {mc}\n'
      f'Degrees in the cut: {rValues} -- {{degree: # nodes}}\n'
      f'Min d: {minDegree}')

fn = [tf, tf]  # [math.floor(tf / 2), tf - math.floor(tf / 2)]  # The faulty nodes per community

print(f'|F| = |C|: {2 * tf} -- {round((2 * tf) / sum(gSize) * 100, 2)} % of total nodes')

fc.display_graph(groundG, 0, 'Graph', path)

with open(path + 'paramsAndMedians.txt', 'a') as f:
    f.write(f'n of faulty nodes: {tf}'
            f'\nN of nodes: {parameters.gSize}'
            f'\nThe Minimum degree: {minDegree}'
            f'\nThe XCommunity cut size: {mc}\n'
            f'\n*********************************************\n')

for iteration in range(ITERATIONS):
    G = copy.deepcopy(groundG)
    # print(f'ITERATION: {iteration}\n')

    "Initial Step"
    fn_indices = fc.choose_fn_idx(fn, mc_edges)
    nu, values, G, goodValues = fc.assign_values(G, fn, fn_indices)
    nn_indices = [[], []]
    for c in range(nCommunities):
        for i in list(G):
            if i not in fn_indices[c]:
                if i < n:
                    nn_indices[c] += [i]
    faultyEdges = fc.count_fn_edges(fn_indices, G)
    normalEdges = fc.count_nodes_edges(nn_indices, fn_indices, G)
    print(f'Number faulty edges = |{{d_i: i in F}}|: {faultyEdges}')
    print(f'Number normal edges = |{{d_i: i in V\F}}|: {[ne/2 for ne in normalEdges]}')

    "Save the data"
    if ITERATIONS == 1:
        fc.save_data(values, 'Network Values - First Step.xlsx', 0)

    "Calculate the initial median"
    initialMedian.append([fc.mMedian(list(values[i].values())) for i in range(nCommunities)])
    l = {}
    [l.update(values[i]) for i in range(nCommunities)]
    medianTotal = fc.mMedian(list(l.values()))
    medianOfMedian = fc.mMedian(initialMedian)

    "Calculate the potentials"
    # print('Calculate the potentials...')
    community_potential = fc.get_comm_pot(goodValues, G)
    global_potential = fc.get_glob_pot(values, G, fn_indices)

    # print(f'Obtain community median...')
    condition = list(np.abs(community_potential[0]))
    for c in range(1, nCommunities):
        condition += list(np.abs(community_potential[c]))

    t, counter = 1, 0
    badValuesIdx = fn_indices[0]
    for c in range(nCommunities):
        badValuesIdx += fn_indices[c]
    while any(condition) > 0.001 and counter < 30 * int(math.log(n)):  # and distanceChange > 0.001:
        temp, nodeAttr = copy.deepcopy(values), {}  # TODO check this deepcopy, please let it be non influential
        for x in list(G):
            neighVals = []
            if x not in badValuesIdx:
                neighbors = list(G.adj[x])
                for c in range(nCommunities):
                    neighVals += [values[c][j] for j in neighbors if j in values[c].keys()]
                for c in range(nCommunities):
                    if x in values[c].keys():
                        med = fc.mMedian(neighVals)
                        temp[c].update({x: med})
                        goodValues[c].update({x: med})
        for c in range(nCommunities):
            nodeAttr.update(temp[c])
        nx.set_node_attributes(G, nodeAttr, 'Values')

        values = temp

        # Update potentials
        community_potential = fc.get_comm_pot(goodValues, G)

        condition = []
        for c in range(nCommunities):
            condition += list(np.abs(community_potential[c]))

        "Save the data"
        if ITERATIONS == 1:
            fc.save_data(values, 'Network Values - First Step.xlsx', t)

        t += 1
        counter += 1

    median.append([fc.mMedian(list(values[i].values())) for i in range(nCommunities)])

    # print(f'Obtain the external medians...')
    otherG = G.copy()
    tSecond = 0
    threshold = {}

    goodValsOther, nodeAttr = [], {}
    valsOther = deepcopy(values)
    threshold = deepcopy(median[-1])

    valsOther, nEdges, xEdges = fc.update_step(otherG, listOfNodes, valsOther, threshold, goodValsOther,
                                               badValuesIdx, tSecond)

    print(f'We sampled {nEdges} unique edges')
    print(f'Of which, {xEdges} are cross cut')
    print(f'p*n*F = {round(parameters.p[0] * n * tf, 2)} < {round(2*n*(2*n-1)*pQueryOracle*0.5, 2)} = 2n(2n-1)r(1-a)')

    "Save the data"
    if ITERATIONS == 1:
        fc.save_data(valsOther, 'Network Values - Second Step.xlsx', 0)

    tSecond += 1

    community_potential = fc.get_comm_pot(goodValsOther, otherG)

    condition = []
    for c in range(nCommunities):
        condition += list(np.abs(community_potential[c]))
        nodeAttr.update(valsOther[c])

    nx.set_node_attributes(otherG, nodeAttr, 'Values')

    counter = 0
    while any(condition) > 0.001 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
        temp_b = deepcopy(valsOther),
        goodValsOther, nodeAttr = [], {}
        valsOther, nEdges, xEdges = fc.update_step(otherG, listOfNodes, valsOther, threshold,
                                                   goodValsOther, badValuesIdx, tSecond)

        print(f'We sampled {nEdges} unique edges')
        print(f'Of which, {xEdges} are cross cut')
        # print(f'p*n*F = {round(parameters.p[0] * n * tf)} > {2*n*(2*n-1)*pQueryOracle*0.5} = 2n(2n-1)r(1-a)')
        community_potential = fc.get_comm_pot(goodValsOther, otherG)

        condition = []
        for c in range(nCommunities):
            condition += list(np.abs(community_potential[c]))
            nodeAttr.update(valsOther[c])
        nx.set_node_attributes(otherG, nodeAttr, 'Values')

        "Save the data"
        if ITERATIONS == 1:
            fc.save_data(valsOther, 'Network Values - Second Step.xlsx', tSecond)

        tSecond += 1
        counter += 1

    # print('External medians obtained.')
    medianOther.append([round(fc.mMedian(list(valsOther[i].values())), 3) for i in range(nCommunities)])
    times.append([t, tSecond])

failureIntra = [sum([median[j][k] == mu_bad for j in range(ITERATIONS)]) / ITERATIONS for k in range(nCommunities)]
failureXComm = [sum([medianOther[j][k] == mu_bad for j in range(ITERATIONS)]) / ITERATIONS for k in range(nCommunities)]

times = np.array(times)
times = np.mean(times, axis=0)
times = [round(i, 3) for i in times]

print('Save failure rates')
with open(path + 'paramsAndMedians.txt', 'a') as f:
    f.write(f'The failure rate 1st step: {failureIntra}\n')
    f.write(f'The failure rate 2nd step: {failureXComm}\n')
    f.write(f'The average times: {times}')

print('Done.')
