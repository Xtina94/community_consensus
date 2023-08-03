import copy
import math
import os
import warnings
from copy import deepcopy

import networkx as nx
import numpy as np

import DataVectors
import functions_size as fc
import parameters
from DataVectors import initialMedian, median, medianOther, times
from parameters import ITERATIONS, n, p, q, gSize, pQueryOracle, path, nCommunities, mu_bad

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
groundG = nx.stochastic_block_model(gSize, probs, seed=55)  # Generate graph
mc_edges, mc, minDegree, rValues, excessInnerDegree = fc.find_cut(groundG)  # Obtain the edges of the inter-blocks cut
listOfNodes = list(groundG)

print(f'p/q: {parameters.p}/{parameters.q}\n'
      f'The length of the cut: {mc}\n'
      f'The degrees on the cut: {rValues}\n'
      # f'The excess inner degrees: {excessInnerDegree}\n'
      f'The min degree: {minDegree}\n'
      f'The pQueryOracle: {pQueryOracle}')

tf = 2 * mc  # top faulty nodes
fn = [math.floor(tf / 2), tf - math.floor(tf / 2)]  # The faulty nodes per community
redNodes = 50  # The amount of redundant nodes

print(f'n of faulty nodes: {tf}'
      f'\nn of redundant nodes: {redNodes}')

# fc.display_graph(groundG, 0, 'Graph', path)

with open(path + 'paramsAndMedians.txt', 'a') as f:
    f.write(f'n of faulty nodes: {tf}'
            f'\nN of nodes: {parameters.gSize}'
            f'\nn of redundant nodes: {redNodes}'
            f'\nThe Minimum degree: {minDegree}'
            f'\nThe XCommunity cut size: {mc}\n'
            f'\n*********************************************\n')

for iteration in range(ITERATIONS):
    G = copy.deepcopy(groundG)
    print(f'IT: {iteration}\n')

    "Initial Step"
    fn_indices = fc.choose_fn_idx(G, fn, mc_edges)
    values, G, goodValues = fc.assign_values(G, fn, fn_indices)

    "Save the data"
    if ITERATIONS == 1:
        fc.save_data(values, 'Network Tokens - First Step.xlsx', 0)

    "Calculate the potentials"
    # print('Calculate the potentials...')
    # community_potential = fc.get_comm_pot(goodValues, G)
    # global_potential = fc.get_glob_pot(values, G, fn_indices)
    #
    # print(f'Obtain community median...')
    # condition = list(np.abs(community_potential[0]))
    # for c in range(1, nCommunities):
    #     condition += list(np.abs(community_potential[c]))

    t, counter = 1, 0
    badValuesIdx = fn_indices[0]
    condition = np.ones(sum(gSize))
    rate = [[1], [1]]  # The max sum at each step

    for c in range(nCommunities):
        badValuesIdx += fn_indices[c]
    while any(condition) > 0.001 and counter < 10 * int(math.log(n, 2)):  # and distanceChange > 0.001:
        temp, nodeAttr = copy.deepcopy(values), {}
        for x in list(G):
            neighVals = []
            if x not in badValuesIdx:
                neighbors = list(G.adj[x])
                for c in range(nCommunities):
                    neighVals += [values[c][j] for j in neighbors if j in values[c].keys()]
                for c in range(nCommunities):
                    if x in values[c].keys():
                        tmp = neighVals.count(values[c][x])
                        if tmp > fn[c]:
                            flip = np.random.binomial(1, 1/2)
                            if flip:
                                up = temp[c][x] + 1
                                temp[c].update({x: up})
                                goodValues[c].update({x: up})
        for c in range(nCommunities):
            nodeAttr.update(temp[c])
        nx.set_node_attributes(G, nodeAttr, 'sizeTokens')

        condition = []
        for c in range(nCommunities):
            condition += list(np.array(list(temp[c].values())) - np.array(list(values[c].values())))
            rate[c].append(max(list(temp[c].values())))

        values = temp

        # Update potentials
        community_potential = fc.get_comm_pot(goodValues, G)

        # condition = []
        # for c in range(nCommunities):
        #     condition += list(np.abs(community_potential[c]))

        "Save the data"
        if ITERATIONS == 1:
            fc.save_data(values, 'Network Tokens - First Step.xlsx', t)

        t += 1
        counter += 1

    PHASE2 = 0
    if PHASE2:
        print(f'Obtain the external medians...')
        otherG = G.copy()
        tSecond = 0
        threshold = {}

        goodValsOther, nodeAttr = [], {}
        valsOther = deepcopy(values)
        threshold = deepcopy(median[-1])

        valsOther = fc.update_step(otherG, listOfNodes, valsOther, threshold, goodValsOther,
                                   badValuesIdx, tSecond, redNodes)

        "Save the data"
        if ITERATIONS == 1:
            fc.save_data(valsOther, 'Network Tokens - Second Step.xlsx', 0)

        tSecond += 1

        community_potential = fc.get_comm_pot(goodValsOther, otherG)

        condition = []
        for c in range(nCommunities):
            condition += list(np.abs(community_potential[c]))
            nodeAttr.update(valsOther[c])

        nx.set_node_attributes(otherG, nodeAttr, 'sizeTokens')

        counter = 0
        while any(condition) > 0.001 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
            temp_b = deepcopy(valsOther),
            goodValsOther, nodeAttr = [], {}
            valsOther = fc.update_step(otherG, listOfNodes, valsOther, threshold,
                                       goodValsOther, badValuesIdx, tSecond, redNodes)

            community_potential = fc.get_comm_pot(goodValsOther, otherG)

            condition = []
            for c in range(nCommunities):
                condition += list(np.abs(community_potential[c]))
                nodeAttr.update(valsOther[c])
            nx.set_node_attributes(otherG, nodeAttr, 'sizeTokens')

            "Save the data"
            if ITERATIONS == 1:
                fc.save_data(valsOther, 'Network Tokens - Second Step.xlsx', tSecond)

            tSecond += 1
            counter += 1

        print('External medians obtained.')
        medianOther.append([round(fc.mMedian(list(valsOther[i].values())), 3) for i in range(nCommunities)])
        # times.append([t, tSecond])
    times.append([t])

# failureIntra = [sum([median[j][k] == mu_bad for j in range(ITERATIONS)]) / ITERATIONS for k in range(nCommunities)]
# failureXComm = [sum([medianOther[j][k] == mu_bad for j in range(ITERATIONS)]) / ITERATIONS for k in range(nCommunities)]

rate = np.array(rate)
times = np.array(times)
times = np.mean(times, axis=0)
times = [round(i, 3) for i in times]

fc.plot_growth(times, rate, path + 'growthRates', gSize[0])

print('Save failure rates')
with open(path + 'paramsAndMedians.txt', 'a') as f:
    # f.write(f'The failure rate 1st step: {failureIntra}\n')
    # f.write(f'The failure rate 2nd step: {failureXComm}\n')
    f.write(f'The average times: {times}')

print('Done.')
