import copy
import math
import os
import warnings

import networkx as nx
import numpy as np
from copy import deepcopy

import functions_copy as fc
import parameters
from parameters import STOCHASTIC_BLOCK_MODEL, ITERATIONS, n, p, q, gSize, pQueryOracle, path, nCommunities, mu_bad

# Suppress future warnings, comment if code not running due to  this
warnings.simplefilter(action='ignore', category=FutureWarning)

"Clean the output folder"
if os.path.exists(path):
    fc.clean_fldr()
else:
    os.makedirs(path)

"Set up the graph"
if STOCHASTIC_BLOCK_MODEL:
    # G is generated through a stochastic block model
    probs = [[p[0], q], [q, p[1]]]  # TODO make it robust to multiple communities
    groundG = nx.stochastic_block_model(gSize, probs, seed=55)  # Generate graph
    mc_edges, mc, minDegree, rValues, excessInnerDegree = fc.find_cut(groundG)  # Obtain the edges of the inter-blocks cut

    print(f'p/q: {parameters.p}/{parameters.q}\n'
          f'The length of the cut: {mc}\n'
          f'The degrees on the cut: {rValues}\n'
          f'The excess inner degrees: {excessInnerDegree}\n'
          f'The min degree: {minDegree}')
else:
    # G is the result of merging two regular graphs
    groundG, mc = fc.generate_graph(n, p, q)
    mc_edges = {}  # TODO: Cristina Complete this part

tf = 2 * mc  # math.ceil(2 * n * 0.025)  # top faulty nodes
fn = [math.floor(tf / 2), tf - math.floor(tf / 2)]  # The faulty nodes per community
redNodes = math.ceil(1 * tf * 1/pQueryOracle)  # The amount of redundant nodes

print(f'n of faulty nodes: {tf}'
      f'\nn of redundant nodes: {redNodes}')

fc.display_graph(groundG, 0, 'Graph', path)
medianOther = []  # The list containing the final medians at every iteration
times = []

with open(path + 'paramsAndMedians.txt', 'w+') as f:
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
    nu, values, G, goodValues = fc.assign_values(G, fn, fn_indices)

    # fc.save_data(values, 'Initial Values.csv')

    # fc.display_graph(G, 0, 'Graph_iter', path)

    "Calculate the initial median"
    median = [fc.mMedian(list(values[i].values())) for i in range(nCommunities)]
    l = {}
    [l.update(values[i]) for i in range(nCommunities)]
    medianTotal = fc.mMedian(list(l.values()))
    medianOfMedian = fc.mMedian(median)

    # with open(path + 'paramsAndMedians.txt', 'a') as f:
    #     f.write(f'\nThe initial median: {[round(m, 4) for m in median.copy()]}'
    #             f'\nThe median of medians: {round(medianOfMedian.copy(), 4)}\n'
    #             f'\n*********************************************\n')

    "Calculate the potentials"
    print('Calculate the potentials...')
    community_potential = fc.get_comm_pot(goodValues, G)
    global_potential = fc.get_glob_pot(values, G, fn_indices)

    print(f'Obtain community median...')

    condition = list(np.abs(community_potential[0]))
    for c in range(1, nCommunities):
        condition += list(np.abs(community_potential[c]))

    t, counter = 1, 0
    badValuesIdx = fn_indices[0]
    for c in range(nCommunities):
        badValuesIdx += fn_indices[c]
    while any(condition) > 0.001 and counter < 30 * int(math.log(n)):  # and distanceChange > 0.001:
        temp, nodeAttr = values.copy(), {}
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
        # fc.display_graph(G, t, 'Graph_iter', path)

        values = temp

        # Update potentials
        community_potential = fc.get_comm_pot(goodValues, G)

        condition = []
        for c in range(nCommunities):
            condition += list(np.abs(community_potential[c]))

        t += 1
        counter += 1

    median = [fc.mMedian(list(values[i].values())) for i in range(nCommunities)]

    "Save Data to files"
    # fc.save_data(values, 'Intra Community Values.csv')
    # with open(path + 'paramsAndMedians.txt', 'a') as f:
    #     f.write(f'The final medians: {[round(m, 4) for m in median]}\n')
    #     f.write(f'The total time: {t - 1}\n')

    print('Data saved to file.')

    print(f'Obtain the external medians...')

    otherG = G.copy()
    tSecond = 0
    threshold = {}

    goodValsOther, nodeAttr = [], {}
    valsOther = deepcopy(values)
    threshold = deepcopy(median)

    valsOther = fc.update_step(otherG, valsOther, threshold, goodValsOther,
                               badValuesIdx, tSecond, redNodes)

    # fc.save_data(valsOther, f'Extra Community Values {tSecond}.csv')

    tSecond += 1

    community_potential = fc.get_comm_pot(goodValsOther, otherG)

    condition = []
    for c in range(nCommunities):
        condition += list(np.abs(community_potential[c]))
        nodeAttr.update(valsOther[c])

    nx.set_node_attributes(otherG, nodeAttr, 'Values')
    # fc.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)

    counter = 0
    while any(condition) > 0.001 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
        temp_b = deepcopy(valsOther),
        goodValsOther, nodeAttr = [], {}
        valsOther = fc.update_step(otherG, valsOther, threshold,
                                   goodValsOther, badValuesIdx, tSecond, redNodes)

        community_potential = fc.get_comm_pot(goodValsOther, otherG)

        condition = []
        for c in range(nCommunities):
            condition += list(np.abs(community_potential[c]))
            nodeAttr.update(valsOther[c])
        nx.set_node_attributes(otherG, nodeAttr, 'Values')
        # fc.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)
        # fc.save_data(valsOther, f'Extra Community Values {tSecond}.csv')

        tSecond += 1
        counter += 1

    medianOther.append([round(fc.mMedian(list(valsOther[i].values())), 3) for i in range(nCommunities)])
    times.append([t-1, tSecond-1])

    # fc.save_data(valsOther, f'Extra Community Values {tSecond}.csv')

    # with open(path + 'paramsAndMedians.txt', 'a') as f:
    #     f.write(f'The other-community total time: {tSecond - 1}\n')
    #
    # print('Data saved to file.')

badLuck = [sum([medianOther[j][0] == mu_bad for j in range(ITERATIONS)]), sum([medianOther[j][1] == mu_bad for j in range(ITERATIONS)])]
badLuck = [badLuck[0]/ITERATIONS, badLuck[1]/ITERATIONS]
times = np.array(times)
times = np.mean(times, axis=0)
times = [round(i, 3) for i in times]

with open(path + 'paramsAndMedians.txt', 'a') as f:
    f.write(f'The failure rate: {badLuck}\n')
    f.write(f'The average times: {times}')

with open(path + 'medians.txt', 'w+') as f:
    f.write(f'The medians: {medianOther}\n')

print('Data saved to file.')