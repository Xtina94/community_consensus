import math
import os
import warnings

import networkx as nx
import numpy as np
from copy import deepcopy

import functions_copy as fc
import parameters
from parameters import STOCHASTIC_BLOCK_MODEL, n, p, q, pQueryOracle, path, nCommunities

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
    probs = [[p, q], [q, p]]
    G = nx.stochastic_block_model([n, n], probs, seed=55)  # Generate graph
    mc_edges, mc, minDegree = fc.find_cut(G)  # Obtain the edges of the inter-blocks cut

    print(f'The partition cut edges: {mc_edges}'
          f'\nThe length of the cut: '
          f'{mc}'
          f'\nThe min degree: {minDegree}')
else:
    # G is the result of merging two regular graphs
    G, mc = fc.generate_graph(n, p, q)
    mc_edges = {}  # TODO: Cristina Complete this part

tf = math.ceil(2 * n * 0.05)  # top faulty nodes
fn = [math.floor(tf / 2), tf - math.floor(tf / 2)]  # The faulty nodes per community
redNodes = math.ceil(tf + int(0.8 * tf) * 1 / pQueryOracle)  # The amount of redundant nodes

print(f'n of faulty nodes: {tf}'
      f'\nn of redundant nodes: {redNodes}')

"Initial Step"
fn_indices = fc.choose_fn_idx(G, fn, mc_edges)
nu, values, G, goodValues = fc.assign_values(G, fn, fn_indices)

fc.display_graph(G, 0, 'Graph_iter', path)

"Calculate the initial median"
median = [fc.mMedian(list(values[i].values())) for i in range(nCommunities)]
l = {}
[l.update(values[i]) for i in range(nCommunities)]
medianTotal = fc.mMedian(list(l.values()))
medianOfMedian = fc.mMedian(median)

with open(path + 'paramsAndMedians.txt', 'w+') as f:
    f.write(f'n of faulty nodes: {tf}'
            f'\nN of nodes: {parameters.gSize}'
            f'\nn of redundant nodes: {redNodes}'
            f'\nThe initial median: {[round(m, 4) for m in median.copy()]}'
            f'\nThe median of medians: {round(medianOfMedian.copy(), 4)}\n')

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
    fc.display_graph(G, t, 'Graph_iter', path)

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
fc.save_data(values, 'Community Medians.csv')
with open(path + 'paramsAndMedians.txt', 'a') as f:
    f.write(f'The final medians: {[round(m, 4) for m in median]}\n')

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

community_potential = fc.get_comm_pot(goodValsOther, otherG)

condition = []
for c in range(nCommunities):
    condition += list(np.abs(community_potential[c]))
    nodeAttr.update(valsOther[c])

nx.set_node_attributes(otherG, nodeAttr, 'Values')
fc.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)

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
    fc.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)

    tSecond += 1
    counter += 1

fc.save_data(valsOther, 'Other Community Medians.csv')