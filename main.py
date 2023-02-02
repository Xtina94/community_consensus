import math
import os
import random
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas

import functions as func
from parameters import STOCHASTIC_BLOCK_MODEL, n, nCommunities, p, q, pQueryOracle, mu, path

# BORDER_NODES_OPTION = 1
# STOCHASTIC_BLOCK_MODEL = 0
# n = 500
# gSize = [n, n]
# nCommunities = 2
# p, q = 0.05, 0.01
# pQueryOracle = 0.3

# Clean the output folder
if os.path.exists(path):
    func.clean_outputs_folder(path)
else:
    os.makedirs(path)

if STOCHASTIC_BLOCK_MODEL:  # G is generated through a stochastic block model
    probs = [[p, q], [q, p]]
    G = nx.stochastic_block_model([n, n], probs, seed=45)  # Generate graph

    # Define the faulty nodes
    mc = nx.minimum_edge_cut(G)
    print(f'The minimum cut: {mc}')
    mc = len(mc)
else:  # G is the result of merging two regular graphs
    G, mc, discrepancy = func.generate_graph(n, p)

tf = 2 * mc  # top faulty nodes, previously int(len(mc)/2)
tf = 1
print(f'The n of total faulty nodes: {tf}')
p_fn = random.randint(0, tf - 1)
fn = [int(tf / 2), int(tf / 2)]

if STOCHASTIC_BLOCK_MODEL:
    redNodes = tf + int(0.8*tf) * 1/pQueryOracle  # The redundancy nodes
else:
    tmp = math.ceil(2 * discrepancy / (mc + discrepancy) * tf)
    print(f'The addition: {tmp}\n and the factor {discrepancy / (mc + discrepancy)}')
    redNodes = tf + tmp  # The redundancy nodes

goodColor = '#ffceac'
badColor = '#ff6347'

# Initial step
nu, values, goodValues, badValuesIdx = func.assign_values(G, fn)
func.display_graph(G, 0, 'Graph_iter',  path)

# About the median
median, temp = [], []
for i in range(len(mu)):
    median.append(func.majority_median(list(values[i].values())))
    temp += list(values[i].values())
median_total = func.majority_median(temp)
medianOfMedian = func.majority_median(median)
print(f'The initial median: {median}')
print(f'The global median: {median_total}')
print(f'The median of medians: {medianOfMedian}')

community_potential = func.calculate_community_potential(goodValues, G)
global_potential = func.calculate_global_potential(goodValues, G)

t = 1
condition_list = []
for c in range(nCommunities):
    condition_list += list(np.abs(community_potential[c]))

counter = 0
while any(condition_list) > 0.01 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
    temp, nodeAttributes = values.copy(), {}
    for x in list(G):
        neighVals = []
        if x not in badValuesIdx:
            neighbors = list(G.adj[x])
            for c in range(nCommunities):
                neighVals += [values[c][j] for j in neighbors if j in values[c].keys()]
            for c in range(nCommunities):
                if x in values[c].keys():
                    med = func.majority_median(neighVals)
                    temp[c].update({x: med})
                    goodValues[c].update({x: med})
    for c in range(nCommunities):
        nodeAttributes.update(temp[c])
    nx.set_node_attributes(G, nodeAttributes, 'Values')
    func.display_graph(G, t, 'Graph_iter', path)

    values = temp

    # Update potentials
    community_potential = func.calculate_community_potential(goodValues, G)
    global_potential = func.calculate_global_potential(goodValues, G)

    tmpp = []
    for c in range(nCommunities):
        tmpp += list(np.abs(community_potential[c]))
    condition_list = tmpp

    t += 1
    counter += 1

print(f'The total time: {t}')


print(f'####################################'
      f'Obtaining the other community median'
      f'####################################')

otherG = G.copy()
tSecond = 0
threshold = {}

x_b, x_b_goodValues, nodeAttributes = deepcopy(values), [], {}
temp_b = deepcopy(values)
for c in range(nCommunities):
    threshold[c] = func.majority_median(list(values[c].values()))

x_b = func.update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, badValuesIdx, tSecond, redNodes)

community_potential = func.calculate_community_potential(x_b_goodValues, otherG)
global_potential = func.calculate_global_potential(x_b_goodValues, otherG)

condition_list = []
for c in range(nCommunities):
    condition_list += list(np.abs(community_potential[c]))
    threshold[c] = func.majority_median(list(values[c].values()))
    nodeAttributes.update(x_b[c])

nx.set_node_attributes(otherG, nodeAttributes, 'Values')
func.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)

counter = 0
while any(condition_list) > 0.01 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
    temp_b, x_b_goodValues, nodeAttributes = deepcopy(x_b), [], {}
    x_b = func.update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, badValuesIdx, tSecond, redNodes)

    community_potential = func.calculate_community_potential(x_b_goodValues, otherG)
    global_potential = func.calculate_global_potential(x_b_goodValues, otherG)

    condition_list = []
    for c in range(nCommunities):
        condition_list += list(np.abs(community_potential[c]))
        nodeAttributes.update(x_b[c])
    nx.set_node_attributes(otherG, nodeAttributes, 'Values')
    func.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)

    tSecond += 1
    counter += 1

b = otherG.nodes.data(True)
print(b)
# data1 = [b[i][0] for i in range(len(b))]
# data2 = [b[i][1]['Values'] for i in range(len(b))]
# data = [data1, data2]
data = [[b[i][0], b[i][1]['Values']] for i in range(len(b))]
myDf = pandas.DataFrame(data, index=False)
pandas.to_excel(myDf, "./DecOutputs/GraphAttrib.xlsx")

# print(f'The global potential: {global_potential}')  # TODO: Cris do not delete this
