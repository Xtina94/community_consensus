import math
import os
import warnings

import networkx as nx
import numpy as np

import functions_copy as fc
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

print(f'The initial median: {median}\nThe global median: {medianTotal}'
      f'\nThe median of medians: {medianOfMedian}')

"Calculate the potentials"
print('Calculate the potentials...')
community_potential = fc.get_comm_pot(goodValues, G)
global_potential = fc.get_glob_pot(values, G, fn_indices)


print(f'######################## Obtain community median ########################')

condition_list = list(np.abs(community_potential[0]))
for c in range(1, nCommunities):
    condition_list += list(np.abs(community_potential[c]))

t, counter = 1, 0
badValuesIdx = fn_indices[0]
for c in range(nCommunities):
    badValuesIdx += fn_indices[c]
while any(condition_list) > 0.001 and counter < 30 * int(math.log(n)):  # and distanceChange > 0.001:
    temp, nodeAttributes = values.copy(), {}
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
        nodeAttributes.update(temp[c])
    nx.set_node_attributes(G, nodeAttributes, 'Values')
    fc.display_graph(G, t, 'Graph_iter', path)

    values = temp

    # Update potentials
    community_potential = fc.get_comm_pot(goodValues, G)

    condition_list = []
    for c in range(nCommunities):
        condition_list += list(np.abs(community_potential[c]))

    t += 1
    counter += 1

fc.save_data(values, 'Community Medians.csv')


print(f'############################### Obtaining the external medians ###############################')

otherG = G.copy()
tSecond = 0
threshold = {}

# x_b, x_b_goodValues, nodeAttributes = deepcopy(values), [], {}
# temp_b = deepcopy(values)
# for c in range(nCommunities):
#     threshold[c] = fc.mMedian(list(values[c].values()))
#
# x_b = fc.update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, badValuesIdx, tSecond, redNodes)
#
# community_potential = fc.calculate_community_potential(x_b_goodValues, otherG)
# global_potential = fc.calculate_global_potential(x_b_goodValues, otherG)
#
# condition_list = []
# for c in range(nCommunities):
#     condition_list += list(np.abs(community_potential[c]))
#     threshold[c] = fc.mMedian(list(values[c].values()))
#     nodeAttributes.update(x_b[c])
#
# nx.set_node_attributes(otherG, nodeAttributes, 'Values')
# fc.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)
#
# counter = 0
# while any(condition_list) > 0.01 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
#     temp_b, x_b_goodValues, nodeAttributes = deepcopy(x_b), [], {}
#     x_b = fc.update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, badValuesIdx, tSecond, redNodes)
#
#     community_potential = fc.calculate_community_potential(x_b_goodValues, otherG)
#     global_potential = fc.calculate_global_potential(x_b_goodValues, otherG)
#
#     condition_list = []
#     for c in range(nCommunities):
#         condition_list += list(np.abs(community_potential[c]))
#         nodeAttributes.update(x_b[c])
#     nx.set_node_attributes(otherG, nodeAttributes, 'Values')
#     fc.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)
#
#     tSecond += 1
#     counter += 1
#
# b = otherG.nodes.data(True)
# print(b)
# # data1 = [b[i][0] for i in range(len(b))]
# # data2 = [b[i][1]['Values'] for i in range(len(b))]
# # data = [data1, data2]
# data = [[b[i][0], b[i][1]['Values']] for i in range(len(b))]
# myDf = pandas.DataFrame(data, index=False)
# pandas.to_excel(myDf, "./DecOutputs/GraphAttrib.xlsx")
#
# # print(f'The global potential: {global_potential}')  # TODO: Cris do not delete this
