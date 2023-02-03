import math
import os

import networkx as nx
import warnings
import functions_copy as func
import numpy as np
import pandas as pd

from parameters import STOCHASTIC_BLOCK_MODEL, n, p, q, pQueryOracle, path, nCommunities

# Suppress future warnings, comment if code not running due to  this
warnings.simplefilter(action='ignore', category=FutureWarning)

"Clean the output folder"
if os.path.exists(path):
    func.clean_outputs_folder()
else:
    os.makedirs(path)

"Set up the graph"
if STOCHASTIC_BLOCK_MODEL:  # G is generated through a stochastic block model
    probs = [[p, q], [q, p]]
    G = nx.stochastic_block_model([n, n], probs, seed=45)  # Generate graph
    mc_edges, mc, minDegree = func.find_cut(G)  # Obtain the edges of the inter-blocks cut
    print(f'The partition cut edges: {mc_edges}\nThe length of the cut: '
          f'{mc}\nThe min degree: {minDegree}')
else:  # G is the result of merging two regular graphs
    G, mc = func.generate_graph(n, p, q)
    mc_edges = {}  # TODO: Cristina Complete this part

tf = math.ceil(2 * n * 0.05)  # top faulty nodes
fn = [math.floor(tf / 2), tf - math.floor(tf / 2)]  # The faulty nodes per community
redNodes = math.ceil(tf + int(0.8 * tf) * 1 / pQueryOracle)  # The amount of redundant nodes

print(f'n of faulty nodes: {tf}\nn of redundant nodes: {redNodes}')

"Initial Step"
fn_indices = func.choose_fn_indices(G, fn, mc_edges)
nu, values, G = func.assign_values(G, fn, fn_indices)

func.display_graph(G, 0, 'Graph_iter', path)

"Calculate the initial median"
median = [func.majority_median(list(values[i].values())) for i in range(nCommunities)]
l = {}
[l.update(values[i]) for i in range(nCommunities)]
medianTotal = func.majority_median(list(l.values()))
medianOfMedian = func.majority_median(median)

print(f'The initial median: {median}\nThe global median: {medianTotal}'
      f'\nThe median of medians: {medianOfMedian}')

"Calculate the potentials"
print('Calculate the potentials...')
community_potential = func.calculate_community_potential(values, G, fn_indices)
global_potential = func.calculate_global_potential(values, G, fn_indices)

condition_list = list(np.abs(community_potential[0]))
for c in range(1, nCommunities):
    condition_list += list(np.abs(community_potential[c]))

# while any(condition_list) > 0.01 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
#     temp, nodeAttributes = values.copy(), {}
#     for x in list(G):
#         neighVals = []
#         if x not in badValuesIdx:
#             neighbors = list(G.adj[x])
#             for c in range(nCommunities):
#                 neighVals += [values[c][j] for j in neighbors if j in values[c].keys()]
#             for c in range(nCommunities):
#                 if x in values[c].keys():
#                     med = func.majority_median(neighVals)
#                     temp[c].update({x: med})
#                     goodValues[c].update({x: med})
#     for c in range(nCommunities):
#         nodeAttributes.update(temp[c])
#     nx.set_node_attributes(G, nodeAttributes, 'Values')
#     func.display_graph(G, t, 'Graph_iter', path)
#
#     values = temp
#
#     # Update potentials
#     community_potential = func.calculate_community_potential(goodValues, G)
#     global_potential = func.calculate_global_potential(goodValues, G)
#
#     tmpp = []
#     for c in range(nCommunities):
#         tmpp += list(np.abs(community_potential[c]))
#     condition_list = tmpp
#
#     t += 1
#     counter += 1
#
print(f'########################'
      f'Obtain community median'
      f'########################')
community_potential, t, values = func.main_loop(condition_list, values, fn_indices, G)

func.save_data(values, 'Community Medians.csv')


print(f'###############################'
      f'Obtaining the external medians'
      f'###############################')

otherG = G.copy()
tSecond = 0
threshold = {}

# x_b, x_b_goodValues, nodeAttributes = deepcopy(values), [], {}
# temp_b = deepcopy(values)
# for c in range(nCommunities):
#     threshold[c] = func.majority_median(list(values[c].values()))
#
# x_b = func.update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, badValuesIdx, tSecond, redNodes)
#
# community_potential = func.calculate_community_potential(x_b_goodValues, otherG)
# global_potential = func.calculate_global_potential(x_b_goodValues, otherG)
#
# condition_list = []
# for c in range(nCommunities):
#     condition_list += list(np.abs(community_potential[c]))
#     threshold[c] = func.majority_median(list(values[c].values()))
#     nodeAttributes.update(x_b[c])
#
# nx.set_node_attributes(otherG, nodeAttributes, 'Values')
# func.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)
#
# counter = 0
# while any(condition_list) > 0.01 and counter < 3 * int(math.log(n)):  # and distanceChange > 0.001:
#     temp_b, x_b_goodValues, nodeAttributes = deepcopy(x_b), [], {}
#     x_b = func.update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, badValuesIdx, tSecond, redNodes)
#
#     community_potential = func.calculate_community_potential(x_b_goodValues, otherG)
#     global_potential = func.calculate_global_potential(x_b_goodValues, otherG)
#
#     condition_list = []
#     for c in range(nCommunities):
#         condition_list += list(np.abs(community_potential[c]))
#         nodeAttributes.update(x_b[c])
#     nx.set_node_attributes(otherG, nodeAttributes, 'Values')
#     func.display_graph(otherG, tSecond, 'OtherCommunity_iter', path)
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
