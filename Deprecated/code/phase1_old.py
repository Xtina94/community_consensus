import copy
import math
import os
import random
import shutil
import statistics
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


path = '../Outputs/'

# Suppress future warnings, comment if code not running due to  this
warnings.simplefilter(action='ignore', category=FutureWarning)


def clean_fldr():
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


"Clean the output folder"
if os.path.exists(path):
    clean_fldr()
else:
    os.makedirs(path)

"Set up the graph"
DEALBREAKER = 0  # Flag to indicate that something is wrong:
                    # 0 -- All conditions are respected and the algorithm should lead to community consensus
                    # 1 -- if the minimum degree is not respected but the graph is excess robust
                    # 2 -- if the graph is not (0, f + 1)- excess robust but the minimum degree is respected
# G is the merge of two complete graphs of size (n-1)/2, (n+1)/2 each
if DEALBREAKER == 2:
    f = [6, 1]
else:
    f = [6, 3]

n1 = 2 * f[0] + 2 + 1 + 1  # It respects d1 >= 2f + 3
n2 = 2 * f[1] + 2 + 1 + 1 + 2  # It respects d2 >= 2f + 3 (2f + k + 1 + 1 since this is the requirement for the degree)
print(f'Good nodes in community 1: {n1 - f[0]} \nGood nodes in community 2: {n2 - f[1]} '
      f'\nFaulty nodes in Community 1: {f[0]} \nFaulty nodes in Community 2: {f[1]}')
k = 2
b = 300  # The bad value of the faulty nodes
c = 2  # Number of communities
mu, sigma = [0, 60], [1, 1]
print(f'# Nodes: {n1 + n2}')

G1 = nx.complete_graph(n1)
G2 = nx.complete_graph(n2)
if DEALBREAKER == 1:  # remove n/3 matchings, the min degree is not respected anymore, but the graph is still robust
    tmp = list(G1)
    for i in range(int(len(list(G1))/10)):
        nds = random.sample(tmp, 2)
        G1.remove_edge(nds[0], nds[1])
        tmp.remove(nds[0])
        tmp.remove(nds[1])
elif DEALBREAKER == 2:  # remove many edges. The min degree is maintained but the graph is not robust anymore
    G2.remove_edge(0, 3)
    G2.remove_edge(0, 4)
    G2.remove_edge(0, 5)
    G2.remove_edge(0, 6)
    G2.remove_edge(1, 2)
    G2.remove_edge(1, 3)
    G2.remove_edge(1, 4)
    G2.remove_edge(1, 5)
    G2.remove_edge(2, 6)
    G2.remove_edge(2, 7)
    G2.remove_edge(3, 6)
    G2.remove_edge(3, 7)
    G2.remove_edge(4, 6)
    G2.remove_edge(4, 7)
    # tmp = list(G2)
    # start = random.sample(tmp, 1)[0]
    # tmp.remove(start)
    # nds = random.sample(tmp, 4)
    # for i in range(4):
    #     G2.remove_edge(start, nds[i])
G = nx.disjoint_union(G1, G2)

nodes = list(G)
nodes_comm, ln_indices = [], []
nodes_comm.append([nodes[i] for i in range(n1)])
nodes_comm.append([nodes[i + n1] for i in range(n2)])
p_edge = 1
tmp = 2 * np.ones(n1 + n2)
strikes = np.array([nodes, tmp])
alpha = 0.5

fn_indices = random.sample(nodes_comm[0], f[0]) + random.sample(nodes_comm[1], f[1])
print(f'fn indices: {fn_indices}')
for j in range(c):
    ln_indices = ln_indices + [i for i in nodes_comm[j] if i not in fn_indices]
fn_values = {i: b for i in fn_indices}
ln_values = {i: 0 for i in ln_indices}


# Add k edges per agent to connect the two communities
try:
    # Establish the connections among two communities
    for i in nodes_comm[1]:
        temp = random.sample(nodes_comm[0], k)
        for t in temp:
            if strikes[1, t] > 0 and strikes[1, i] > 0:  # Add edge only if there have not been added 2 edges to the node already
                edge_placed = np.random.binomial(1, p_edge, 1)[0]
                if edge_placed:
                    G.add_edge(i, t)
                    strikes[1, i] = strikes[1, i] - 1
                    strikes[1, t] = strikes[1, t] - 1
                    # Check that no edge is selected twice
                    if strikes[1, i] < 0 or strikes[1, t] < 0:
                        raise Exception(f'More than k edges departing from {i} or possibly from {t}')
except Exception as inst:
    print(inst.args)
    sys.exit()

# Assign the values to the legitimate nodes
tmp = []
for j in range(c):
    tmp = tmp + list(np.random.normal(mu[j], sigma[j], size=len(nodes_comm[j]) - f[j]))
tmp = [round(i, 4) for i in tmp]
print(f'The values to assign: {tmp}')
for i in ln_indices:
    ln_values[i] = tmp.pop(0)

values = fn_values | ln_values
valuesDf = pd.DataFrame(values.items(), columns=['Node', 'Value'])
valuesDf = valuesDf.explode('Value')
values_comm = []
for i in range(c):
    values_comm.append({j: values[j] for j in nodes_comm[i]})


def display_graph(mG, vs, mString):
    pos = nx.spring_layout(mG, seed=4)
    temp = {i: round(vs[i], 2) for i in vs.keys()}
    colors = list(vs.values())
    color_map = plt.get_cmap('GnBu')  # cm.GnBu
    color_map = [color_map(1. * i / (len(list(mG)))) for i in range(len(list(mG)))]
    color_map = matplotlib.colors.ListedColormap(color_map, name='faulty_nodes_cm')
    nx.draw(mG, pos=pos, node_color=colors, labels=temp, cmap=color_map, node_size=600)
    mStr = f'{mString}.png'
    plt.savefig(path + mStr, format='png')
    plt.close()


def update_step(vs, vsDf):
    tmp = copy.copy(vs)
    for i in ln_indices:
        neigh = list(G.adj[i])
        neigh_val = [vs[j] for j in neigh]
        neigh_val.sort()
        m = statistics.median(neigh_val)
        tmp[i] = alpha * vs[i] + (1 - alpha) * m
    vs = tmp
    tmpDf = pd.DataFrame(vs.items(), columns=['Nodes', 't > 0'])
    tmpDf = tmpDf.explode('t > 0')
    vsDf = pd.concat([vsDf, tmpDf['t > 0']], axis=1)
    return vs, vsDf

# Apply MCA
t = 0
print('Starting updates...')
display_graph(G, values, 'graph')
while t < 30 * math.log(len(nodes)):
    values, valuesDf = update_step(values, valuesDf)
    t += 1

print(f'The final values: {values}')
valuesDf.to_csv(path + 'values.csv', index=False)
