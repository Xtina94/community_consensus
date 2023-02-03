import copy
import math
import os
import random
import shutil

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from itertools import islice

import parameters
from parameters import gSize, nCommunities, \
    mu, mu_bad, sigma, sigma_bad, n, path


# Clean the outputs folder or make a new one if it doesn't exist
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


# Generate a graph stochastic block model - like
def generate_graph(commSize, pd, qd):
    degree = int(pd * commSize)
    print(f'The degree: {degree}')
    minCut = int(qd * commSize)
    print(f'The min cut: {minCut}')
    G1 = nx.random_regular_graph(degree, commSize)
    G2 = nx.random_regular_graph(degree, commSize)
    G2 = nx.relabel_nodes(G2, {i: i + commSize for i in list(G1)})
    cG = nx.compose(G1, G2)
    border1 = random.sample(list(G1), minCut)
    border2 = random.sample(list(G2), minCut)
    for i in range(minCut):
        cG.add_edge(border1[i], border2[i])
    return cG, minCut


# Find the cut between the two blocks and the minimum degree TODO Update to c communities
def find_cut(mG):
    minDegree = min([i[1] for i in mG.degree])
    partitionCut = []
    for e in mG.edges:
        if e[0] < parameters.n:
            if e[1] >= parameters.n:
                partitionCut.append(e)
        else:
            break
    cutSize = len(partitionCut)
    return partitionCut, cutSize, minDegree


# Select the indices of the edges across the border
def choose_fn_idx(mG, faultyNodes, borderNodes):
    fn_copy = faultyNodes.copy()
    faultyNodesIdx = [[], []]  # TODO: rewrite it to account for more than two communities
    for i in range(nCommunities):
        for j in borderNodes:
            faultyNodesIdx[i] += [j[i]]  # TODO: rewrite it to account for more than two communities
            fn_copy[i] -= 1
            if not fn_copy[i]:
                break
        if fn_copy[i]:
            choices = [j for j in list(mG) if j not in faultyNodesIdx[i]]
            faultyNodesIdx[i] += random.sample(choices, fn_copy[i])

    return faultyNodesIdx


# Assign the faulty values to the nodes on the border
def assign_values(mG, faultyNodes, faultyNodesIdx):
    mNu, gv, bv = [], [], []
    S, Sgood = [], []
    for i in range(nCommunities):
        if i:
            indices = list(mG)[gSize[i - 1]:gSize[i] + gSize[i - 1]]
        else:
            indices = list(mG)[0:gSize[i]]

        indices = [j for j in indices if j not in faultyNodesIdx[i]]  # The good nodes indices now
        tmp = np.random.normal(mu[i], sigma[i], size=gSize[i] - faultyNodes[i])
        mNu.append(np.mean(tmp))
        gv.append(list(tmp))
        tmp = np.random.normal(mu_bad, sigma_bad, size=faultyNodes[i])
        bv.append(list(tmp))
        S1 = {faultyNodesIdx[i][f]: bv[-1][f] for f in range(len(bv[-1]))}
        S2 = {indices[g]: gv[-1][g] for g in range(len(gv[-1]))}
        S1.update(S2)
        Sgood.append(S2)

        S.append(S1)

    attributes = {}
    [attributes.update(S[i]) for i in range(nCommunities)]
    nx.set_node_attributes(mG, attributes, 'Values')
    return mNu, S, mG, Sgood


# Display the graph
def display_graph(mG, rep, mString, path):
    pos = nx.spring_layout(mG, seed=4)
    attr = nx.get_node_attributes(mG, 'Values')
    temp = {i: round(attr[i], 2) for i in attr.keys()}
    colors = list(attr.values())
    color_map = plt.get_cmap('GnBu')  # cm.GnBu
    color_map = [color_map(1. * i / (sum(gSize))) for i in range((sum(gSize)))]
    color_map = matplotlib.colors.ListedColormap(color_map, name='faulty_nodes_cm')
    nx.draw(mG, pos=pos, node_color=colors, labels=temp, cmap=color_map, node_size=600)
    mStr = mString + f'{rep}.png'
    plt.savefig(path + mStr, format='png')
    plt.close()


# Define majority median
def mMedian(v):
    v.sort()
    idx = math.ceil(len(v) / 2)
    if len(v) > 1:
        return v[idx]
    else:
        return v[0]


# Calculates the potential as u = 1/2 \sum_{x \in V}\sum_{y in N_x}(\xi_y - \xi_x) = - 1/2 \Nabla \phi
def get_comm_pot(vals, mG):
    pot = {}
    for i in range(nCommunities):
        sG = mG.subgraph(list(vals[i].keys()))  # Obtain the community subgraph
        mL_comm = nx.laplacian_matrix(sG)
        mVals = np.array([j for j in vals[i].values()])
        tmp = (-1) * np.dot(mL_comm.toarray(), mVals)
        pot[i] = np.array([round(i, 4) for i in tmp])
    return pot


def get_glob_pot(vals, mG, fnIdx):
    goodNodes = {}
    for c in range(nCommunities):
        goodNodes.update({j: vals[c][j] for j in vals[c].keys() if j not in fnIdx[c]})
    sG = mG.subgraph(goodNodes.keys())
    mL = nx.laplacian_matrix(sG)
    goodVals = np.array(list(goodNodes.values()))
    pot = (-1) * np.dot(mL.toarray(), goodVals)
    pot = np.array([round(i, 4) for i in pot])
    return pot


def main_loop_old(mG, values, goodValues, fn_indices, condition_list):
    t, counter = 1, 0
    badValuesIdx = fn_indices[0]
    for c in range(nCommunities):
        badValuesIdx += fn_indices[c]
    while any(condition_list) > 0.001 and counter < 30 * int(math.log(n)):  # and distanceChange > 0.001:
        temp, nodeAttributes = values.copy(), {}
        for x in list(mG):
            neighVals = []
            if x not in badValuesIdx:
                neighbors = list(mG.adj[x])
                for c in range(nCommunities):
                    neighVals += [values[c][j] for j in neighbors if j in values[c].keys()]
                for c in range(nCommunities):
                    if x in values[c].keys():
                        med = majority_median(neighVals)
                        temp[c].update({x: med})
                        goodValues[c].update({x: med})
        for c in range(nCommunities):
            nodeAttributes.update(temp[c])
        nx.set_node_attributes(mG, nodeAttributes, 'Values')
        display_graph(mG, t, 'Graph_iter', path)

        values = temp

        # Update potentials
        community_potential = calculate_community_potential(goodValues, mG)

        condition_list = []
        for c in range(nCommunities):
            condition_list += list(np.abs(community_potential[c]))

        t += 1
        counter += 1

        return community_potential, t, values


def save_data(vals, mStr):
    data = {'Comm 0': [round(r, 4) for r in vals[0].values()]}
    for c in range(1, nCommunities):
        data.update({f'Comm {c}': [round(r, 4) for r in vals[c].values()]})
    df = pd.DataFrame(data)
    # for c in range(1, nCommunities):
    #     mStr = f'Comm {c}'
    #     df[mStr] = list(vals[c].values())
    df.to_csv(path + mStr, index=False)

