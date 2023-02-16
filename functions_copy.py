import copy
import math
import os
import random
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from itertools import islice
from collections import Counter

import parameters
from parameters import gSize, nCommunities, \
    mu, mu_bad, sigma, sigma_bad, n, path, thr, pQueryOracle


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
    print(f'The Average degree: {degree}')
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
        if e[0] < parameters.gSize[0]:
            if e[1] >= parameters.gSize[0]:
                partitionCut.append(e)
        else:
            break
    cutSize = len(partitionCut)
    border = [[e[0] for e in partitionCut], [e[1] for e in partitionCut]]
    faultyDegrees = [{i: mG.degree[i] for i in border[0]}, {i: mG.degree[i] for i in border[1]}]
    excess = [{i: faultyDegrees[0][i] - Counter(border[0])[i] for i in border[0]},
              {i: faultyDegrees[1][i] - Counter(border[1])[i] for i in border[1]}]
    rValues = list(Counter(border[0]).values()) + list(Counter(border[1]).values())
    rValues = Counter(rValues)  # keys are the degree, values are the number of nodes with that degree
    return partitionCut, cutSize, minDegree, rValues, excess


# Select the indices of the edges across the border
def choose_fn_idx(mG, faultyNodes, borderNodes):
    fn_copy = faultyNodes.copy()
    faultyNodesIdx = [[], []]
    previousNode = ['a']
    # TODO: rewrite it to account for more than two communities
    for i in range(nCommunities):
        for j in borderNodes:
            if j[i] != previousNode[-1]:
                faultyNodesIdx[i] += [j[i]]  # TODO: rewrite it to account for more than two communities
                fn_copy[i] -= 1
                if not fn_copy[i]:
                    break
                previousNode[-1] = j[i]
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


def save_data(vals, mStr, t):
    data = {'Comm 0': [round(r, 4) for r in vals[0].values()]}
    df = pd.DataFrame(data)
    for c in range(1, nCommunities):
        data = {f'Comm {c}': [round(r, 4) for r in vals[c].values()]}
        addition = pd.DataFrame(data)
        df = pd.concat([df, addition], axis=1)
    with pd.ExcelWriter(path + mStr, mode='a',
                        if_sheet_exists='overlay') as writer:  # NOTE: If the overlay option gives error, then Pandas needs upgrading to >= 1.4 version
        df.to_excel(writer, sheet_name=f't{t}', index=False)


def update_step(otherG, listOfNodes, x_b, threshold, x_b_goodValues, bvi, tScnd, red):
    temp_b = copy.deepcopy(x_b)
    edges = oracle_help(x_b, listOfNodes)  # TODO Latest changes
    # edgesDict = {e[0]: [] for e in edges}
    edgesDict = {x: [] for x in list(otherG)}
    for e in edges:
        edgesDict[e[0]].append(e[1])

    for x in list(otherG):
        neighVals = {}
        if x not in bvi:
            neighbors = list(otherG.adj[x])
            for c in range(nCommunities):
                neighVals.update({j: x_b[c][j] for j in neighbors if j in x_b[c].keys()})

            for c in range(nCommunities):
                # Key part in here
                if x in x_b[c].keys():
                    x_b_goodValues.append({})
                    otherCommunityVals = [k for k in neighVals.values() if
                                          round(k, 4) != round(threshold[c], 4)]
                    otherCommunityVals += [k for k in edgesDict[x] if
                                           round(k, 4) != round(threshold[c], 4)]

                    # edgeNeigh = [edges[i][1] for i in range(len(edges)) if edges[i][0] == x]
                    # otherCommunityVals += edgeNeigh

                    # if tScnd < thr:
                    # redundantVals = oracle_help(x, neighbors, otherG, red, x_b, pQueryOracle)
                    # if x in [edges[i][0] for i in range(len(edges))]:
                    # otherCommunityVals += [k for k in redundantVals.values() if
                    #                        k != threshold[c]]

                    if otherCommunityVals:
                        med = mMedian(otherCommunityVals)
                    else:
                        med = x_b[c][x]
                    temp_b[c].update({x: med})
                    x_b_goodValues[c].update({x: med})
    return temp_b


def oracle_help_old(node, neighs, oG, mRed, nodeVals, pQuery):
    redundantValues = {}
    if np.random.binomial(1, pQuery):
        redundancyOptions = [i for i in list(oG) if i not in neighs + [node]]
        redundantNodes = random.sample(redundancyOptions, mRed)
        for c in range(nCommunities):
            redundantValues.update({i: nodeVals[c][i] for i in redundantNodes if i in nodeVals[c].keys()})
    return redundantValues


def oracle_help(nodeVals, lon):
    myList = copy.copy(lon)
    edges = []
    edgeVal = []
    E = np.random.binomial(n * (n - 1), pQueryOracle)
    'Option 1 - without replacement in edges'
    startTime = time.time()
    probs = [1 / len(lon) for _ in range(len(lon))]
    nodeFrequency = np.random.multinomial(E,
                                          probs)  # vector of n elements: every elements has a values v, 0<= v <= E indicating how many edges contain that node as extreme
    startNodes = np.nonzero(nodeFrequency)  # vector of the nodes whose frequency has been sampled in the line above
    for i in startNodes[0]:  # for loop of E iterations
        tmp = random.sample(myList, nodeFrequency[
            i])  # Sample without replacement nodeFrequency[i] neighbors for i from the entire list
        edges += [(i, k) for k in tmp]
        for c in range(nCommunities):
            edgeVal += [(i, nodeVals[c][k]) for k in tmp if k in nodeVals[c].keys()]
    endTime = time.time()
    print(f'Total time: {endTime - startTime}')
    'Option 2 - With replacement in edges'
    # startTime = time.time()
    # for e in range(E):
    #     [u, v] = random.choices(myList, k=2)  # samples WITH replacement
    #     edges.append((u, v))
    #     # [u, v] = random.sample(myList, 2)  # samples WITHOUT replacement
    #
    #     # u = random.randint(0, 2*n)
    #     # v = random.randint(0, 2*n)
    #     for c in range(nCommunities):
    #         if v in nodeVals[c].keys():
    #             edgeVal.append((u, nodeVals[c][v]))
    # endTime = time.time()
    # print(f'Total time: {endTime - startTime}')
    edges = set(edges)
    nEdges = len(edges)
    print(f'We sampled {nEdges} unique edges')
    return edgeVal
