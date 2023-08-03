import copy
import math
import os
import random
import shutil
import statistics
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import parameters
from parameters import gSize, nCommunities, \
    mu, mu_bad, sigma, sigma_bad, n, path, thr, pQueryOracle, alpha, OLD


# Assign the faulty values to the nodes on the border
def assign_labels(mG, faultyNodesIdx):
    S = [{} for _ in range(nCommunities)]
    Sgood = [{} for _ in range(nCommunities)]
    Sfaulty = [{} for _ in range(nCommunities)]
    gSizeReduced = [gSize[c] - len(faultyNodesIdx[c]) for c in range(nCommunities)]
    ids = [np.random.binomial(1, 0.5, (i, parameters.h)) for i in gSizeReduced]
    # ids = [np.random.multinomial(parameters.h, [1/4 for _ in range(4)], (i, 1)) for i in gSizeReduced]
    for i in range(nCommunities):
        if i:
            indices = list(mG)[gSize[i - 1]:gSize[i] + gSize[i - 1]]
        else:
            indices = list(mG)[0:gSize[i]]

        indices = [j for j in indices if j not in faultyNodesIdx[i]]  # The good nodes indices now

        Sgood[i].update({indices[g]: ids[i][g] for g in range(gSizeReduced[i])})
        Sfaulty[i].update({g: np.ones(parameters.h) for g in faultyNodesIdx[i]})
        S[i].update(Sgood[i])
        S[i].update(Sfaulty[i])

    attributes = {}
    [attributes.update(S[i]) for i in range(nCommunities)]
    nx.set_node_attributes(mG, attributes, 'IDs')
    return mG, S, Sgood


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


# Select the indices of the edges across the border
def choose_fn_idx(faultyNodes, borderNodes):  # borderNodes = [(e1, e2), (e3, e4), ...]
    faultyNodesIdx = [[], []]

    border = [[], []]
    for i in range(nCommunities):
        border[i] = set([l[i] for l in borderNodes])
        faultyNodesIdx[i] = random.sample(border[i], faultyNodes[i])

    return faultyNodesIdx


# Count the overall edges connecting to faulty nodes
def count_fn_edges(fn_index, mG):
    edg = [0 for _ in range(nCommunities)]
    for c in range(nCommunities):
        for f in fn_index[c]:
            k = len(mG.adj[f])
            edg[c] += k
    return edg


def count_nodes_edges(nn_index, fn_index, mG):
    edg = [0 for _ in range(nCommunities)]
    for c in range(nCommunities):
        for f in nn_index[c]:
            k = [adj for adj in mG.adj[f] if adj not in fn_index[c]]
            k = len(k)
            edg[c] += k
    return edg


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


# Display the graph
def display_graph(mG, rep, mString, path, attribute):
    pos = nx.spring_layout(mG, seed=4)
    attr = nx.get_node_attributes(mG, attribute)
    temp = {i: round(attr[i], 2) for i in attr.keys()}
    colors = list(attr.values())
    color_map = plt.get_cmap('GnBu')  # cm.GnBu
    color_map = [color_map(1. * i / (sum(gSize))) for i in range((sum(gSize)))]
    color_map = matplotlib.colors.ListedColormap(color_map, name='faulty_nodes_cm')
    nx.draw(mG, pos=pos, node_color=colors, labels=temp, cmap=color_map, node_size=600)
    mStr = mString + f'{rep}.png'
    plt.savefig(path + mStr, format='png')
    plt.close()


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


# Define majority median
def mMedian(v):
    v.sort()
    idx = math.ceil(len(v) / 2)
    if len(v) > 1:
        return v[idx]
    else:
        return v[0]


def oracle_help(nodeVals, lon):  # lon = listOfNodes (list with nodes IDs)
    myList = copy.copy(lon)
    edges = []
    edgeVal = []
    E = np.random.binomial(sum(gSize) * (sum(gSize) - 1), pQueryOracle)
    'Option 1 - without replacement in edges'
    probs = [1 / len(lon) for _ in range(len(lon))]
    nodeFrequency = np.random.multinomial(E, probs)  # vector of n elements: every elements has a values v, 0<=v<=E
                                                     # indicating how many edges contain that node as extreme
    startNodes = np.nonzero(nodeFrequency)  # vector of the nodes whose frequency has been sampled in the line above
    xEdges = []
    for i in startNodes[0]:  # for loop of E iterations, startNodes contain all the node indices of the nodes that have been selected
        tmp = random.sample(myList, nodeFrequency[
            i])  # Sample without replacement nodeFrequency[i] neighbors for i from the entire list
        edges += [(i, k) for k in tmp]
        if i < gSize[0]:  # TODO adapt to multi community
            # xEdges += len([e for e in tmp if e >= gSize[0]])
            xEdges += [(i, k) for k in tmp if k >= gSize[0]]
        else:  # TODO adapt to multi community
            # xEdges += len([e for e in tmp if e < gSize[0]])
            xEdges += [(i, k) for k in tmp if k < gSize[0]]
        for c in range(nCommunities):
            edgeVal += [(i, nodeVals[c][k]) for k in tmp if k in nodeVals[c].keys()]
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
    nXedges = len(xEdges)
    return edgeVal, nEdges, nXedges, xEdges


def oracle_help_old(node, neighs, oG, mRed, nodeVals, pQuery):
    redundantValues = {}
    if np.random.binomial(1, pQuery):
        redundancyOptions = [i for i in list(oG) if i not in neighs + [node]]
        redundantNodes = random.sample(redundancyOptions, mRed)
        for c in range(nCommunities):
            redundantValues.update({i: nodeVals[c][i] for i in redundantNodes if i in nodeVals[c].keys()})
    return redundantValues


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


def save_ids(vals, mStr, t):
    data = {'Comm 0': [r for r in vals[0].values()]}
    df = pd.DataFrame(data)
    for c in range(1, nCommunities):
        data = {f'Comm {c}': [r for r in vals[c].values()]}
        addition = pd.DataFrame(data)
        df = pd.concat([df, addition], axis=1)
    with pd.ExcelWriter(path + mStr, mode='a',
                        if_sheet_exists='overlay') as writer:  # NOTE: If the overlay option gives error, then Pandas needs upgrading to >= 1.4 version
        df.to_excel(writer, sheet_name=f't{t}', index=False)


def update_step(otherG, listOfNodes, x_b, threshold, x_b_goodValues, bvi, tScnd, sizeRates):
    temp_b = copy.deepcopy(x_b)
    if tScnd < thr:
        edges, nEdges, nXedges, xEdges = oracle_help(x_b, listOfNodes)
    else:
        edges, nEdges, nXedges, xEdges = [], 0, 0, []
    edgesDict = {x: [] for x in list(otherG)}
    xEdgesDict = {x: [] for x in list(otherG)}
    for e in edges:
        edgesDict[e[0]].append(e[1])
    for e in xEdges:
        xEdgesDict[e[0]].append(e[1])

    # avgSizeRate = []
    # [avgSizeRate.append([]) for _ in range(nCommunities)]
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

                    if otherCommunityVals:
                        med = mMedian(otherCommunityVals)
                    else:
                        med = x_b[c][x]
                    temp_b[c].update({x: med})
                    x_b_goodValues[c].update({x: med})

            if OLD:
                x_srVector = [[] for _ in range(nCommunities)]
                x_srVector_median = []
                RATESPRESENT = []
                for c in range(nCommunities):
                    if all([bool(sizeRates[c][n]) for n in sizeRates[c].keys()]):
                    # if all([bool(sizeRates[c])]):
                        RATESPRESENT += [1]
                        x_srVector[c] += [sizeRates[c][n][0] for n in neighbors]
                        x_srVector_median.append(statistics.median(x_srVector[c]))
                    else:
                        RATESPRESENT += [0]

            tmp = len(edgesDict[x])
            xTmp = len(xEdgesDict[x])
            # TODO: old option, 3 phases. OLD flag in the parameters.py file
            if OLD:
                if tmp:
                    for c in range(nCommunities):
                        if x in x_b[c].keys():
                            if RATESPRESENT[c] == 1:
                                # sr_local_mean = ((tmp-xTmp) / tmp + sizeRates[c][x][0]) / 2
                                sr_local_mean = sizeRates[c][x][0]
                                sizeRates[c][x] = [alpha * sr_local_mean + (1 - alpha) * x_srVector_median[c]]
                            else:
                                sr_local_mean = (tmp - xTmp) / tmp
                                sizeRates[c][x] = [sr_local_mean]
                        else:
                            if RATESPRESENT[c] == 1:
                                # sr_local_mean = (xTmp / tmp + sizeRates[c][x][0]) / 2
                                sr_local_mean = sizeRates[c][x][0]
                                sizeRates[c][x] = [alpha * sr_local_mean + (1 - alpha) * x_srVector_median[c]]
                                # sizeRates[c][x] += [round(xTmp/tmp, 4)]
                            else:
                                sr_local_mean = xTmp / tmp
                                sizeRates[c][x] = [sr_local_mean]
            else:
                if tmp:
                    for c in range(nCommunities):
                        if x in x_b[c].keys():
                            sizeRates[c][x] += [round((tmp - xTmp) / tmp, 4)]

                        else:
                            sizeRates[c][x] += [round(xTmp / tmp, 4)]
        else:
            sizeRates[0][x] += [mu_bad]  # bad nodes
            sizeRates[1][x] += [mu_bad]  # bad nodes
        # for c in range(nCommunities):
        #     avgSizeRate[c].append(sizeRates[x][c])
    # for c in range(nCommunities):
    #     tmp = [ar for ar in avgSizeRate[c] if ar not in [0, -1]]
    #     tmp = np.array(tmp)
    #
    #     # avgSizeRate[c] = np.array(avgSizeRate[c])
    #     avgSizeRate[c] = round(np.mean(tmp), 4)
    # print(f'The avg size rates: {avgSizeRate}')

    return temp_b, nEdges, nXedges, xEdges, sizeRates
