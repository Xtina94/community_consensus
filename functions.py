import math
import os
import random
import shutil

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from parameters import BORDER_NODES_OPTION, n, gSize, nCommunities, \
    pQueryOracle, mu, mu_bad, thr, sigma, sigma_bad


# Clean the outputs folder or make a new one if it doesn't exist
def clean_outputs_folder(path):
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


# Define majority median
def majority_median(v):
    v.sort()
    idx = math.ceil(len(v) / 2)
    if len(v) > 1:
        return v[idx]
    else:
        return v[0]


# Display graph
def display_graph(mG, rep, mString, path):
    pos = nx.spring_layout(mG, seed=55)
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


# Assign values
def assign_values_old(mG, faultyNodes):
    mValues, mNu, goodVals, badValsIdx, vals, S, nodeAttributes = [], [], [], [], [], [], {}
    for i in range(nCommunities):
        if i:
            indices = list(mG)[gSize[i - 1]:gSize[i] + gSize[i - 1]]
        else:
            indices = list(mG)[0:gSize[i]]

        if BORDER_NODES_OPTION:
            for x in indices:
                for bd in list(mG.adj[x]):
                    if bd not in indices:
                        badValsIdx += [bd]

        bIndices = indices.copy()
        indices = {j: indices[j] for j in range(len(indices))}

        goodVals.append(np.random.normal(mu[i], sigma[i], size=gSize[i] - faultyNodes[i]))
        mNu.append(np.mean(goodVals[-1]))
        badVals = np.random.normal(mu_bad, sigma_bad, size=faultyNodes[i])

        if not BORDER_NODES_OPTION:  # bad nodes are allocated randomly
            vals.append(np.concatenate((goodVals[-1], badVals), axis=0))
            np.random.shuffle(vals[-1])

            badValsIdx += [indices[i] for i in indices.keys() if vals[-1][i] == mu_bad]
            # #badValsIdx += [i for i in indices if vals[-1][i] == mu_bad]
            # #tmp = {gv: badVals.pop(-1) for gv in badValsIdx}
            # #badVals = tmp
            goodValsIdx = {i: indices[i] for i in indices.keys() if vals[-1][i] != mu_bad}
            # #goodValsIdx = [i for i in indices if vals[-1][i] != mu_bad]
            goodVals[-1] = {goodValsIdx[gv]: vals[-1][gv] for gv in goodValsIdx.keys()}
            # #tmp = {gv: goodVals[-1].pop(-1) for gv in goodValsIdx}
            # #goodVals[-1] = tmp
            # #vals = goodVals[-1] | badVals[-1]  # Merge the two dictionaries

            # S = {j: vals[i][] for j in indices} # TODO: Cris, fix this mess
            S.append({indices[j]: vals[i][j] for j in indices.keys()})  ## i * gSize[i] + j
            nodeAttributes.update({indices[j]: vals[i][j] for j in indices.keys()})
            mValues.append(S[-1].copy())
        else:  # bad nodes stay on the border of the minimum cut
            # goodValsIdx = {i: indices[i] for i in indices.keys() if i not in badValsIdx}
            goodValsIdx = [k for k in bIndices if k not in badValsIdx]
            goodVals[-1] = {k: vals[-1][k] for k in goodValsIdx}
            # goodVals[-1] = {goodValsIdx[gv]: vals[-1][gv] for gv in goodValsIdx.keys()}

            # vals.append()

    nx.set_node_attributes(mG, nodeAttributes, 'Values')
    badValsNodes = [i for i in range(sum(gSize)) if nx.get_node_attributes(mG, 'Values')[i] == mu_bad]
    return mNu, mValues, goodVals, badValsNodes, badValsIdx


# Assign values
def assign_values(mG, faultyNodes):
    mValues, mNu, goodVals, badVals, vals = [], [], [], [], []
    badValsIdx, goodValsIdx, S, nodeAttributes = [[], []], [], [], {}
    gv, bv, bvidx = [], [], []
    if BORDER_NODES_OPTION:
        for edge in mG.edges():
            if edge[0] < n < edge[1]:
                badValsIdx[0] += [edge[0]]
                badValsIdx[1] += [edge[1]]
            if edge[1] < n < edge[0]:
                badValsIdx[1] += [edge[0]]
                badValsIdx[0] += [edge[1]]
    for i in range(nCommunities):
        if i:
            indices = list(mG)[gSize[i - 1]:gSize[i] + gSize[i - 1]]
        else:
            indices = list(mG)[0:gSize[i]]

        # if BORDER_NODES_OPTION:
        #     for edge in mG.edges():
        #         if edge[0] < n < edge[1] or edge[1] < n < edge[0]:
        #             badValsIdx += [edge[0], edge[1]]
        #     if faultyNodes[i]:
        #         f = faultyNodes[i]
        #         for x in indices:
        #             mAdj = list(mG.adj[x])
        #             bn = [k for k in mAdj if k not in indices]
        #             if bn:
        #                 badValsIdx += [x]
        #                 f -= 1
        #                 if not f:
        #                     break

        tmp = np.random.normal(mu[i], sigma[i], size=gSize[i] - faultyNodes[i])
        mNu.append(np.mean(tmp))
        gv.append(list(tmp))
        tmp = np.random.normal(mu_bad, sigma_bad, size=faultyNodes[i])
        bv.append(list(tmp))

        if not BORDER_NODES_OPTION:  # bad nodes are allocated randomly
            vals.append(gv[-1] + bv[-1])  # list containing all the nodes for the whole graph
            random.shuffle(vals[-1])

            tmp1 = {indices[k]: vals[-1][k] for k in range(len(vals[-1])) if vals[-1][k] == mu_bad}
            tmp2 = {indices[k]: vals[-1][k] for k in range(len(vals[-1])) if vals[-1][k] != mu_bad}
        else:  # bad nodes stay on the border of the minimum cut
            if badValsIdx[i]:
                tmp1 = {k: bv[-1].pop() for k in badValsIdx[i]}
            else:
                tmp1 = {}
            tmp2 = {k: gv[-1].pop() for k in indices if k not in badValsIdx[i]}
            # badValsIdx = []

        badVals.append(tmp1.copy())
        goodVals.append(tmp2.copy())
        bvidx += (list(tmp1.keys()))

        tmp1.update(tmp2)

        nodeAttributes.update(tmp1)
        mValues.append(tmp1.copy())

    nx.set_node_attributes(mG, nodeAttributes, 'Values')
    # badValsNodes = [i for i in range(sum(gSize)) if nx.get_node_attributes(mG, 'Values')[i] == mu_bad]
    return mNu, mValues, goodVals, bvidx


# Calculates the potential as u = 1/2 \sum_{x \in V}\sum_{y in N_x}(\xi_y - \xi_x) = - 1/2 \Nabla \phi
def calculate_community_potential(vals, mG):
    pot = {}
    for i in range(nCommunities):
        sG = mG.subgraph(list(vals[i].keys()))  # Obtain the community subgraph
        mL_comm = nx.laplacian_matrix(sG)
        mVals = np.array([j for j in vals[i].values()])
        tmp = (-1) * np.dot(mL_comm.toarray(), mVals)
        pot[i] = np.array([round(i, 4) for i in tmp])
    return pot


def calculate_global_potential(vals, mG):
    goodNodes = {}
    for c in range(nCommunities):
        goodNodes.update(vals[c])
    # sG = mG.subgraph([goodNodes[j] for j in goodNodes.keys() if j not in badV])  # Obtain the good nodes subgraph
    sG = mG.subgraph(goodNodes.keys())
    mL = nx.laplacian_matrix(sG)
    temp = []
    for i in range(nCommunities):
        temp += list(vals[i].values())
    mVals = np.array([i for i in temp])
    pot = (-1) * np.dot(mL.toarray(), mVals)
    pot = np.array([round(i, 4) for i in pot])
    return pot


def update_step(otherG, x_b, threshold, temp_b, x_b_goodValues, bvi, tScnd, red):
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
                                          k != threshold[c]]

                    if tScnd < thr:
                        redundantVals = oracle_help(x, neighbors, otherG, red, x_b, pQueryOracle)
                        otherCommunityVals += [k for k in redundantVals.values() if
                                               k != threshold[c]]

                    if otherCommunityVals:
                        med = majority_median(otherCommunityVals)
                    else:
                        med = x_b[c][x]
                    temp_b[c].update({x: med})
                    x_b_goodValues[c].update({x: med})
    return temp_b


# TODO: Cris, latest updates including the oracle in the computation
def oracle_help(node, neighs, oG, mRed, nodeVals, pQuery):
    redundantValues = {}
    if np.random.binomial(1, pQuery):
        redundancyOptions = [i for i in list(oG) if i not in neighs + [node]]
        redundantNodes = random.sample(redundancyOptions, mRed)
        for c in range(nCommunities):
            redundantValues.update({i: nodeVals[c][i] for i in redundantNodes if i in nodeVals[c].keys()})
    return redundantValues


def generate_graph(size, pd):
    mD = int(pd * size)
    print(f'The degree: {mD}')
    # minCut = random.randint(1, mD)
    minCut = int(mD/10)
    print(f'The min cut: {minCut}')
    discr = mD - minCut
    G1 = nx.random_regular_graph(mD, size)
    G2 = nx.random_regular_graph(mD, size)
    G2 = nx.relabel_nodes(G2, {i: i + size for i in list(G1)})
    cG = nx.compose(G1, G2)
    border1 = random.sample(list(G1), minCut)
    border2 = random.sample(list(G2), minCut)
    for i in range(minCut):
        cG.add_edge(border1[i], border2[i])
    return cG, minCut, discr