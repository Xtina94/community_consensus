import os
import random
import shutil

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import parameters
from parameters import gSize, nCommunities, \
    mu, mu_bad, sigma, sigma_bad


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


def choose_fn_indices(mG, faultyNodes, borderNodes):
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


def assign_values(mG, faultyNodes, faultyNodesIdx):
    mNu, gv, bv = [], [], []
    attributes = {}
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

        attributes.update(S1)

    nx.set_node_attributes(mG, attributes, 'Values')
    return mNu, attributes, mG


# Display graph
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