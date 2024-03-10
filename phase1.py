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
from matplotlib.ticker import MaxNLocator

# Fix the random seed
random.seed(10)
np.random.seed(10)

# Suppress future warnings, comment if code not running due to  this
warnings.simplefilter(action='ignore', category=FutureWarning)


def clean_fldr(folder_path):
    folder = folder_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def setup_graph():
    """
    Sets up the number of malicious nodes and the size of the communities based on that and the chosen option to test

    Returns:
        f (list[int]): list of elements where element i indicates the number of malicious nodes in community i
        n1, n2 (int): the size of each community, malicious nodes included
    """
    if DEALBREAKER == 2:
        random.seed(1)
        np.random.seed(1)
        f = [6, 1]
    else:
        random.seed(8)
        np.random.seed(8)
        f = [6, 3]

    n1 = 2 * f[0] + 2 + 1 + 1  # It respects d1 >= 2f + 3
    n2 = 2 * f[1] + 2 + 1 + 1 + 2 + 1  # It respects d2 >= 2f + 3 (2f + k + 1 + 1 since this is the requirement for the degree)

    if DEALBREAKER == 1:
        n1 = 2 * f[0] + 3
        n2 = 2 * f[1] + 5
    if not DEALBREAKER:
        random.seed(10)
        np.random.seed(10)
        f = [20, 10]
        n1 = 6 * f[0] + 2 + 1 + 1  # It respects d1 >= 2f + 3
        n2 = 3 * f[
            1] + 2 + 1 + 1 + 2  # It respects d2 >= 2f + 3 (2f + k + 1 + 1 since this is the requirement for the degree)

    print(f'Good nodes in community 1: {n1 - f[0]} \nGood nodes in community 2: {n2 - f[1]} '
          f'\nFaulty nodes in Community 1: {f[0]} \nFaulty nodes in Community 2: {f[1]}')

    return f, n1, n2


def generate_graph(n1, n2):
    """
    Generate the graph to be tested following the DEALBREAKER option

    Parameters:
        n1, n2 (int): the number of nodes in each community

    Returns:
        G (nxGraph): the global graph G
        plotTitle (str): the string for plot tiles and files name
    """
    G1 = nx.complete_graph(n1)
    G2 = nx.complete_graph(n2)
    if DEALBREAKER == 1:  # The min degree is not respected, but the graph is still robust
        plotTitle = 'noDegree'
    elif DEALBREAKER == 2:  # Remove many edges. The min degree is maintained but the graph is not robust anymore
        G2.remove_edge(0, 4)
        G2.remove_edge(0, 5)
        G2.remove_edge(0, 6)
        G2.remove_edge(0, 7)
        G2.remove_edge(1, 3)
        G2.remove_edge(1, 4)
        G2.remove_edge(1, 5)
        G2.remove_edge(1, 6)
        G2.remove_edge(2, 5)
        G2.remove_edge(2, 7)
        G2.remove_edge(2, 8)
        G2.remove_edge(3, 5)
        G2.remove_edge(3, 6)
        G2.remove_edge(3, 7)
        G2.remove_edge(4, 8)
        plotTitle = 'noRobustness'
    else:
        plotTitle = 'allGood'

    return nx.disjoint_union(G1, G2), plotTitle


def place_k_edges():
    """
    Add k edges per agent to connect the two communities
    """
    strikes = np.array([list_of_nodes, 2 * np.ones(nodes_g1 + nodes_g2)])
    try:
        # Establish the connections among two communities
        for i in nodes_comm[1]:
            external_nodes = random.sample(nodes_comm[0], k)
            for en in external_nodes:
                if strikes[1, en] > 0 and strikes[1, i] > 0:  # Add edge if there have not been added 2 edges already
                    edge_placed = np.random.binomial(1, p_edge, 1)[0]
                    if edge_placed:
                        G.add_edge(i, en)
                        strikes[1, i] = strikes[1, i] - 1
                        strikes[1, en] = strikes[1, en] - 1
                        # Check that no edge is selected twice
                        if strikes[1, i] < 0 or strikes[1, en] < 0:
                            raise Exception(f'More than k edges departing from {i} or possibly from {en}')
    except Exception as inst:
        print(inst.args)
        sys.exit()


def assign_values(means, stds, number_of_communities):
    """
    Assign the values to the legitimate nodes: values are samples uniformly at random from a normal distribution

    Parameters:
        means (list): list of c elements, each element i is the mean of the normal distribution
                   associated to agents in subset i
        stds (list): list of c elements, each element i is the standard deviation of the normal distribution
                      associated to agents in subset i
        number_of_communities (int): number of potential communities in the graph G

    Returns:
        valuesDf (DataFrame): dataframe of values in the c communities
    """
    tmp = []
    for j in range(number_of_communities):
        tmp = tmp + list(np.random.normal(means[j], stds[j], size=len(nodes_comm[j]) - number_faulty_nodes[j]))
    tmp = [round(i, 4) for i in tmp]
    print(f'The legitimate values to assign: {tmp}')
    for i in legitimate_nodes_indices:
        legitimate_nodes_values[i] = tmp.pop(0)

    vals = malicious_nodes_values | legitimate_nodes_values
    vals_df = pd.DataFrame(vals.items(), columns=['Node', 'Value']).explode('Value')
    return vals, vals_df


def update_step(vs, vsDf):
    """Performs the MCA update step

    Parameters:
        vs (list): list of values of all the nodes
        vsDf (DataFrame): dataframe where each new column is the list of updated values for all the nodes in G at step t

    Returns:
         vs (list): the updated list of values of all the nodes. Malicious nodes values are not updated
         vsDf (DataFrame): the values dataframe with the new values appended as a new column
    """
    tmp = copy.copy(vs)
    for i in legitimate_nodes_indices:
        neigh = list(G.adj[i])
        neigh_val = [vs[i] for i in neigh]
        neigh_val.sort()
        m = statistics.median(neigh_val)
        tmp[i] = alpha * vs[i] + (1 - alpha) * m
    vs = tmp
    tmpDf = pd.DataFrame(vs.items(), columns=['Nodes', 't > 0'])
    tmpDf = tmpDf.explode('t > 0')
    vsDf = pd.concat([vsDf, tmpDf['t > 0']], axis=1)
    return vs, vsDf


def display_graph(mG, vs, mString, the_path):
    print(f'Displaying the graph...')
    pos = nx.spring_layout(mG, seed=4)
    temp = {i: round(vs[i], 2) for i in vs.keys()}
    colors = list(vs.values())
    color_map = plt.get_cmap('GnBu')  # cm.GnBu
    color_map = [color_map(1. * i / (len(list(mG)))) for i in range(len(list(mG)))]
    color_map = matplotlib.colors.ListedColormap(color_map, name='faulty_nodes_cm')
    nx.draw(mG, pos=pos, node_color=colors, labels=temp, cmap=color_map, node_size=600)
    mStr = f'{mString}.png'
    plt.savefig(the_path + mStr, format='png')
    plt.close()


def plot_process(vs, mString, the_path):
    print(f'Plotting the results {mString}')
    fig, ax = plt.subplots(1, 1, layout='tight')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.ylim([-10, 350])
    BADVALFLAG = 1
    for i in list(G):
        nodeVal = vs.loc[vs['Node'] == i]
        nodeVal.drop(labels='Node', axis=1)
        nodeVal = nodeVal.values.tolist()[0]
        nodeVal.pop(0)
        if nodeVal[0] == badVal and BADVALFLAG:
            ax.plot(nodeVal, linewidth=0.7, color='red', linestyle='dashdot', label=r'Malicious Agents, $u \in F$')
            ax.legend(loc='best')
            BADVALFLAG = 0
        else:
            ax.plot(nodeVal, linewidth=0.2)

    ax.set_yscale('log')
    ax.set_xlabel(r'Time Steps $t$')
    ax.set_ylabel(r'$\xi_u(t)$')
    plt.savefig(the_path + mString + '.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    path = './Outputs/'

    DEALBREAKER = int(sys.argv[1])
    """Flag to indicate that something is wrong:
    # 0 -- All conditions are respected and the algorithm should lead to community consensus
    # 1 -- if the minimum degree is not respected but the graph is excess robust
    # 2 -- if the graph is not (0, f + 1)- excess robust but the minimum degree is respected
    # G is the merge of two complete graphs of size (n-1)/2, (n+1)/2 each
    """

    if os.path.exists(path):  # Clean output folder
        clean_fldr(path)
    else:
        os.makedirs(path)

    number_faulty_nodes, nodes_g1, nodes_g2 = setup_graph()

    k = 2
    badVal = 60  # The bad value of the faulty nodes
    c = 2  # Number of communities
    mu, sigma = [2, 30], [1, 5]  # The mean and std for the c communities
    time_constant = 5  # Set time bound for convergence
    print(f'Graphs size: {nodes_g1 + nodes_g2}')

    "Generate the graph"
    G, plot_title = generate_graph(nodes_g1, nodes_g2)

    list_of_nodes = list(G)
    nodes_comm, legitimate_nodes_indices = [], []
    nodes_comm.append([list_of_nodes[i] for i in range(nodes_g1)])
    nodes_comm.append([list_of_nodes[i + nodes_g1] for i in range(nodes_g2)])
    alpha = 0.6

    malicious_nodes_indices = random.sample(nodes_comm[0], number_faulty_nodes[0]) + random.sample(nodes_comm[1], number_faulty_nodes[1])
    print(f'Malicious Nodes Indices: {malicious_nodes_indices}')

    for j in range(c):
        legitimate_nodes_indices = legitimate_nodes_indices + [i for i in nodes_comm[j] if i not in malicious_nodes_indices]
    malicious_nodes_values = {i: badVal for i in malicious_nodes_indices}
    legitimate_nodes_values = {i: 0 for i in legitimate_nodes_indices}

    "Add the external edges"
    p_edge = 1
    place_k_edges()

    "Assign the values to the nodes in each community"
    values_comm = []
    values, valuesDf = assign_values(mu, sigma, c)
    for i in range(c):
        values_comm.append({j: values[j] for j in nodes_comm[i]})

    "Run MCA"
    t = 0
    display_graph(G, values, 'graph', path)
    while t < time_constant * math.log(len(list_of_nodes)):
        values, valuesDf = update_step(values, valuesDf)
        print(f'Step updated: {t}')
        t += 1
    print('Process completed.')

    "Plot the results"
    plot_process(valuesDf, plot_title, path)
    "Show the final graph"
    display_graph(G, values, 'graph_final', path)

    "Save nodes values to csv"
    # print(f'The final values: {values}')
    valuesDf.to_csv(path + 'values.csv', index=False)
