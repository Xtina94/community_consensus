import math
import sys

import numpy as np

path = './Outputs/'
BORDER_NODES_OPTION = 1
ITERATIONS = 1
n = 1000  # previously 1000
C = 1  # previously 7  # The suitable large constant for the probabilities of forming edges
gSize = [n, n]
nCommunities = 2  # The number of communities
p = np.zeros(nCommunities)
q = np.zeros(nCommunities)
lower_tail_prob = 0.058  # Lower tail probability for the node degree

for c in range(nCommunities):
    p[c] = C * math.sqrt(n) * math.log(gSize[c])/gSize[c]
    '''By Chernoff, we can estimate the corresponding tail value for the minimum degree'''
    E_k = (n - 1) * p[c]  # The average degree within communities
    delta = math.sqrt(math.log(lower_tail_prob ** (-2 / E_k)))
    q[c] = C * ((1 - delta) * (n - math.sqrt(n)) * math.log(n)) / (n ** (5 / 2))
    # q[c] = p[c]/(math.sqrt(n) * math.log(n) * math.log(n))  # previously 1.2
q = min(q)
tf = int(0.04*n)  # mc  # top faulty nodes

pQueryOracle = 0.0005  # 0.01311  # p[0]*(tf+math.sqrt(tf))/(0.5*2*(2*n-1))  # 0.0005  # float(sys.argv[1])/n

thr = 10  # The number of times the oracle is queried

mu = [0, 30]  # Mean of the normal distribution
sigma = [1, 1]  # Standard deviation of the normal distribution
mu_bad = 7.0  # Mean of the bad nodes
sigma_bad = 0  # Standard deviation of the faulty nodes
