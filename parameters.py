import math
import sys

import numpy as np

path = './Outputs/'
BORDER_NODES_OPTION = 1
ITERATIONS = 1
n = 10000  # previously 1000
C = 9  # previously 7  # The suitable large constant for the probabilities of forming edges
gSize = [n, n]
nCommunities = 2  # The number of communities
p = np.zeros(nCommunities)
q = np.zeros(nCommunities)
for c in range(nCommunities):
    p[c] = 10 * C * math.log(gSize[c])/gSize[c]
    q[c] = 1/(C/0.6 * math.log(gSize[c])*gSize[c])  # previously 1.2
q = min(q)
pQueryOracle = float(sys.argv[1])/20

thr = 1  # The number of times the oracle is queried

mu = [0, 30]  # Mean of the normal distribution
sigma = [1, 1]  # Standard deviation of the normal distribution
mu_bad = 7.0  # Mean of the bad nodes
sigma_bad = 0  # Standard deviation of the faulty nodes
