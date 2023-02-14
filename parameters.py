import math

path = './Outputs/'
BORDER_NODES_OPTION = 1
STOCHASTIC_BLOCK_MODEL = 1
n = 500
gSize = [n, 2 * n]
nCommunities = 2  # The number of communities
p = 5*math.log(n)/n
q = p/10
pQueryOracle = 0.3

thr = 3  # The number of times the oracle is queried

mu = [0, 30]  # Mean of the normal distribution
sigma = [1, 1]  # Standard deviation of the normal distribution
mu_bad = 7.0  # Mean of the bad nodes
sigma_bad = 0  # Standard deviation of the faulty nodes
