import math
import numpy as np

r = 99/100
a = 0.02
s = 30  # math.log((a-1+r)/a, r)
n1 = 2000
n2 = 2000

print(f's = {s}')

CASE = 'exponential1/2'  #'4exponential1/2'  # 'exponential1/4'  # exponential1/2 - exponential1/4
if CASE == 'linear':
    omega = [i for i in range(1, s + 1)]
    distribution = {omega[i-1]: (1/2)**i for i in range(1, s + 1)}
    trueMedian = 1
elif CASE == 'exponential1/2':
    omega = [2**i for i in range(1, s + 1)]
    distribution = {omega[i-1]: (1/2)**i for i in range(1, s + 1)}
    trueMedian = 2
elif CASE == '4exponential1/2':
    omega = [4**i for i in range(1, s + 1)]
    distribution = {omega[i-1]: (1/2)**i for i in range(1, s + 1)}
    trueMedian = 4
else:
    omega = [2**i for i in range(1, s + 1)]
    distribution = {omega[i-1]: (8/3)*(1/4)**i for i in range(1, s + 1)}
    trueMedian = 'N/A'  # round(math.sqrt(2), 4)
print(f'The distribution: {distribution}')

print(f'Sample space size: {s}')
print(f'Sample space: {omega}')

tmp = [i * distribution[i] for i in distribution.keys()]
trueMean = sum(tmp)
print(f'Omega True Median: {trueMedian}\nOmega True Mean: {trueMean}')
count1 = 0
count2 = 0
count3 = 0
iterations = 10000

for _ in range(iterations):
    reps = np.random.multinomial(n1, list(distribution.values()))  # reps has length s
    samples = []
    for i in range(1, s+1):
        samples += [omega[i-1] for _ in range(reps[i - 1])]
    # mMean1 = np.mean(samples)
    median1 = np.median(samples)

    reps = np.random.multinomial(n1, list(distribution.values()))  # reps has length s
    samples = []
    for i in range(1, s + 1):
        samples += [omega[i - 1] for _ in range(reps[i - 1])]
    # mMean2 = np.mean(samples)
    median2 = np.median(samples)

    count1 += int(median2 > median1)
    count2 += int(median2 < median1)
    count3 += int(median2 == median1)
print(f'The fraction of times when M(2n) > M(n): {count1/iterations}')
print(f'The fraction of times when M(2n) < M(n): {count2/iterations}')
print(f'The fraction of times when M(2n) = M(n): {count3/iterations}')

DETAILS = 0
if DETAILS:
    print('***************************')
    print(f'********CASE: {CASE}********')

    print('------------------------------')
    print(f'-------N Samples = {n1}-------')

    reps = np.random.multinomial(n1, list(distribution.values()))  # reps has length s
    print(f'The reps: {reps}')
    samples = []
    for i in range(1, s+1):
        samples += [omega[i-1] for _ in range(reps[i - 1])]
    mMean = np.mean(samples)
    median = np.median(samples)
    print(f'The median: {median}')
    print(f'The mean: {mMean}')
    print(f'\mu - m = {mMean - median}')

    print('------------------------------')
    print(f'-------N Samples = {n2}-------')

    reps = np.random.multinomial(n2, list(distribution.values()))  # reps has length s
    print(f'The reps: {reps}')
    samples = []
    for i in range(1, s+1):
        samples += [omega[i-1] for _ in range(reps[i - 1])]
    mMean = np.mean(samples)
    median = np.median(samples)
    print(f'The median: {median}')
    print(f'The mean: {mMean}')
    print(f'\mu - m = {mMean - median}')