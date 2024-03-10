"""File containing all the data structures needed to save the relevant data and the files created and saved
accordingly """
import pandas as pd

from parameters import path

"""The data structures"""

initialMedian = []  # The vector of initial medians
median = []  # The vector of final medians in first step
medianRates = []
medianOther = []  # The vector of final other median in second step
times = []  # The vector of times for first and second step
firstStepValues = []  # The values in the network over time for first step
secondStepValues = []  # The values in the network over time for second step

"""The files to save elements to"""


def create_files():
    with open(path + 'paramsAndMedians.txt', 'w+') as f:
        f.write(f'File containing the simulation parameters, the failure rates within each community,\n'
                f'the convergence times in the first and second step.\n\n'
                f'-----------------------------------------------------------------------------------\n\n')

    valuesOverTimeDf = pd.DataFrame(columns=['Comm 0', 'Comm 1'])
    valuesOverTimeDf.to_excel(path + 'Network Values - First Step.xlsx', sheet_name='t0',  index=False)
    valuesOverTimeDf.to_excel(path + 'Network Ids.xlsx', sheet_name='t0',  index=False)
    valuesOverTimeDf.to_excel(path + 'Network Values - Second Step.xlsx', sheet_name='t0',  index=False)
    valuesOverTimeDf.to_excel(path + 'Network Tokens - First Step.xlsx', sheet_name='t0',  index=False)
    valuesOverTimeDf.to_excel(path + 'Community size rates.xlsx', sheet_name='t0', index=False)