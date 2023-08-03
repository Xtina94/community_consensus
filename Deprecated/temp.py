# Create a file where to store all the results

import os
import pathlib
import re

import pandas as pd

# Enter path in unix style - this is your top level folder which you created in Step 1

main_path = "C:/Users/Cristina/Documents/TA/6CCS3AIN_2022/TopLevelDirectory"  # Change for next year folder
cw_path = main_path + "/Coursework-regularSubmissions"  # Uncomment when this folder is used
cl_path = main_path + "/CommandLine"
markingSheetName = '/6CCS3AIN 22_23 SEM1 000001 ARTIFICIAL INTE Grades.xlsx'  # Used when saving results.
# Change this string when having a new file

txtPath = main_path + '/FixedRandom/FixedRandom/CheckTxt/'

result = {'Seed': [], 'SG_wins': [], 'SG_points': [], 'MC_wins': [], 'MC_points': [], 'Excellence': [],
          'Excellence_points': [], 'Total': []}
student = []
print(f'The results: {result}')

myPadding = 9  # Currently, the output txt files have 8 entries. Together with the student's name column it makes 9
# elements.
# Change this if you save to txt more results

resultsList = []
resultDf = pd.DataFrame(columns=list(result.keys()))
resultDf.to_excel(txtPath + 'OtherSeedsResults.xlsx', index=False)

for path, subdirs, files in os.walk(txtPath):
    for name in files:
        lnk = pathlib.PurePath(path, name)
        file_path = txtPath + name
        seedList = []
        try:
            with open(file_path) as f:
                lines = f.readlines()
            lines = [i.partition('\n')[0] for i in lines]  # Extract the values and put them in a list
            for l in range(len(lines)):
                if 'Seed' in lines[l]:
                    tmp = re.findall(r'\d+', lines[l])
                    if tmp[0] not in seedList:
                        result['Seed'].append(int(tmp[0]))
                        r = int(lines[l + 1])
                        print(f"The item: {r}")
                        result['SG_wins'].append(r)
                        if r < 5:
                            result['SG_points'].append(7)
                        elif 5 <= r < 10:
                            result['SG_points'].append(15)
                        elif 10 <= r < 15:
                            result['SG_points'].append(25)
                        elif 15 <= r < 20:
                            result['SG_points'].append(30)
                        elif r >= 20:
                            result['SG_points'].append(38)
                        else:
                            print('Something is wrong')
                        print(f"The result: {result}")
                        seedList.append(tmp[0])
                    else:
                        r = int(lines[l + 1])
                        result['MC_wins'].append(r)
                        if r < 3:
                            result['MC_points'].append(6)
                        elif 3 <= r < 5:
                            result['MC_points'].append(15)
                        elif 5 <= r < 7:
                            result['MC_points'].append(23)
                        elif 7 <= r < 10:
                            result['MC_points'].append(30)
                        elif r >= 10:
                            result['MC_points'].append(37)
                        else:
                            print('Something is wrong')
                        r = float(lines[l + 4]) + 50
                        result['Excellence'].append(r)
                        if 800 <= r < 1600:
                            result['Excellence_points'].append(5)
                        elif 1600 <= r < 2400:
                            result['Excellence_points'].append(9)
                        elif 2400 <= r < 3200:
                            result['Excellence_points'].append(14)
                        elif 3200 <= r < 4000:
                            result['Excellence_points'].append(18)
                        elif r >= 4000:
                            result['Excellence_points'].append(25)
                        else:
                            result['Excellence_points'].append(0)
                        result['Total'].append(result['SG_points'][-1] + result['MC_points'][-1] + result['Excellence_points'][-1])
            name = name.partition("_results")[0]  # Obtain the name of the student
            df = pd.DataFrame(result)
            print(f"The student: {name}")
            with pd.ExcelWriter(txtPath + 'OtherSeedsResults.xlsx', mode='a',
                                if_sheet_exists='replace') as writer:  # NOTE: If the overlay option gives error, then Pandas needs upgrading to >= 1.4 version
                # df = pd.DataFrame(result)
                print(f'The results DF: {df}')
                df.to_excel(writer, sheet_name=name, index=False)
                result = {'Seed': [], 'SG_wins': [], 'SG_points': [], 'MC_wins': [], 'MC_points': [], 'Excellence': [],
                          'Excellence_points': [], 'Total': []}
        except:
            continue
