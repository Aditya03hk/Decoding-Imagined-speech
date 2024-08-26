import os
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from scipy.io import loadmat
g = 15
mat = loadmat("C:\\Users\\adity\\Desktop\\DATA for mini\\Training set\\Data_Sample15.mat")
dat = mat['epo_train']
data1 = dat['y'][0][0]
#print(data1)
data2 = [list(row) for row in zip(*data1)]
#print(data2)

result_list = []

for list in data2:
    if list == [1,0,0,0,0]:
        result_list.append(int('1'))
    elif list == [0,1,0,0,0]:
        result_list.append(int('2'))
    elif list == [0,0,1,0,0]:
        result_list.append(int('3'))
    elif list == [0,0,0,1,0]:
        result_list.append(int('4'))
    elif list == [0,0,0,0,1]:
        result_list.append(int('5'))
    else:
        continue

#print(result_list)

# filename = f"C:\\Users\\adity\\Desktop\\DATA for mini\\tags\\{g}.txt"

# # Open the file in write mode
# with open(filename, "w") as file:
#     # Write each element of the list on a new line with an index
#     for index, item in enumerate(result_list, start=0):
#         file.write(f"{index} : {item}\n")

# print(f"List has been written to {filename}")