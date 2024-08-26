import os
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from scipy.io import loadmat
import pandas as pd

# Directory containing the data files
data_dir = 'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12_converted'

# File containing the class labels
class_file = 'C:\\Users\\adity\\Desktop\\DATA for mini\\tags\\12.txt'
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

class_labels = result_list
data_list = []
labels_list = []

# Iterate through each file in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith('.fif'):  # Assuming data files are in CSV format
        file_path = os.path.join(data_dir, filename)
        
        # Load the data (assuming CSV format, adjust if different)
        data = mne.io.read_raw_fif('C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12_converted\\0.fif', preload=True)
        
        # Append data and corresponding label to lists
        data_list.append(data.values)
        labels_list.append(class_labels[filename])

# Convert lists to numpy arrays for use with sklearn
X = np.array(data_list)
y = np.array(labels_list)

print(X)
print(y)