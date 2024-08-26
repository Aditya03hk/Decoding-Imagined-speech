import os
import numpy as np
import pandas as pd

# Directory containing the data files
data_dir = 'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12_converted\\'

# File containing the class labels
class_file = 'C:\\Users\\adity\\Desktop\\DATA for mini\\tags\\12.txt'

# Load class labels
with open(class_file, 'r') as f:
    class_labels = f.read().splitlines()
print(class_labels)
# Assuming class labels are stored in the format: 'filename class'
class_labels = [line.split() for line in class_labels]
class_labels = {filename: int(label) for filename, label in class_labels}

print(class_labels)
# Initialize lists to hold data and labels
data_list = []
labels_list = []

# Iterate through each file in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith('.fif'):  # Assuming data files are in CSV format
        file_path = os.path.join(data_dir, filename)
        
        # Load the data (assuming CSV format, adjust if different)
        data = mne.io.read_raw_fif('0.fif', preload=True)
        
        # Append data and corresponding label to lists
        data_list.append(data.values)
        labels_list.append(class_labels[filename])

# Convert lists to numpy arrays for use with sklearn
X = np.array(data_list)
y = np.array(labels_list)

print(X)
print(y)