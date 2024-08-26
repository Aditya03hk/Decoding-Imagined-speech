import os
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
logging.getLogger('mne').setLevel(logging.ERROR)

# Directory containing the data files
data_dir = 'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12_converted\\'

# File containing the class labels
class_file = 'C:\\Users\\adity\\Desktop\\DATA for mini\\tags\\12.txt'

# Load class labels
with open(class_file, 'r') as f:
    class_labels = f.readlines()

# Initialize an empty dictionary for class labels
class_labels_dict = {}

# Iterate through each line in class_labels to fill the dictionary
for line in class_labels:
    parts = line.strip().split()
    if len(parts) == 3:
        filename, separator, label = parts
        class_labels_dict[filename] = int(label)
    else:
        print(f"Warning: skipping line due to incorrect format: {line}")
print(class_labels_dict)

# Initialize lists to hold data and labels
data_list = []
labels_list = []

# Iterate through each file in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith('.fif'):  # Assuming data files are in .fif format
        file_path = os.path.join(data_dir, filename)
        
        # Load the data using MNE
        raw = mne.io.read_raw_fif(file_path, preload=True)
        
        # Extract data (for example, using epochs or raw data)
        data, times = raw[:]
        
        # Check if filename exists in class_labels_dict
        if filename in class_labels_dict:
            # Append data and corresponding label to lists
            data_list.append(data.T)  # Transpose to have shape (samples, channels)
            labels_list.append(class_labels_dict[filename])
        else:
            print(f"Warning: no class label found for file {filename}")

# Convert lists to numpy arrays for use with sklearn
X = np.array(data_list)
y = np.array(labels_list)

print(X)
print(X.shape)
print(y)
print(y.shape)

# Example further steps: standardization, train-test split, and SVM training
scaler = StandardScaler()
X_reshaped = X.reshape(X.shape[0], -1)  # Reshape to (samples, features)
X_scaled = scaler.fit_transform(X_reshaped)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
