import os
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import joblib


logging.getLogger('mne').setLevel(logging.ERROR)

all_data = []
all_labels = []

# Directory containing the data files
for i in range(1, 16):
    data_dir = f'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\{i}_converted\\'
    label_path = f'C:\\Users\\adity\\Desktop\\DATA for mini\\tags\\{i}.txt'
    
    data_list = []
    
    # Iterate through each file in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.fif'):  # Assuming data files are in .fif format
            file_path = os.path.join(data_dir, filename)
            
            # Load the data using MNE
            raw = mne.io.read_raw_fif(file_path, preload=True)
            
            # Extract data (for example, using epochs or raw data)
            # This example uses raw data, you might want to use epochs depending on your use case
            data, times = raw[:]
            
            # Append data to the data list
            data_list.append(data.T)
            raw.close()  # Close the raw object
            
    # Convert the data list to a numpy array
    X = np.array(data_list)
    print(X)
    # Append the data to the final list
    all_data.append(X)
    
    # Load the labels
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            _, value = line.split(':')
            labels.append(int(value.strip()))
    
    # Convert labels to numpy array and append to the final list
    y = np.array(labels)
    all_labels.extend(y)

# Convert lists to numpy arrays
final_data = np.concatenate(all_data)
final_labels = np.array(all_labels)

print(final_data.shape)
print(final_labels.shape)
# print(Y)
# print(Y.shape)

X = final_data
y = final_labels


# preprocessing data


# Flatten the data if necessary (e.g., if each file's data is multi-dimensional)
n_samples = X.shape[0]
X_flattened = X.reshape(n_samples, -1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flattened)


print('Aditya loves this model')

#train svm classifier


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm = SVC(kernel='linear')  # You can choose other kernels like 'rbf', 'poly', etc.
print('CLASSIFIER STARTED')
# Train the classifier
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print a detailed classification report
print(classification_report(y_test, y_pred))





#save the model

# Save the model
joblib.dump(svm, 'svm_model.joblib')

# Load the model
# svm = joblib.load('svm_model.joblib')
