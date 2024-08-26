import os
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from scipy.io import loadmat

mat = loadmat("C:\\Users\\adity\\Desktop\\DATA for mini\\Training set\\Data_Sample12.mat")
i =1
# Extract the relevant data
# Assuming the data is stored under the key 'epo_train' and further nested under 'x'
dat = mat['epo_train']
data1 = dat['x'][0, 0]
data = np.transpose(data1, (2, 1, 0))
data = data[i, [11,45,13,15,47,12,51,19,18,24], :]  # Select the first epoch (or use np.mean(data, axis=0) for average)

sfreq = 256
ch_names = ['FC6', 'FT8', 'C3', 'C4', 'C5', 'T7', 'CP3', 'CP1', 'CP5', 'P3']

# Create the MNE info structure
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Create the RawArray with the correctly shaped data
raw = mne.io.RawArray(data, info)

# Save the Raw object to a FIF file
raw.save(f'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12_converted\\{i}.fif', overwrite=True)
print(f'done {i}')

# Load the saved FIF file
raw = mne.io.read_raw_fif(f'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12_converted\\{i}.fif', preload=True)

# Plot the raw data
raw.plot(duration=795, scalings='auto', n_channels=10, title='EEG data plot', show=True)
plt.show()

plt.savefig(f'C:\\Users\\adity\\Desktop\\DATA for mini\\plots\\12\\{i}.png')
plt.close()