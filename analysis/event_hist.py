import matplotlib.pyplot as plt
import numpy as np
import h5py

# Import data
DATA_PATH = "/home/alex/Desktop/NPix_dataset/ratL/"

f = h5py.File(DATA_PATH + "RatL_241018.mat") 

# Navigate database object like a dictionary
fired_record = f['v']['spikes']['spikeHist'][0]
print(fired_record)