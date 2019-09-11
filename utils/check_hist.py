import matplotlib.pyplot as plt
import h5py

# Import data
DATA_PATH = "/home/alex/Desktop/NPix_dataset/ratL/"
f = h5py.File(DATA_PATH + "RatL_241018.mat") 

# Navigate database object like a dictionary
time_axis = f['v']['spikes']['iFRz']
fired_record = f['v']['spikes']['spikeHist']

# Check rate vs spikehist
plt.plot(f['v']['spikes']['iFRz'][30][:1000], c="r", label="Rate")
plt.scatter(range(1000),f['v']['spikes']['spikeHistZ'][30][:1000], marker='.', c="b",label="Binned Firing")
plt.xlabel("Time Axis (40ms Sampling on bins)")
plt.legend()
plt.show()