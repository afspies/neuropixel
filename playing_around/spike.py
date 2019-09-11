import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
from matplotlib.colors import to_rgba_array
# Import data
DATA_PATH = "/home/alex/Desktop/NPix_dataset/ratL/"

f = h5py.File(DATA_PATH + "RatL_241018.mat") 

# Navigate database object like a dictionary
time_axis = f['v']['spikes']['tb']
fired_record = f['v']['spikes']['spikeHist']
time_steps = 1000
neuron = fired_record[0]

# Create quantities for plot
arr = []
color = []
viridis = cm.get_cmap('viridis', 8).colors
print(viridis)
for time_step in range(100):#len(neuron)):
    times_fired = int(neuron[time_step])
    if times_fired > 0:
        arr.append(time_step)
        color.append(tuple(viridis[int(times_fired)]))                       

print(len(arr), len(color))
# Draw a spike raster plot
plt.eventplot(arr, colors=to_rgba_array(color), orientation="horizontal") 
plt.title('Spike raster plot')
plt.show()

