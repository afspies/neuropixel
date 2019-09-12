import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as sio
from sklearn.decomposition import PCA

# Import data
# DATA_PATH = "../../../DR/DR/ws/ratM/"
# f = h5py.File(DATA_PATH + "RatM_271118.mat")

# time_stamps12 = np.loadtxt("./space/M/12.dat")[:,1]
# time_stamps21 = np.loadtxt("./space/M/21.dat")[:,1]
# time_stamps22 = np.loadtxt("./space/M/22.dat")[:,1]
# time_stamps23 = np.loadtxt("./space/M/23.dat")[:,1]	
# time_stamps32 = np.loadtxt("./space/M/32.dat")[:,1]

DATA_PATH = "../../../DR/DR/ws/ratL/"
f = h5py.File(DATA_PATH + "RatL_241018.mat")

time_stamps12 = np.loadtxt("./space/12.dat")[:,1]
time_stamps21 = np.loadtxt("./space/21.dat")[:,1]
time_stamps22 = np.loadtxt("./space/22.dat")[:,1]
time_stamps23 = np.loadtxt("./space/23.dat")[:,1]	
time_stamps32 = np.loadtxt("./space/32.dat")[:,1]


time_axis = f['v']['spikes']['tb']

ifr = f['v']['spikes']['iFR']
ifrZ = f['v']['spikes']['iFRz']

neurons = ifr.shape[0]

tx12 = np.zeros(shape=(neurons, 500))
tx21 = np.zeros(shape=(neurons, 500))
tx22 = np.zeros(shape=(neurons, 500))
tx23 = np.zeros(shape=(neurons, 500))
tx32 = np.zeros(shape=(neurons, 500))


start = time_axis[0]
for i in range(0, time_stamps12.size):
	time_stamp = time_stamps12[i]
	bin = int((time_stamp-start)*1000/40)
	tx12 = tx12 + ifrZ[ : , bin-250:bin+250]

tx12 = tx12/(time_stamps12.size)
print(tx12.shape)

start = time_axis[0]
for i in range(0, time_stamps21.size):
	time_stamp = time_stamps21[i]
	bin = int((time_stamp-start)*1000/40)
	tx21 = tx21 + ifrZ[ : , bin-250:bin+250]

tx21 = tx21/(time_stamps21.size)
print(tx21.shape)

start = time_axis[0]
for i in range(0, time_stamps22.size):
	time_stamp = time_stamps22[i]
	bin = int((time_stamp-start)*1000/40)
	tx22 = tx22 + ifrZ[ : , bin-250:bin+250]

tx22 = tx22/(time_stamps22.size)
print(tx22.shape)

start = time_axis[0]
for i in range(0, time_stamps23.size):
	time_stamp = time_stamps23[i]
	bin = int((time_stamp-start)*1000/40)
	if ifrZ[ : , bin-250:bin+250].shape[1]!=0:
		print(i)
		tx23 = tx23 + ifrZ[ : , bin-250:bin+250]

tx23 = tx23/(time_stamps23.size)
print(tx23.shape)

start = time_axis[0]
for i in range(0, time_stamps32.size):
	time_stamp = time_stamps32[i]
	bin = int((time_stamp-start)*1000/40)
	tx32 = tx32 + ifrZ[ : , bin-250:bin+250]

tx32 = tx32/(time_stamps32.size)
print(tx32.shape)

np.save("L/tx12.npy", tx12)
np.save("L/tx21.npy", tx21)
np.save("L/tx22.npy", tx22)
np.save("L/tx23.npy", tx23)
np.save("L/tx32.npy", tx32)

sio.savemat('L/tx12.mat', {'data':tx12})
sio.savemat('L/tx21.mat', {'data':tx21})
sio.savemat('L/tx22.mat', {'data':tx22})
sio.savemat('L/tx23.mat', {'data':tx23})
sio.savemat('L/tx32.mat', {'data':tx32})