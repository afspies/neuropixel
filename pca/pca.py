import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.decomposition import PCA


# Import data
# DATA_PATH = "../../../DR/DR/ws/ratL/"
# f = h5py.File(DATA_PATH + "RatL_241018.mat")
# speed = np.load("../pca/L/speed.npy")
# d4 = np.load("../../../DR/DR/ws/ratL/PutativeInterneuron.npy")[0]

DATA_PATH = "../../../DR/DR/ws/ratM/"
f = h5py.File(DATA_PATH + "RatM_271118.mat")
speed = np.load("../pca/M/speed.npy")
d4 = np.load("../../../DR/DR/ws/ratM/PutativeInterneuron.npy")
print(d4.shape)

time_axis = f['v']['spikes']['tb']
hist = f['v']['spikes']['spikeHist']
histZ = f['v']['spikes']['spikeHistZ']

ifr = f['v']['spikes']['iFR']
ifrZ = f['v']['spikes']['iFRz']

secs = int(time_axis.shape[0]/25)
neurons = hist.shape[0]

res = np.zeros(shape=(neurons, secs))

for i in range(0, secs):
	x = ifrZ[:, i*25:i*25+24]
	res[:, i] = np.mean(x, axis=1)


pca = PCA(n_components=2)
x_new = pca.fit_transform(np.transpose(res))
x_wts = pca.components_


print(x_new.shape)

print(pca.explained_variance_ratio_)
print(pca.singular_values_) 

speed_s = [sum(speed[i:i+25]) for i in range(0, len(speed), 25)]
speed_s = np.asarray(speed_s)/25
dim1 = x_new[:, 0]

#speed_s = (speed_s-np.mean(speed_s))/np.std(speed_s)
#dim1 = (dim1-np.mean(dim1))/np.std(dim1)

fig = plt.figure(figsize=(9,4))
ax = plt.subplot(111)
ax.plot(dim1, '-', label="dimension1")#, markersize=0.4)
plt.title('PCA ratL')
plt.xlabel('time')
plt.ylabel('dimension1')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
ax.yaxis.set_label_coords(-0.08,0.5)
plt.savefig('pca1.png')

fig = plt.figure(figsize=(9,4))
ax = plt.subplot(111)
ax.plot(x_new[:, 1], '-', label="dimension2")#, markersize=0.4)
plt.title('PCA ratL')
plt.xlabel('time')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
plt.savefig('pca2.png')

fig = plt.figure(figsize=(9,4))
ax = plt.subplot(111)
ax.plot(speed_s, '-', label="speed")#, markersize=0.4)
plt.title('PCA ratL')
plt.xlabel('time')
plt.ylabel('speed')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
ax.yaxis.set_label_coords(-0.08,0.5)
plt.savefig('speed.png')



wts_in1 = []
wts_in2 = []
wts_rst1 = []
wts_rst2 = []

for i in range(0,neurons):
	if d4[i]:
		wts_in1.append(x_wts[0, :][i])
		wts_in2.append(x_wts[1, :][i])
	else:
		wts_rst1.append(x_wts[0, :][i])
		wts_rst2.append(x_wts[1, :][i])

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
#ax.plot(x_wts[0, :], x_wts[1, :], 'o', label="speed")#, markersize=0.4)
ax.plot(wts_in1, wts_in2, 'o', label="Interneuron")
ax.plot(wts_rst1, wts_rst2, 'o', label="Excitatory")
plt.title('PCA biplot ratL')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
ax.yaxis.set_label_coords(-0.08,0.5)
plt.savefig('biplot.png')