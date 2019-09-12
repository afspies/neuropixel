import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.decomposition import PCA


# Import data
DATA_PATH = "../../../DR/DR/ws/ratL/"
f = h5py.File(DATA_PATH + "RatL_241018.mat")
# DATA_PATH = "./ratM/"
# f = h5py.File(DATA_PATH + "RatM_271118.mat")

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

print(x_new.shape)

print(pca.explained_variance_ratio_)
print(pca.singular_values_) 

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(x_new[:, 0], '-', label="Target neurons")#, markersize=0.4)
plt.title('PCA ratL')
plt.xlabel('time')
plt.ylabel('dimension1')
ax.yaxis.set_label_coords(-0.08,0.5)
plt.savefig('pca1.png')

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(x_new[:, 1], '-', label="Target neurons")#, markersize=0.4)
plt.title('PCA ratL')
plt.xlabel('time')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
plt.savefig('pca2.png')