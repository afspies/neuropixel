import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.decomposition import PCA
import pandas as pd

# Import data
DATA_PATH = "../../../DR/DR/ws/ratL/"
f = h5py.File(DATA_PATH + "RatL_241018.mat")
# DATA_PATH = "./ratL/"
# f = h5py.File(DATA_PATH + "RatL_241018.mat")

time_axis = f['v']['spikes']['tb']
hist = f['v']['spikes']['spikeHist']
histZ = f['v']['spikes']['spikeHistZ']

ifr = f['v']['spikes']['iFR']
ifrZ = f['v']['spikes']['iFRz']

ctimes = f['v']['event']['CorrectTimes'][0]
ictimes = f['v']['event']['ErrorTimes'][0]

neurons = hist.shape[0]

cx = np.zeros(shape=(neurons, 500))
ex = np.zeros(shape=(neurons, 500))


start = time_axis[0]
for i in range(0, ctimes.size-1):
	ctime = ctimes[i]
	bin = int((ctime-start)*1000/40)
	cx = cx + ifrZ[ : , bin-250:bin+250]

for i in range(0, ictimes.size-1):
	ictime = ictimes[i]
	bin = int((ictime-start)*1000/40)
	ex = ex + ifrZ[ : , bin-250:bin+250]

tx = (cx + ex)/ (ctimes.size-1 + ictimes.size-1 )
cx = cx/(ctimes.size-1)
ex = ex/(ictimes.size-1)

print(tx.shape)
print(cx.shape)
print(ex.shape)


pca = PCA(n_components=2)
x_new = pca.fit_transform(np.transpose(cx))
x_wts = pca.components_

pca = PCA(n_components=2)
ix_new = pca.fit_transform(np.transpose(ex))
ix_wts = pca.components_


print(pca.explained_variance_ratio_)
print(pca.singular_values_) 

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(x_new[:, 0], '-', label="dim1 on CorrectTimes")
ax.plot(x_new[:, 1], '-', label="dim2 on CorrectTimes")
ax.plot(ix_new[:, 0], '-', label="dim1 on IncorrectTimes")
ax.plot(ix_new[:, 1], '-', label="dim2 on IncorrectTimes")
plt.title('PCA ratL')
plt.xlabel('time')
plt.ylabel('dimensions')
ax.yaxis.set_label_coords(-0.08,0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1), shadow=True, ncol=1)
#plt.show()
plt.savefig('pca_trial.png')


d4 = np.load("../../../DR/DR/ws/ratL/PutativeInterneuron.npy")[0]
inx = np.zeros(neurons)

for i in range(0,neurons):
	if d4[i]:
		inx[i] = -0.3

idx = x_wts[1, :].argsort()
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
inx = inx[idx]
ax.plot(x_wts[1, :][idx], '-', label="wts on CorrectTimes")
ax.plot(ix_wts[1, :][idx], '-', label="wts on InorrectTimes")
ax.plot(np.nonzero(inx)[0], np.full(np.nonzero(inx)[0].shape, -0.3), 'o', label="PutativeInterneuron")
for i in range(np.nonzero(inx)[0].size):
	plt.axvline(x=np.nonzero(inx)[0][i], color='gray', linestyle='--')
plt.title('PCA weights')
plt.xlabel('neurons')
plt.ylabel('weight')
ax.yaxis.set_label_coords(-0.08,0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1), shadow=True, ncol=1)
#plt.show()
plt.savefig('pca_trial_wts.png')



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

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(wts_in1, wts_in2, 'o', label="Interneuron")
ax.plot(wts_rst1, wts_rst2, 'o', label="Excitatory")
plt.title('Bi plot')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1), shadow=True, ncol=1)
#plt.show()
plt.savefig('pca_trial_biplot.png')

# pca = PCA(n_components=2)
# tx_new = pca.fit_transform(np.transpose(tx))
# tx_wts = pca.components_


# idx = x_wts[0, :].argsort()
# fig = plt.figure(figsize=(8,8))
# ax = plt.subplot(111)
# ax.plot(x_wts[0, :][idx], '-', label="wts on CorrectTimes")
# ax.plot(ix_wts[1, :][idx], '-', label="wts on InorrectTimes")
# plt.title('PCA weights')
# plt.xlabel('neurons')
# plt.ylabel('weight')
# ax.yaxis.set_label_coords(-0.08,0.5)
# ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1), shadow=True, ncol=1)
# #plt.show()
# plt.savefig('pca_trial_wts.png')
