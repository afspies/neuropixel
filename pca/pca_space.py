import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.decomposition import PCA

# Import data
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

np.save("tx12.npy", tx12)
np.save("tx21.npy", tx21)
np.save("tx22.npy", tx22)
np.save("tx23.npy", tx23)
np.save("tx32.npy", tx32)


pca12 = PCA(n_components=5)
x_new12 = pca12.fit_transform(np.transpose(tx12))

pca21 = PCA(n_components=5)
x_new21 = pca21.fit_transform(np.transpose(tx21))

pca22 = PCA(n_components=5)
x_new22 = pca22.fit_transform(np.transpose(tx22))

pca23 = PCA(n_components=5)
x_new23 = pca23.fit_transform(np.transpose(tx23))

pca32 = PCA(n_components=5)
x_new32 = pca32.fit_transform(np.transpose(tx32))

print(pca12.explained_variance_ratio_)
print(pca21.explained_variance_ratio_)
print(pca22.explained_variance_ratio_)
print(pca23.explained_variance_ratio_)
print(pca32.explained_variance_ratio_)
print(pca12.singular_values_) 

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(range(-250,250), x_new12[:, 0], '-', label="dim1 on 12")
ax.plot(range(-250,250), x_new21[:, 0], '-', label="dim1 on 21")
ax.plot(range(-250,250), x_new22[:, 0], '-', label="dim1 on 22")
ax.plot(range(-250,250), x_new23[:, 0], '-', label="dim1 on 23")
ax.plot(range(-250,250), x_new32[:, 0], '-', label="dim1 on 32")
plt.title('PCA ratL trials')
plt.xlabel('time')
plt.ylabel('dimensions')
ax.yaxis.set_label_coords(-0.08,0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1), shadow=True, ncol=1)
#plt.show()
plt.savefig('pca_dim1.png')


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(range(-250,250), x_new12[:, 1], '-', label="dim1 on 12")
ax.plot(range(-250,250), x_new21[:, 1], '-', label="dim1 on 21")
ax.plot(range(-250,250), x_new22[:, 1], '-', label="dim1 on 22")
ax.plot(range(-250,250), x_new23[:, 1], '-', label="dim1 on 23")
ax.plot(range(-250,250), x_new32[:, 1], '-', label="dim1 on 32")
plt.title('PCA ratL trials')
plt.xlabel('time')
plt.ylabel('dimensions')
ax.yaxis.set_label_coords(-0.08,0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1), shadow=True, ncol=1)
#plt.show()
plt.savefig('pca_dim2.png')

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(x_new12[:, 0], x_new12[:, 1], '-', label="dim1 on 12")
ax.plot(x_new21[:, 0], x_new21[:, 1], '-', label="dim1 on 21")
ax.plot(x_new22[:, 0], x_new22[:, 1], '-', label="dim1 on 22")
ax.plot(x_new23[:, 0], x_new23[:, 1], '-', label="dim1 on 23")
ax.plot(x_new32[:, 0], x_new32[:, 1], '-', label="dim1 on 32")
plt.title('PCA ratL trials')
plt.xlabel('time')
plt.ylabel('dimensions')
ax.yaxis.set_label_coords(-0.08,0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1), shadow=True, ncol=1)
#plt.show()
plt.savefig('pca_dim1_2.png')


# print pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2'])