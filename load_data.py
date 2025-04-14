import numpy as np
import h5py

datapath = 'DATA/'

file = h5py.File(datapath + 'train_labels.h5', 'r')
labels = file.get('my_data')
labels= np.array(labels)
print(labels.shape)

file = h5py.File(datapath + 'train_insoles.h5', 'r')
insoles = file.get('my_data')
insoles= np.array(insoles)
insoles= np.reshape(insoles,(6938,100,50))
print(insoles.shape)

file = h5py.File(datapath + 'train_mocap.h5', 'r')
mocap = file.get('my_data')
mocap= np.array(mocap)
mocap= np.reshape(mocap,(6938,100,129))
print(mocap.shape)

file = h5py.File(datapath + 'test_insoles.h5', 'r')
insoles = file.get('my_data')
insoles= np.array(insoles)
insoles = np.reshape(insoles, (273, 100, 50))
print(insoles.shape)



file = h5py.File(datapath + 'test_mocap.h5', 'r')
mocap = file.get('my_data')
mocap= np.array(mocap)
mocap= np.reshape(mocap,(273,100,129))
print(mocap.shape)