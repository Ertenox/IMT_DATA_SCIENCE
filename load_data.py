import numpy as np
import h5py

datapath = 'DATA/'

# --- Chargement des labels ---
file = h5py.File(datapath + 'train_labels.h5', 'r')
labels = file.get('my_data')
labels = np.array(labels)
print("Labels :", labels.shape)

# --- Chargement des insoles ---
file = h5py.File(datapath + 'train_insoles.h5', 'r')
train_insoles = file.get('my_data')
train_insoles = np.array(train_insoles)
train_insoles = train_insoles.reshape((6938, 100, 50))
print("Train insoles :", train_insoles.shape)

# --- Chargement des mocap ---
file = h5py.File(datapath + 'train_mocap.h5', 'r')
train_mocap = file.get('my_data')
train_mocap = np.array(train_mocap)
train_mocap = train_mocap.reshape((6938, 100, 129))
print("Train mocap :", train_mocap.shape)

# --- Création du masque : aucune valeur NaN dans mocap et insoles ---
masque_mocap = ~np.isnan(train_mocap).any(axis=(1, 2))
masque_insoles = ~np.isnan(train_insoles).any(axis=(1, 2))
masque = masque_mocap & masque_insoles
print("Nombre de lignes valides :", np.sum(masque))

# --- Filtrage ---
train_mocap_cleaned = train_mocap[masque]
train_insoles_cleaned = train_insoles[masque]
labels_cleaned = labels[masque]

print("Mocap cleaned :", train_mocap_cleaned.shape)
print("Insoles cleaned :", train_insoles_cleaned.shape)
print("Labels cleaned :", labels_cleaned.shape)

# --- Chargement des données de test ---
file = h5py.File(datapath + 'test_insoles.h5', 'r')
test_insoles = file.get('my_data')
test_insoles = np.array(test_insoles)
test_insoles = test_insoles.reshape((273, 100, 50))
print("Test insoles :", test_insoles.shape)

file = h5py.File(datapath + 'test_mocap.h5', 'r')
test_mocap = file.get('my_data')
test_mocap = np.array(test_mocap)
test_mocap = test_mocap.reshape((273, 100, 129))
print("Test mocap :", test_mocap.shape)
