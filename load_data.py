import numpy as np
import h5py
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader

datapath = 'DATA/'

def openAndReshape(file_name, shape=None):
    file = h5py.File(file_name, 'r')
    data = file.get('my_data')
    data = np.array(data)
    if shape is not None:
        data = data.reshape(shape)
    print(file_name, " shape :", data.shape)
    return data

def dataCleaner(insoles, mocap, labels, seuil=3):
    """Permet de retirer les lignes possédant des valeurs aberrantes dans les données d'entrainement."""
    
    #On créer un masque pour retirer les lignes contenant des NaN
    masque_mocap = ~np.isnan(mocap).any(axis=(1, 2))
    masque_insoles = ~np.isnan(insoles).any(axis=(1, 2))
    masque = masque_mocap & masque_insoles
    print("Nombre de lignes valides :", np.sum(masque))

    # On applique le masque
    mocap_cleaned_nonan = mocap[masque]
    insoles_cleaned_nonan = insoles[masque]
    labels_cleaned_nonan = labels[masque]

    print("Mocap cleaned sans nan:", mocap_cleaned_nonan.shape)
    print("Insoles cleaned sans nan:", insoles_cleaned_nonan.shape)
    print("Labels cleaned sans nan:", labels_cleaned_nonan.shape)

    # On calcule la moyenne et l'écart type pour chaque capteur
    moy_insoles = np.mean(insoles_cleaned_nonan)
    ecart_type_insoles = np.std(insoles_cleaned_nonan)
    print("Moyenne insoles:", moy_insoles)
    print("Ecart type insoles:", ecart_type_insoles)

    moy_mocap = np.mean(mocap_cleaned_nonan)
    ecart_type_mocap = np.std(mocap_cleaned_nonan)
    print("Moyenne mocap:", moy_mocap)
    print("Ecart type mocap:", ecart_type_mocap)
    """
    # On crée un masque pour les valeurs aberrantes
    aberrantes_insoles = np.abs(insoles_cleaned_nonan - moy_insoles) > seuil * ecart_type_insoles
    aberrantes_mocap = np.abs(mocap_cleaned_nonan - moy_mocap) > seuil * ecart_type_mocap

    aberrantes_lignes_insoles = np.any(aberrantes_insoles, axis=(1, 2))
    aberrantes_lignes_mocap = np.any(aberrantes_mocap, axis=(1, 2))
    aberrantes_lignes = aberrantes_lignes_insoles | aberrantes_lignes_mocap
    print("Nombre de lignes aberrantes :", np.sum(aberrantes_lignes))

    # On applique le masque pour retirer les lignes contenant des valeurs aberrantes
    cleaned_insoles = insoles_cleaned_nonan[~aberrantes_lignes]
    cleaned_mocap = mocap_cleaned_nonan[~aberrantes_lignes]
    cleaned_labels = labels_cleaned_nonan[~aberrantes_lignes]

    print("Mocap cleaned sans aberrantes:", cleaned_mocap.shape)
    print("Insoles cleaned sans aberrantes:", cleaned_insoles.shape)
    print("Labels cleaned sans aberrantes:", cleaned_labels.shape)

    return cleaned_insoles, cleaned_mocap, cleaned_labels
    """
    return mocap_cleaned_nonan, insoles_cleaned_nonan, labels_cleaned_nonan



# --- Chargement des données d'entrainement ---
labels = openAndReshape(datapath + 'train_labels.h5')
train_insoles = openAndReshape(datapath + 'train_insoles.h5', shape=(6938, 100, 50))
train_mocap = openAndReshape(datapath + 'train_mocap.h5', shape=(6938, 100, 129))


train_insoles_cleaned, train_mocap_cleaned, labels_cleaned = dataCleaner(train_insoles, train_mocap, labels, seuil=3)


# --- Chargement des données de test ---
test_mocap = openAndReshape(datapath + 'test_mocap.h5', shape=(273, 100, 129))
test_insoles = openAndReshape(datapath + 'test_insoles.h5', shape=(273, 100, 50))
# --- Création du jeu de données d'entrainement ---
train_dataset = CustomDataset(train_insoles_cleaned, train_mocap_cleaned, labels_cleaned)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --- Création du jeu de données de test ---
test_dataset = CustomDataset(test_insoles, test_mocap)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)