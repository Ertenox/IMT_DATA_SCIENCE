import numpy as np
import h5py
from sklearn.utils import compute_class_weight
import torch
from CustomDataset import InsolesDataset
from torch.utils.data import DataLoader, Subset
from torch import nn
from Network import Network
from sklearn.model_selection import KFold

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
    return mocap_cleaned_nonan, insoles_cleaned_nonan, labels_cleaned_nonan

def select_features(data):
    """Garde 16 premières colonnes, saute 9, garde 16 suivantes, saute les 9 dernières sur l'axe capteurs (dim 2)."""
    return np.concatenate([data[:, :, 0:16], data[:, :, 25:41]], axis=2)


def train(dataloader, model, loss_fn, optim, device):
    """Fonction d'entrainement du modèle"""
    model.train()
    sum_loss = 0
    total = 0
    for step, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        total += labels.size(0)
        sum_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    
    return sum_loss / total

def test(dataloader, model, loss_fn, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            predicted = output.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test dataset: '+str(100 * correct / total)+' %')
    return 100 * correct / total

# --- Chargement des données d'entrainement ---
labels = openAndReshape(datapath + 'train_labels.h5')
train_insoles = openAndReshape(datapath + 'train_insoles.h5', shape=(6938, 100, 50))
train_mocap = openAndReshape(datapath + 'train_mocap.h5', shape=(6938, 100, 129))
train_insoles = select_features(train_insoles)
print("taille après select: ", train_insoles.shape)

train_mocap_cleaned, train_insoles_cleaned, labels_cleaned = dataCleaner(train_insoles, train_mocap, labels, seuil=3)

# --- Chargement des données de test ---
test_mocap = openAndReshape(datapath + 'test_mocap.h5', shape=(273, 100, 129))
test_insoles = openAndReshape(datapath + 'test_insoles.h5', shape=(273, 100, 50))
test_insoles = select_features(test_insoles)

print("Valeurs uniques des labels (bruts) :", np.unique(labels))
# --- Création des datasets ---
train_dataset = InsolesDataset(train_insoles_cleaned, labels_cleaned)
test_dataset = InsolesDataset(test_insoles)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    print(f"\n--- Fold {fold+1} ---")
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = Network(input_size=32, hidden_size=256, num_classes=13, num_layers=2, dropout_rate=0.5).to(device)
    classes = np.unique(labels_cleaned)
    weight = compute_class_weight(class_weight='balanced', classes=classes, y=labels_cleaned.ravel())
    weight = torch.tensor(weight, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        loss = train(train_loader, model, loss_fn, optimizer, device)
        val_acc = test(val_loader, model, loss_fn, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    final_acc = test(val_loader, model, loss_fn, device)
    fold_accuracies.append(final_acc)
    print(f"Accuracy for fold {fold+1}: {final_acc:.2f}%")

# --- Résumé final ---
print("\n--- Résultat K-Fold ---")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.2f}%")
print(f"Moyenne : {np.mean(fold_accuracies):.2f}%, Écart-type : {np.std(fold_accuracies):.2f}%")