import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        # Séparation entre insoles (50 premières colonnes) et mocap (129 dernières)
        self.insoles = data[:, :, :50]
        self.mocap = data[:, :, 50:]

        # --- Traitement des insoles ---
        # 1. Séparation IUM (0:32) et Physique (32:50)
        self.ium = self.insoles[:, :, :32]
        self.phys = self.insoles[:, :, 32:]

        # 2. Normalisation min-max pour IUM
        ium_min = self.ium.min()
        ium_max = self.ium.max()
        self.ium = (self.ium - ium_min) / (ium_max - ium_min + 1e-8)

        # 3. Normalisation min-max pour Physiques
        phys_min = self.phys.min()
        phys_max = self.phys.max()
        self.phys = (self.phys - phys_min) / (phys_max - phys_min + 1e-8)

        # 4. Concatenation des insoles normalisées
        self.insoles = torch.tensor(torch.cat([self.ium, self.phys], dim=2), dtype=torch.float32)

        # --- Conversion du mocap (déjà prétraité dans load_data) ---
        self.mocap = torch.tensor(self.mocap, dtype=torch.float32)

        # --- Concaténation finale (insoles + mocap) ---
        self.data = torch.cat([self.insoles, self.mocap], dim=2)  # (N, 100, 179)

        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx].squeeze()
        else:
            return self.data[idx], torch.tensor(-1)
