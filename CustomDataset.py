import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, insoles, mocap, labels=None):
        self.insoles = torch.tensor(insoles, dtype=torch.float32)
        self.mocap = torch.tensor(mocap, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.insoles)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.insoles[idx], self.mocap[idx], self.labels[idx]
        else:
            return self.insoles[idx], self.mocap[idx]
