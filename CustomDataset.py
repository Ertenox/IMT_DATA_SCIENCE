import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            self.labels = self.labels.squeeze()
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx],torch.tensor(-1)
