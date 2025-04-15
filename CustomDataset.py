import torch
from torch.utils.data import Dataset

class InsolesDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data

        data_min = self.data.min()
        data_max = self.data.max()
        self.data = (self.data - data_min) / (data_max - data_min + 1e-8)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx].squeeze()
        else:
            return self.data[idx], torch.tensor(-1)