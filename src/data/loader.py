from typing import List
import pandas
import numpy
import torch
from torch.utils.data import Dataset


class CvrDataset(Dataset):
    def __init__(self,
                 features: numpy.matrix,
                 labels: numpy.ndarray,
                 transform=None):
        super().__init__()
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx:int) -> (torch.Tensor, torch.Tensor):
        line = self.features[idx].toarray()
        label = self.labels[idx]
        return torch.FloatTensor(line), torch.LongTensor([label])
    

def loader(dataset:Dataset, batch_size:int, shuffle:bool=True) -> torch.utils.data.DataLoader:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader