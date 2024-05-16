import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


def last_nonzero_index(row):
    indices = np.nonzero(row)[0]
    return indices[-1] if indices.size > 0 else -1


class CustomTimeSeriesDataset(Dataset):
    def __init__(self, csv, NetType=None, nan=False):
        df = pd.read_csv(csv)
        timeseries = df.values[:, :-1]
        if nan:
            lengths = np.apply_along_axis(last_nonzero_index, axis=1, arr=timeseries)
            mask = np.arange(timeseries.shape[1]) > lengths[:, np.newaxis]
            timeseries[mask] = np.nan
        if NetType == "LSTM":
            self.timeseries = torch.from_numpy(timeseries).float().unsqueeze(2)
            self.labels = torch.from_numpy(df.values[:, -1]).float().unsqueeze(1)
        elif NetType == "CNN" or NetType == "Transformer":
            # switch dimension to (batch, channel, seq_len)
            self.timeseries = torch.from_numpy(timeseries).float().unsqueeze(1)
            self.labels = torch.from_numpy(df.values[:, -1]).float().unsqueeze(1)
        else:
            self.timeseries = torch.from_numpy(timeseries).float()
            self.labels = torch.from_numpy(df.values[:, -1]).float()

    def __len__(self):
        return len(self.timeseries)

    def __getitem__(self, idx):
        timeseries = self.timeseries[idx] ### THIS IS BECAUSE LSTM (with batch_first=True) expects a 3D tensor (seq_len, batch, input_size), and input_size is 1.
        label = self.labels[idx]
        return timeseries, label
    
def weighted_sampler_dataloader(dataset_subset, batch_size, repl=True):
    target=dataset_subset.dataset.labels[dataset_subset.indices]
    class_sample_count = np.array(    [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    weight=weight/weight.sum()
    samples_weight = torch.tensor([weight[int(t)] for t in target]).double()    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=repl)    
    return DataLoader(dataset_subset, batch_size=batch_size, sampler=sampler)