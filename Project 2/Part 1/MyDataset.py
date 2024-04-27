import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class CustomTimeSeriesDataset(Dataset):
    def __init__(self, csv,NetType=None):
        df=pd.read_csv(csv)
        if NetType=="LSTM":
            self.timeseries=torch.from_numpy(df.values[:,0:-1]).float().unsqueeze(2)
            self.labels=torch.from_numpy(df.values[:,-1]).float().unsqueeze(1)
        elif NetType=="CNN":
            #switch dimension to (batch, channel, seq_len)
            self.timeseries=torch.from_numpy(df.values[:,0:-1]).float().unsqueeze(1)
            self.labels=torch.from_numpy(df.values[:,-1]).float().unsqueeze(1)
        else:
            self.timeseries=torch.from_numpy(df.values[:,0:-1]).float()
            self.labels=torch.from_numpy(df.values[:,-1]).float()

    def __len__(self):
        return len(self.timeseries)

    def __getitem__(self, idx):
        timeserie = self.timeseries[idx] ### THIS IS BECAUSE LSTM (with batch_first=True) expects a 3D tensor (seq_len, batch, input_size), and input_size is 1.
        label = self.labels[idx]
        return timeserie, label
    
def weighted_sampler_dataloader(dataset_subset, batch_size, repl=True):
    target=dataset_subset.dataset.labels[dataset_subset.indices]
    class_sample_count = np.array(    [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    weight=weight/weight.sum()
    samples_weight = torch.tensor([weight[int(t)] for t in target]).double()    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=repl)    
    return DataLoader(dataset_subset, batch_size=batch_size, sampler=sampler)