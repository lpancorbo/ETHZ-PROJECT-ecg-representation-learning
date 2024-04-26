import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

class CustomTimeSeriesDataset(Dataset):
    def __init__(self, csv,LSTM=False):
        df=pd.read_csv(csv)
        if LSTM:
            self.timeseries=torch.from_numpy(df.values[:,0:-1]).float().unsqueeze(2)
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