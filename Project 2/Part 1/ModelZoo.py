#import libraries
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


#Define my LSTM model
class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout_rate, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        #OUTPUT IS NOT A PROBABILITY; SO WE USE BCEWithLogitsLoss
        return out

# Create new bidirectional LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64,1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        _, (_,cf) = self.lstm(x, (h0, c0))
        out = F.elu(cf[-1, :, :]) #try to escape local minima
        out = self.fc(out)
        out = F.elu(out)
        out = self.fc2(out)
        #OUTPUT IS NOT A PROBABILITY; SO WE USE BCEWithLogitsLoss
        return out
    
#Define my CNN models
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        #create sequential block of convolution + maxpooling
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = nn.ELU()

    def forward(self, x):
        out=self.maxpool1(self.bn1(self.conv1(x)))
        out=self.maxpool2(self.bn2(self.conv2(out)))
        out=self.maxpool3(self.bn3(self.conv3(out)))
        out=out.view(out.size(0), -1)
        out=self.act(self.fc1(out))
        out=self.fc2(out)
        #OUTPUT IS NOT A PROBABILITY; SO WE USE BCEWithLogitsLoss
        return out
    
#Define my CNN model with residual connections

#Define a residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
        
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.act(out)
        return out

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        #Use resblocks
        self.resblock1 = ResBlock(1, 32, 3, 2, 1)
        self.maxpool1 = nn.MaxPool1d(2)
        self.resblock2 = ResBlock(32, 64, 3, 2, 1)
        self.maxpool2 = nn.MaxPool1d(2)
        self.resblock3 = ResBlock(64, 128, 3, 2, 1)
        self.maxpool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = nn.ELU()

    def forward(self, x):
        out=self.maxpool1(self.resblock1(x))
        out=self.maxpool2(self.resblock2(out))
        out=self.maxpool3(self.resblock3(out))
        out=out.view(out.size(0), -1)
        out=self.act(self.fc1(out))
        out=self.fc2(out)
        #OUTPUT IS NOT A PROBABILITY; SO WE USE BCEWithLogitsLoss
        return out
    
#Define my model with attention, not recurrent

