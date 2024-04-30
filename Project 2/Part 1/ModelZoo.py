#import libraries
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


#Define my LSTM model
class simpleLSTM(nn.Module):
    def __init__(self):
        super(simpleLSTM, self).__init__()
        input_size=1
        hidden_size=128
        num_layers=5
        dropout_rate=0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout_rate, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64,1)
        self.act = nn.ELU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.act(self.fc(out[:, -1, :]))
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

# Create new bidirectional LSTM
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        input_size=1
        hidden_size=128
        num_layers=5
        dropout_rate=0
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
        out = F.sigmoid(out)
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
        out = F.sigmoid(out)
        return out
    
#Define my CNN model with residual connections

#Define a residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size, stride, padding)
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
        self.conv1 = nn.Conv1d(1, 24, 5, padding=1, stride=3)
        self.bn1 = nn.BatchNorm1d(24)
        self.resblock1 = ResBlock(24, 24, 3, 1, 1)
        self.resblock2 = ResBlock(24, 24, 3, 1, 1)
        self.resblock3 = ResBlock(24, 24, 3, 1, 1)
        self.maxpoolf = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(744, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = nn.ELU()

    def forward(self, x):
        out=self.bn1(self.conv1(x))
        out=self.resblock3(self.resblock2(self.resblock1(out)))
        out=self.maxpoolf(out)
        out=out.view(out.size(0), -1)
        out=self.act(self.fc1(out))
        out=self.fc2(out)
        out = F.sigmoid(out)
        return out

#Define my model with attention, not recurrent
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.MaxPool1d(2)
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=384, nhead=1,norm_first=True,batch_first=True,dim_feedforward=1), num_layers=2,enable_nested_tensor=False)
        self.linear2 = nn.Linear(384,128)
        self.linear3 = nn.Linear(128,1)
        self.act = nn.ELU()
    def forward(self, x):
        out=self.maxpool1(self.bn1(self.conv1(x)))
        out=self.maxpool2(self.bn2(self.conv2(out)))
        out=self.maxpool3(self.bn3(self.conv3(out)))
        out=out.view(out.size(0), -1)
        out=self.TransformerEncoder(out)
        out=self.act(self.linear2(out))
        out=self.linear3(out)
        out = F.sigmoid(out)
        return out