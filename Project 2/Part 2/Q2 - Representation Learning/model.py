#import libraries
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import math

#Define my CNN model with residual connections
# MODIFIED FOR MITBIH OUTPUT SIZE TO 5

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


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        #Use resblocks
        self.conv1 = nn.Conv1d(1, 24, 5, padding=1, stride=3)
        self.bn1 = nn.BatchNorm1d(24)
        self.resblock1 = ResBlock(24, 24, 3, 1, 1)
        self.resblock2 = ResBlock(24, 24, 3, 1, 1)
        self.resblock3 = ResBlock(24, 24, 3, 1, 1)
        self.maxpoolf = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(744, embedding_dim)
        self.act = nn.ELU()

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.resblock3(self.resblock2(self.resblock1(out)))
        out = self.maxpoolf(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Decoder(nn.Module):
    def __init__(self, embedding_dim=744, out_len=187):
        super(Decoder, self).__init__()
        self.enc_dim = embedding_dim
        self.out_len = out_len
        # Use transposed convolutions
        self.fc = nn.Linear(self.enc_dim, 24 * self.out_len)  # Adjust output shape to match input shape of the first conv layer in the encoder
        self.deconv1 = nn.ConvTranspose1d(24, 24, 3, padding=1, stride=1)  # Reverse of ResBlock
        self.bn1 = nn.BatchNorm1d(24)
        self.deconv2 = nn.ConvTranspose1d(24, 24, 3, padding=1, stride=1)  # Reverse of ResBlock
        self.bn2 = nn.BatchNorm1d(24)
        self.deconv3 = nn.ConvTranspose1d(24, 1, 3, padding=1, stride=1)  # Reverse of Conv1 in encoder
        self.bn3 = nn.BatchNorm1d(1)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), 24, self.out_len)  # Reshape to match the shape before max pooling in the encoder
        out = F.relu(self.bn1(self.deconv1(out)))
        out = F.relu(self.bn2(self.deconv2(out)))
        out = F.relu(self.bn3(self.deconv3(out)))
        return out


class AutoEncoder(nn.Module):
    def __init__(self, embedding_dim=128, mode="reconstruction"):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        self.act = nn.ELU()
        self.fc1 = nn.Linear(embedding_dim, 1)
        self.mode = mode

    def forward(self, x):
        out = self.encoder(x)
        if self.mode == "reconstruction":
            out = self.act(out)
            out = self.decoder(out)
        elif self.mode == "classification":
            out = self.fc1(out)
            out = F.sigmoid(out)
        return out


class Classifier(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        #self.fc = nn.Linear(embedding_dim, 1)
        self.act = nn.ELU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        #out = self.act(self.fc2(out))
        out = self.fc3(out)
        #out = self.fc(x)
        out = F.sigmoid(out)
        return out
