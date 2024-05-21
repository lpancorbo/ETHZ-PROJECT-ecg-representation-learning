#import libraries
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import math
from copy import deepcopy
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, balanced_accuracy_score

###### MODEL FROM PART 1 (Modified for 5 classes) ######
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
        self.fc2 = nn.Linear(128, 5)
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
    
###### MODEL FOR TRANSFER LEARNING ######
class transfer_model(nn.Module):
    def __init__(self, base_model):
        super(transfer_model, self).__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(in_features=744, out_features=64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

def get_feature_extractor():
    model = ResCNN()
    model_parameters_path = "./ResCNN_mitbih_best_parameters.pth"
    model.load_state_dict(torch.load(model_parameters_path))
    
    encoder_q1 = deepcopy(model)
    del encoder_q1.fc1
    del encoder_q1.fc2
    
    # Modify forward function
    encoder_q1.forward = new_forward.__get__(encoder_q1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_q1.to(device)
    
    return encoder_q1

def new_forward(self, x):
    out = self.bn1(self.conv1(x))
    out = self.resblock3(self.resblock2(self.resblock1(out)))
    out = self.maxpoolf(out)
    out = out.view(out.size(0), -1)
    return out