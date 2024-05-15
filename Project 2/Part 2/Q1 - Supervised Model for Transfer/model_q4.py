import torch
from copy import deepcopy
import torch.nn as nn
from model_q1 import ResCNN
import torch
from copy import deepcopy
from model_q1 import ResCNN

import torch.nn as nn

def get_feature_extractor():
    model = ResCNN()
    model_parameters_path = "./Model_Parameters/ResCNN_mitbih_best_parameters.pth"
    model.load_state_dict(torch.load(model_parameters_path))
    
    encoder_q1 = deepcopy(model)
    del encoder_q1.fc1
    del encoder_q1.fc2

    # Freeze all parameters
    for param in encoder_q1.parameters():
        param.requires_grad = False

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

def get_transfer_model():
    encoder_q1 = get_feature_extractor()
    encoder_mlp = deepcopy(encoder_q1)

    # Freeze the encoder
    for param in encoder_mlp.parameters():
        param.requires_grad = False

    # Add output layer(s) for PTB binary class
    encoder_mlp.fc1 = nn.Linear(in_features=encoder_mlp.fc1.in_features, out_features=64)
    encoder_mlp.fc2 = nn.Linear(in_features=64, out_features=2)
    encoder_mlp.forward = new_forward_mlp.__get__(encoder_mlp)

    return encoder_mlp

def new_forward_mlp(self, x):
    out = self.bn1(self.conv1(x))
    out = self.resblock3(self.resblock2(self.resblock1(out)))
    out = self.maxpoolf(out)
    out = out.view(out.size(0), -1)
    out = torch.relu(self.fc1(out))
    out = self.fc2(out)
    out = torch.sigmoid(out)
    return out