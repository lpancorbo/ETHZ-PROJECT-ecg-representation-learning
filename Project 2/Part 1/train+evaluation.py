#import libraries
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, unpad_sequence
from MyDataset import CustomTimeSeriesDataset, weighted_sampler_dataloader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from ModelZoo import simpleCNN, simpleLSTM, BiLSTM, ResCNN, TransformerWithAttentionOutputted
from sklearn.metrics import roc_curve,roc_auc_score

#Define model to be trained
name = "finalBiLSTM"
model = BiLSTM()
nettype = 'LSTM'
continue_training = False
batch_size = 16

#Define optimizer as SGD with a lot of hyperparameters to avoid local minima

#For simpleCNN, ResCNN, TransformerWithAttentionOutputted: 
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True)
#FOR simpleLSTM and BiLSTM, USED BY JONA: 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

if continue_training:
    model.load_state_dict(torch.load(os.path.join('Model_Parameters', f'{name}_best_parameters.pth')))
    checkpoint = torch.load(os.path.join('Model_Parameters', f'{name}_scheduler_optimizer_state.pth'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Number of epochs
n_epochs = 200

## Initialize dataset and do train validation split

# Create dataset object
dataset = CustomTimeSeriesDataset('ptbdb_train.csv', NetType=nettype)

# Define the split sizes. In this case, 70% for training and 30% for validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Create two Datasets from the original one
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

## Training loop

#Initialize train dataloader
train_loader = weighted_sampler_dataloader(train_dataset, batch_size=batch_size, repl=True)
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #was not using this because of the weighted sampler

#Initialize validation loader THE BATCH SIZE NEEDS TO BE AS BIG AS POSSIBLE BECAUSE THERE ARE NO GRADIENTS. IF I CAN RUN ALL VAL SET IN PARALLEL, BETTER!
val_loader = DataLoader(val_dataset, batch_size=2328, shuffle=False)

#Define loss function and optimizer as BCE, as output is NOW a probability
criterion = nn.BCELoss()

#Sent model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Move optimizer state to device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)
            
# Loop over the dataset multiple times
#make initial validation loss infinite
val_loss = float('inf')

#Set the val loss array with the size of the number of epochs
all_val_loss = [float('inf')] * n_epochs
all_train_loss = [float('inf')] * n_epochs
all_outputs = [float('inf')] * n_epochs
all_labels = [float('inf')] * n_epochs

#To save the best model later
best_model = type(model)()
for epoch in range(n_epochs):
    model.train()  # Set model to training mode
    #step the learning rate scheduler
    scheduler.step(val_loss)

    # Training loop
    for i, (inputs, labels) in enumerate(train_loader):

        if nettype == 'LSTM':
            lengths = []
            for sequence in inputs[:, :, 0]:
                length = torch.nonzero(sequence).size(0)
                lengths.append(length)
                #lengths.append(sequence.size(0))
            inputs = pack_padded_sequence(inputs, lengths=lengths, batch_first=True, enforce_sorted=False)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        if name=="TransformerWithAttentionOutputted":
            outputs,_ = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if i==0:
            all_train_loss[epoch]=loss.cpu().detach().numpy()
            #Save the outputs and correct labels of all batches, as numpy arrays
            all_outputs[epoch]=outputs.cpu().detach().numpy()
            all_labels[epoch]=labels.cpu().detach().numpy()

    # Validation loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            if nettype == 'LSTM':
                lengths = []
                for sequence in inputs[:, :, 0]:
                    length = torch.nonzero(sequence).size(0)
                    lengths.append(length)
                    #lengths.append(sequence.size(0))
                inputs = pack_padded_sequence(inputs, lengths=lengths, batch_first=True, enforce_sorted=False)

            inputs = inputs.to(device)
            labels = labels.to(device)

            if name=="TransformerWithAttentionOutputted":
                val_outputs,_ = model(inputs)
            else:
                val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)

    val_loss = val_loss.cpu().detach().numpy()
    #keep track of best model if validation loss is minimum
    if val_loss < min(all_val_loss):
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%, new best model!')
        # Save the model parameters,in case we lose them
        best_model.load_state_dict(model.state_dict())  # Copy the model parameters
    else:
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%')

    all_val_loss[epoch]=val_loss

torch.save({
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict' : scheduler.state_dict(),
    }, os.path.join('Model_Parameters', f'{name}_scheduler_optimizer_state.pth'))

torch.save(best_model.state_dict(), os.path.join('Model_Parameters', f'{name}_best_parameters.pth'))

val_outputs = val_outputs.cpu().detach().numpy()

print('Finished Training')

all_outputs = np.concatenate(all_outputs)
all_labels = np.concatenate(all_labels)

#clar gpu
torch.cuda.empty_cache()

#Create a figure with subfigures
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

axs[0, 0].plot(all_outputs[all_labels==1], 'ro', label='True')
axs[0, 0].plot(all_outputs[all_labels==0], 'bo', label='False')
axs[0, 0].legend()
axs[0, 0].set_xlabel('Batch')
axs[0, 0].set_ylabel('Probability Value')
axs[0, 0].set_title('Probability given by the model as a function of batch and label')

#Plot histogram of validation outputs
axs[0, 1].hist(val_outputs, bins=100)
axs[0, 1].set_ylabel('Probability Value')
axs[0, 1].set_title('Probabilities given by the model for validation set')

# Add the subfigure to the big figure
axs[1, 0].plot(all_val_loss)
axs[1, 0].plot(all_train_loss)
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].set_title('Train and Validation Loss as function of epoch')
axs[1, 0].legend(['Validation Loss', 'Train Loss'])

#EVALUATION
#Prepare test dataset
dataset_test = CustomTimeSeriesDataset('ptbdb_test.csv', NetType=nettype)
test_loader = DataLoader(dataset_test, batch_size=2910, shuffle=False)

# Test loop
#send best model to device
best_model.to(device)
best_model.eval()  # Set best model to evaluation mode
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        if nettype == 'LSTM':
            lengths = []
            for sequence in inputs[:, :, 0]:
                length = torch.nonzero(sequence).size(0)
                lengths.append(length)
                #lengths.append(sequence.size(0))
            inputs = pack_padded_sequence(inputs, lengths=lengths, batch_first=True, enforce_sorted=False)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass and prepare for plot
        if name=="TransformerWithAttentionOutputted":
            outputs = best_model(inputs)[0].detach().cpu().numpy()
        else:
            outputs = best_model(inputs).detach().cpu().numpy()
        labels=labels.detach().cpu().numpy()

#Plot ROC curve
fpr, tpr, _ = roc_curve(labels, outputs)
roc_auc = round(roc_auc_score(labels, outputs),4)

axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2,label=f'ROC AUC {name}: {roc_auc}')
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('ROC curve')
axs[1, 1].legend(loc="lower right")

#Save all figures
plt.savefig(os.path.join('Figures', f'{name}_train_and_test.png'))
plt.close()
