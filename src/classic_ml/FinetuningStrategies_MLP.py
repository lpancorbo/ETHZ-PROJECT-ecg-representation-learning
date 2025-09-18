import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset_q1 import CustomTimeSeriesDataset, weighted_sampler_dataloader
from src.models.model_transfer_supervised import get_feature_extractor, transfer_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from sklearn.metrics import roc_curve,roc_auc_score

# TO DEFINE
TASK = 'C'
batch_size = 16
n_epochs = 100
name = f"transfer_model_{TASK}_{batch_size}batch_{n_epochs}epochs"

#############################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = get_feature_extractor()
feature_extractor.to(device)

model = transfer_model(feature_extractor)
model.to(device)

if TASK=='A' or TASK=='C':
    # Freeze all layers except fc1 and fc2
    for name_parameter, param in model.named_parameters():
        if 'fc1' in name_parameter or 'fc2' in name_parameter:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True)
elif TASK=='B':
    # Train all model but different lr for fc1 and fc2
    params_fc = [param for name_parameter, param in model.named_parameters() if 'fc1' in name_parameter or 'fc2' in name_parameter]
    params_encoder = [param for name_parameter, param in model.named_parameters() if 'fc1' not in name_parameter and 'fc2' not in name_parameter]
    lr_fc = 0.001
    lr_encoder = 0.0001
    param_groups = [
        {'params': params_fc, 'lr': lr_fc},
        {'params': params_encoder, 'lr': lr_encoder}
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0.0, nesterov=True)
    # Check if any layers are frozen
    frozen_layers = [name_parameter for name_parameter, param in model.named_parameters() if not param.requires_grad]
    if len(frozen_layers) == 0:
        print("No layers frozen!")

# Check which layers are trainable
# for name_parameter, param in model.named_parameters():
#     print(name_parameter, param.requires_grad)

# Load PTB
nettype = 'CNN'
dataset = CustomTimeSeriesDataset('ptbdb_train.csv', NetType=nettype)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = weighted_sampler_dataloader(train_dataset, batch_size=batch_size, repl=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)

dataset_test = CustomTimeSeriesDataset('ptbdb_test.csv', NetType=nettype)
test_loader = DataLoader(dataset_test, batch_size=len(dataset), shuffle=False)

# Set up training
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Move optimizer state to device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

val_loss = float('inf')
if TASK=='C':
    all_val_loss = [float('inf')] * (2 * n_epochs)
    all_train_loss = [float('inf')] * (2 * n_epochs)
    all_outputs = [float('inf')] * (2 * n_epochs)
    all_labels = [float('inf')] * (2 * n_epochs)
else:
    all_val_loss = [float('inf')] * n_epochs
    all_train_loss = [float('inf')] * n_epochs
    all_outputs = [float('inf')] * n_epochs
    all_labels = [float('inf')] * n_epochs

#To save the best model later
best_model = type(model)(feature_extractor)
for epoch in range(n_epochs):
    model.train()  # Set model to training mode
    #step the learning rate scheduler
    scheduler.step(val_loss)

    # Training loop
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
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
            inputs, labels = inputs.to(device), labels.to(device)
            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels.float())

    val_loss = val_loss.cpu().detach().numpy()
    #keep track of best model if validation loss is minimum
    if val_loss < min(all_val_loss):
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%, new best model!')
        # Save the model parameters,in case we lose them
        best_model.load_state_dict(model.state_dict())  # Copy the model parameters
    else:
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%')

    all_val_loss[epoch]=val_loss

if TASK=='C':
    # Unfreeze all layers
    for name_parameter, param in model.named_parameters():
        param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    # Move optimizer state to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Retrain model
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        #step the learning rate scheduler
        scheduler.step(val_loss)

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            if i==0:
                all_train_loss[epoch + n_epochs]=loss.cpu().detach().numpy()
                #Save the outputs and correct labels of all batches, as numpy arrays
                all_outputs[epoch + n_epochs]=outputs.cpu().detach().numpy()
                all_labels[epoch + n_epochs]=labels.cpu().detach().numpy()
            
        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                val_outputs = model(inputs)
                val_loss = criterion(val_outputs, labels.float())

        val_loss = val_loss.cpu().detach().numpy()
        #keep track of best model if validation loss is minimum
        if val_loss < min(all_val_loss):
            print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%, new best model!')
            # Save the model parameters,in case we lose them
            best_model.load_state_dict(model.state_dict())  # Copy the model parameters
        else:
            print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%')

        all_val_loss[epoch + n_epochs]=val_loss

torch.save({
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict' : scheduler.state_dict(),
    }, os.path.join('models', f'{name}_scheduler_optimizer_state.pth'))

torch.save(best_model.state_dict(), os.path.join('models', f'{name}_best_parameters.pth'))

val_outputs = val_outputs.cpu().detach().numpy()
print('Finished Training')

all_outputs = np.concatenate(all_outputs)
all_labels = np.concatenate(all_labels)

#clear gpu
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

# Evaluation
best_model.to(device)
best_model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs).detach().cpu().numpy()
        labels=labels.detach().cpu().numpy()

#Plot ROC curve
fpr, tpr, thresh = roc_curve(labels, outputs)
roc_auc = round(roc_auc_score(labels, outputs),4)

axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2,label=f'ROC AUC {name}: {roc_auc}')
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('ROC curve')
axs[1, 1].legend(loc="lower right")

#Save all figures
plt.savefig(os.path.join('figures', f'{name}_train_and_test.png'))
plt.close()