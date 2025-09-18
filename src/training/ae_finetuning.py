# import libraries
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, unpad_sequence
from src.data.dataset import CustomTimeSeriesDataset, EmbeddingDataset, weighted_sampler_dataloader
from src.models.contrastive_loss import TripletLossVaryingLength
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from src.models.model import AutoEncoder, Classifier
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

# Parameters:
ae_name = "autoencoder_128"
name = "ae_encoder_128_freeze_encoder_6"
dataset_train_path = "../../Part 1/ptbdb_train.csv"
dataset_test_path = "../../Part 1/ptbdb_test.csv"
embedding_dim = 128     # dimension of the encoder representations
n_epochs = 100
n_epochs_freeze_encoder = 100   # how many epochs to keep the encoder layers frozen
batch_size = 16
lr = 0.001  # classifier lr
lr_encoder = 0.001  # encoder lr when unfrozen


# Model:
mode = "encoder"    # use Autoencoder model in encoder-only mode
encoder = AutoEncoder(embedding_dim=embedding_dim, mode=mode)
classifier = Classifier(embedding_dim=embedding_dim)
nettype = "CNN"     # used for dataset creation
continue_training = True

# Load pre-trained encoder:
if continue_training:
    encoder.load_state_dict(torch.load(os.path.join('models', f'{ae_name}_best_parameters.pth')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
classifier.to(device)

# Freeze encoder parameters:
if n_epochs_freeze_encoder:
    for parameter in encoder.parameters():
        parameter.requires_grad = False


# Optimizer:
# optimizer = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()),
#                             lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True)
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Learning rate scheduler:
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
scheduler = None
step_after_each_batch = False   # whether to apply scheduler step after each batch or after each epoch


# Dataset:
dataset = CustomTimeSeriesDataset(dataset_train_path, NetType=nettype)
# dataset = EmbeddingDataset("Embeddings/ptbdb_train_embeddings.npz")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = weighted_sampler_dataloader(train_dataset, batch_size=batch_size, repl=True)
val_loader = DataLoader(val_dataset, batch_size=2328, shuffle=False)


# Create parameter config file:
with open(f"Configs/{name}.txt", "w+") as f:
    f.writelines([
        f"Optimizer: {type(optimizer)}(lr={lr})\n",
        f"Scheduler: {type(scheduler) if scheduler is not None else 'None'}" + ("step after each batch\n" if step_after_each_batch else "\n"),
        f"Batch Size: {batch_size}\n",
        f"Epochs: {n_epochs}\n",
        f"Encoder: Frozen for {n_epochs_freeze_encoder} epochs, embedding_dim={embedding_dim}\n",
        f"Classifier: MLP(100, 100)\n"
    ])


# Loss function:
criterion = nn.BCELoss()

# Make initial validation loss infinite:
val_loss = float('inf')

# Set the val loss array with the size of the number of epochs
all_val_loss = [float('inf')] * n_epochs
all_train_loss = [float('inf')] * n_epochs
all_outputs = [float('inf')] * n_epochs
all_labels = [float('inf')] * n_epochs


# To save the best model later:
best_encoder = type(encoder)(embedding_dim=embedding_dim, mode=mode)
best_classifier = type(classifier)(embedding_dim=embedding_dim)


# Training loop:
for epoch in range(n_epochs):
    encoder.train()
    classifier.train()

    if epoch == n_epochs_freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        optimizer.add_param_group({"params": encoder.parameters(), "lr": lr_encoder})

    # step the learning rate scheduler
    if scheduler is not None and not step_after_each_batch:
        scheduler.step()

    # Training loop
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if isinstance(dataset, EmbeddingDataset):
            outputs = classifier(inputs)    # inputs are already embeddings created by the encoder
        else:
            outputs = encoder(inputs)   # inputs are the original timeseries and need to be encoded first
            outputs = classifier(outputs)

        # Loss computation:
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if scheduler is not None and step_after_each_batch:
            scheduler.step()

        if i == 0:
            all_train_loss[epoch] = loss.cpu().detach().numpy()
            # Save the outputs and correct labels of all batches, as numpy arrays
            all_outputs[epoch] = outputs.cpu().detach().numpy()
            all_labels[epoch] = labels.cpu().detach().numpy()

    # Validation loop
    encoder.eval()  # Set model to evaluation mode
    classifier.eval()
    val_loss = torch.tensor(0, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if isinstance(dataset, EmbeddingDataset):
                val_outputs = classifier(inputs)
            else:
                val_outputs = encoder(inputs)
                val_outputs = classifier(val_outputs)

            val_loss = (val_loss * i + criterion(val_outputs, labels).item()) / (i + 1)

    val_loss = val_loss.cpu().detach().numpy()
    # keep track of best model if validation loss is minimum
    if val_loss < min(all_val_loss):
        print(f'Epoch {epoch + 1}/{n_epochs}, Validation Loss: {val_loss}%, new best model!')
        # Save the model parameters,in case we lose them
        best_encoder.load_state_dict(encoder.state_dict())  # Copy the model parameters
        best_classifier.load_state_dict(classifier.state_dict())
        torch.save(best_encoder.state_dict(), os.path.join('models', f'{name}_best_parameters.pth'))
        torch.save(best_classifier.state_dict(), os.path.join('models', f'{name}_best_classifier.pth'))
    else:
        print(f'Epoch {epoch + 1}/{n_epochs}, Validation Loss: {val_loss}%')

    all_val_loss[epoch] = val_loss

torch.save({
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
}, os.path.join('models', f'{name}_scheduler_optimizer_state.pth'))

torch.save(best_encoder.state_dict(), os.path.join('models', f'{name}_best_parameters.pth'))
torch.save(best_classifier.state_dict(), os.path.join('models', f'{name}_best_classifier.pth'))
val_outputs = val_outputs.cpu().detach().numpy()

print('Finished Training')

all_outputs = np.concatenate(all_outputs)
all_labels = np.concatenate(all_labels)

# clear gpu
torch.cuda.empty_cache()

# Create a figure with subfigures
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
dataset_test = CustomTimeSeriesDataset(dataset_test_path, NetType=nettype)
# dataset_test = EmbeddingDataset("Embeddings/ptbdb_test_embeddings.npz")
test_loader = DataLoader(dataset_test, batch_size=2910, shuffle=False)

# Test loop
#send best model to device
best_encoder.to(device)
best_classifier.to(device)
best_encoder.eval()  # Set best model to evaluation mode
best_classifier.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if isinstance(dataset_test, EmbeddingDataset):
            outputs = best_classifier(inputs).detach().cpu().numpy()
        else:
            outputs = best_encoder(inputs)
            outputs = best_classifier(outputs).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy()

# Plot ROC curve
fpr, tpr, _ = roc_curve(labels, outputs)
roc_auc = round(roc_auc_score(labels, outputs), 4)

axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2,label=f'ROC AUC {name}: {roc_auc}')
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('ROC curve')
axs[1, 1].legend(loc="lower right")

# Save all figures
plt.savefig(os.path.join('figures', f'{name}_train_and_test.png'))
plt.close()
