import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, unpad_sequence
from dataset_q1 import CustomTimeSeriesDataset, weighted_sampler_dataloader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from model_transfer_supervised import ResCNN
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, balanced_accuracy_score, precision_score
from sklearn.metrics import f1_score as f1_scorer

#Define model to be trained
name = "ResCNNbatch16_mitbih"
model = ResCNN()
nettype = 'CNN'
continue_training = False
batch_size = 16

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True)

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
dataset = CustomTimeSeriesDataset('mitbih_train.csv', NetType=nettype)

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
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)

#Define loss function and optimizer as BCE, as output is NOW a probability
criterion = nn.CrossEntropyLoss()

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

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze(1))
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if i==0:
            all_train_loss[epoch]=loss.cpu().detach().numpy()
            #Save the outputs and correct labels of all batches, as numpy arrays
            all_outputs[epoch]=outputs.cpu().detach().numpy()
            all_labels[epoch]=labels.squeeze(1).cpu().detach().numpy()

    # Validation loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels.squeeze(1))

    val_loss = val_loss.cpu().detach().numpy()
    #keep track of best model if validation loss is minimum
    if val_loss < min(all_val_loss):
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%, new best model!')
        # Save the model parameters,in case we lose them
        best_model.load_state_dict(model.state_dict())  # Copy the model parameters
    else:
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss}%')

    all_val_loss[epoch]=val_loss

# torch.save({
#     'optimizer_state_dict': optimizer.state_dict(),
#     'scheduler_state_dict' : scheduler.state_dict(),
#     }, os.path.join('Model_Parameters', f'{name}_scheduler_optimizer_state.pth'))

# torch.save(best_model.state_dict(), os.path.join('Model_Parameters', f'{name}_best_parameters.pth'))

val_outputs = val_outputs.cpu().detach().numpy()

print('Finished Training')

all_outputs = np.concatenate(all_outputs)
all_labels = np.concatenate(all_labels)

#clear gpu
torch.cuda.empty_cache()

#Create a figure with subfigures
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

# Plot outputs for each class separately
axs[0, 0].plot(all_outputs[all_labels==0, 0], 'o', label='Class 0', color='blue')
axs[0, 0].plot(all_outputs[all_labels==1, 1], 'o', label='Class 1', color='orange')
axs[0, 0].plot(all_outputs[all_labels==2, 2], 'o', label='Class 2', color='green')
axs[0, 0].plot(all_outputs[all_labels==3, 3], 'o', label='Class 3', color='red')
axs[0, 0].plot(all_outputs[all_labels==4, 4], 'o', label='Class 4', color='purple')
axs[0, 0].legend()
axs[0, 0].set_xlabel('Batch')
axs[0, 0].set_ylabel('Model Output')
axs[0, 0].set_title('Model Output for each Class')

#Plot histogram of validation outputs
axs[0, 1].hist(val_outputs[:,0], bins=100, label='Class 0')
axs[0, 1].hist(val_outputs[:,1], bins=100, label='Class 1')
axs[0, 1].hist(val_outputs[:,2], bins=100, label='Class 2')
axs[0, 1].hist(val_outputs[:,3], bins=100, label='Class 3')
axs[0, 1].hist(val_outputs[:,4], bins=100, label='Class 4')
axs[0, 1].legend()
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].set_title('Histogram of Model Outputs for each Class')

# Add the subfigure to the big figure
axs[1, 0].plot(all_val_loss)
axs[1, 0].plot(all_train_loss)
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].set_title('Train and Validation Loss as function of epoch')
axs[1, 0].legend(['Validation Loss', 'Train Loss'])

#EVALUATION
#Prepare test dataset
dataset_test = CustomTimeSeriesDataset('mitbih_test.csv', NetType=nettype)
test_loader = DataLoader(dataset_test, batch_size=len(dataset), shuffle=False)

# Test loop
#send best model to device
best_model.to(device)
best_model.eval()  # Set best model to evaluation mode

true_labels = []
predicted_labels = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass and prepare for plot
        outputs = best_model(inputs).detach().cpu().numpy()
        labels=labels.detach().cpu().numpy()

        # Append true labels and predicted labels
        true_labels.extend(labels)
        predicted_labels.extend(np.argmax(outputs, axis=1))  # Predicted labels are the class with highest probability

# Plot ROC curves for each class
for i in range(5):
    fpr, tpr, _ = roc_curve(labels == i, outputs[:, i])
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('ROC curves for each class')

# Add legend
axs[1, 1].legend(loc="lower right")

# Show grid
axs[1, 1].grid(True)

#Save all figures
# plt.savefig(os.path.join('Figures', f'{name}_train_and_test.png'))
plt.close()

# Plot ROC and PR Curves
fig, axs = plt.subplots(1, 2, figsize=(18, 9))

# Plot ROC curves
for i in range(5):
    fpr, tpr, _ = roc_curve(labels == i, outputs[:, i])
    roc_auc = auc(fpr, tpr)
    axs[0].plot(fpr, tpr, lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc:.4f})')

axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC curves for each class')
axs[0].legend(loc="lower right",fontsize="large")
axs[0].grid(True)

# Plot PR curves
for i in range(5):
    precision, recall, _ = precision_recall_curve(labels == i, outputs[:, i])
    pr_auc = auc(recall, precision)
    axs[1].plot(recall, precision, lw=2, label=f'PR curve (class {i}) (AUC = {pr_auc:.4f})')

axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('PR curves for each class')
axs[1].legend(loc="lower left", fontsize="large")
axs[1].grid(True)
#tight axes
plt.tight_layout()
# plt.savefig(os.path.join('Figures', f'{name}_train_and_test_v2.png'))
plt.close()

accuracy = accuracy_score(true_labels, predicted_labels)
balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
f1 = f1_scorer(true_labels, predicted_labels, average='weighted')
precision = precision_score(true_labels, predicted_labels, average='weighted')

print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")