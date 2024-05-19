import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, unpad_sequence
from dataset import CustomTimeSeriesDataset, weighted_sampler_dataloader
from contrastive_loss import TripletLossVaryingLength
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from model import AutoEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import umap
from itertools import product
from sklearn.decomposition import PCA


embedding_dim = 128
model = AutoEncoder(embedding_dim=embedding_dim)
name = "autoencoder_128"
nettype = 'CNN'
batch_size = 128
model.load_state_dict(torch.load(os.path.join('Model_Parameters', f'{name}_best_parameters.pth')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

dataset_test = CustomTimeSeriesDataset('../mitbih_test.csv', NetType=nettype)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


all_outputs = []
all_labels = []
all_inputs = []

model.eval()  # Set best model to evaluation mode
with torch.no_grad():
    for i, (inputs, labels) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()

        all_inputs += [ts[0] for ts in inputs]
        all_outputs += [out for out in outputs]
        all_labels += [label[0] for label in labels]

all_inputs = np.asarray(all_inputs)
all_outputs = np.asarray(all_outputs)
labels = np.asarray(all_labels)

# fig, ax = plt.subplots(2, 5, figsize=(15, 5))
# for i in range(5):
#     ax.flatten()[i].plot(all_inputs[i], color="green")
#     ax.flatten()[5+i].plot(all_outputs[i, 0, :], color="red")
# ax[0, 0].set_ylabel("Original")
# ax[1, 0].set_ylabel("Reconstruction")
# plt.show()
# exit()

#all_outputs = np.nan_to_num(all_outputs)

min_dist_values = [0.001, 0.01, 0.1]
n_neighbors_values = [5, 15, 50]

# Apply PCA
pca = PCA(n_components=50)  # Reduce to the first 50 principal components
embeddings_pca = pca.fit_transform(all_inputs)

# Create subplots
fig, axs = plt.subplots(len(min_dist_values), len(n_neighbors_values), figsize=(15, 10))

# Iterate over combinations of parameters
for i, min_dist in enumerate(min_dist_values):
    for j, n_neighbors in enumerate(n_neighbors_values):
        print(f"min_dist: {min_dist}, n_neighbors: {n_neighbors}")
        # Initialize UMAP with current parameters
        umap_model = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors)

        # Fit UMAP to the reduced embeddings
        umap_result = umap_model.fit_transform(embeddings_pca)

        # Plot the embeddings
        ax = axs[i, j]
        for label in np.unique(labels):
            indices = np.where(labels == label)
            ax.scatter(umap_result[indices, 0], umap_result[indices, 1], label=label, s=5)
        ax.set_title(f'min_dist={min_dist}, n_neighbors={n_neighbors}')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()
