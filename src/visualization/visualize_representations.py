import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, unpad_sequence
from src.data.dataset import CustomTimeSeriesDataset, weighted_sampler_dataloader
from src.models.contrastive_loss import TripletLossVaryingLength
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from src.models.model import AutoEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import umap
from itertools import product
from sklearn.decomposition import PCA
from src.models.kl_divergence import KLdivergence

embedding_dim = 128
model = AutoEncoder(embedding_dim=embedding_dim, mode="encoder")
name = "autoencoder_128"
nettype = 'CNN'
batch_size = 128
model.load_state_dict(torch.load(os.path.join('models', f'{name}_best_parameters.pth')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

dataset_mitbih = CustomTimeSeriesDataset('../mitbih_test.csv', NetType=nettype)
dataset_ptb = CustomTimeSeriesDataset('../../Part 1/ptbdb_test.csv', NetType=nettype)
mitbih_loader = DataLoader(dataset_mitbih, batch_size=batch_size, shuffle=False)
ptb_loader = DataLoader(dataset_ptb, batch_size=batch_size, shuffle=False)

all_outputs_mitbih = []
all_labels_mitbih = []
all_inputs_mitbih = []

all_outputs_ptb = []
all_labels_ptb = []
all_inputs_ptb = []

model.eval()  # Set best model to evaluation mode
with torch.no_grad():
    for i, (inputs, labels) in enumerate(tqdm(mitbih_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()

        all_inputs_mitbih += [ts[0] for ts in inputs]
        all_outputs_mitbih += [out for out in outputs]
        all_labels_mitbih += [label[0] for label in labels]

    for i, (inputs, labels) in enumerate(tqdm(ptb_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()

        all_inputs_ptb += [ts[0] for ts in inputs]
        all_outputs_ptb += [out for out in outputs]
        all_labels_ptb += [label[0] for label in labels]

all_inputs_mitbih = np.asarray(all_inputs_mitbih)
all_outputs_mitbih = np.asarray(all_outputs_mitbih)
labels_mitbih = np.asarray(all_labels_mitbih)

all_inputs_ptb = np.asarray(all_inputs_ptb)
all_outputs_ptb = np.asarray(all_outputs_ptb)
labels_ptb = np.asarray(all_labels_ptb)

# # Save embedding files
# np.savez("mitbih_test_embeddings.npz", embeddings=all_outputs_mitbih, labels=all_labels_mitbih)
# np.savez("ptb_test_embeddings.npz", embeddings=all_inputs_ptb, labels=all_inputs_ptb)

# Apply PCA
pca = PCA(n_components=50)  # Reduce to the first 50 principal components
pca_mitbih = pca.fit_transform(all_outputs_mitbih)
pca_ptb = pca.fit_transform(all_outputs_ptb)

# # Compute KL divergences between mitbih embeddings with different labels:
# kl_matrix = np.zeros((len(np.unique(labels_mitbih)), len(np.unique(labels_mitbih))))
# for i, label_1 in enumerate(np.unique(labels_mitbih)):
#     for j, label_2 in enumerate(np.unique(labels_mitbih)):
#         outputs_1 = pca_mitbih[all_labels_mitbih == label_1]
#         outputs_2 = pca_mitbih[all_labels_mitbih == label_2]
#         kl_matrix[i, j] = KLdivergence(outputs_1, outputs_2)
# print((kl_matrix + kl_matrix.T) / 2)
#
# # Compute JS divergence between mitbih embeddings and ptb embeddings:
# kl_datasets = (KLdivergence(pca_mitbih, pca_ptb) + KLdivergence(pca_ptb, pca_mitbih)) / 2
# print(kl_datasets)

# # Plot autoencoder reconstructions:
# fig, ax = plt.subplots(2, 5, figsize=(15, 5))
# for i in range(5):
#     ax.flatten()[i].plot(all_inputs_mitbih[i], color="green")
#     ax.flatten()[5+i].plot(all_outputs_mitbih[i, 0, :], color="red")
# ax[0, 0].set_ylabel("Original")
# ax[1, 0].set_ylabel("Reconstruction")
# plt.show()

# Create subplots
min_dist = 0.05
n_neighbors = 50

# Initialize UMAP with current parameters
umap_model = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors)

# Fit UMAP to the reduced embeddings
umap_result_ptb = umap_model.fit_transform(pca_ptb)
umap_result_mitbih = umap_model.fit_transform(pca_mitbih)

# Plot the embeddings
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
scatter = ax[0].scatter(umap_result_ptb[:, 0], umap_result_ptb[:, 1], c=labels_ptb, s=5, cmap=plt.cm.plasma)
ax[0].legend(*scatter.legend_elements(), loc='lower right')

ax[0].set_title("PTB Representations")
ax[0].set_xlabel(f'min_dist={min_dist}, n_neighbors={n_neighbors}')

scatter = ax[1].scatter(umap_result_mitbih[:, 0], umap_result_mitbih[:, 1], c=labels_mitbih, s=5, cmap=plt.cm.plasma)
ax[1].legend(*scatter.legend_elements(), loc='lower right')

ax[1].set_title("MIT-BIH Representations")
ax[1].set_xlabel(f'min_dist={min_dist}, n_neighbors={n_neighbors}')

# Adjust layout
plt.tight_layout()
plt.show()

# Compute KL divergences between mitbih embeddings with different labels:
kl_matrix = np.zeros((len(np.unique(labels_mitbih)), len(np.unique(labels_mitbih))))
for i, label_1 in enumerate(np.unique(labels_mitbih)):
    for j, label_2 in enumerate(np.unique(labels_mitbih)):
        outputs_1 = umap_result_mitbih[all_labels_mitbih == label_1]
        outputs_2 = umap_result_mitbih[all_labels_mitbih == label_2]
        kl_matrix[i, j] = KLdivergence(outputs_1, outputs_2)
print((kl_matrix + kl_matrix.T) / 2)

# Compute JS divergence between mitbih embeddings and ptb embeddings:
kl_datasets = (KLdivergence(umap_result_mitbih, umap_result_ptb) + KLdivergence(umap_result_ptb, umap_result_mitbih)) / 2
print(kl_datasets)
