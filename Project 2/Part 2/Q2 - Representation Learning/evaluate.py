# import libraries
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, unpad_sequence
from dataset import CustomTimeSeriesDataset, EmbeddingDataset, weighted_sampler_dataloader
from contrastive_loss import TripletLossVaryingLength
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from model import AutoEncoder, Classifier
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, f1_score, precision_recall_curve, auc
from tqdm import tqdm

# Parameters:
names = ["ae_encoder_128_freeze_encoder_6",
         "ae_encoder_128_train_all_4",
         "ae_encoder_128_two_stage_4"]
plot_labels = ["Strategy A", "Strategy B", "Strategy C"]
dataset_test_path = "../../Part 1/ptbdb_test.csv"
embedding_dim = 128

for ind, name in enumerate(names):
    # Model:
    mode = "encoder"    # use Autoencoder model in encoder-only mode
    encoder = AutoEncoder(embedding_dim=embedding_dim, mode=mode)
    classifier = Classifier(embedding_dim=embedding_dim)
    nettype = "CNN"     # used for dataset creation
    encoder.load_state_dict(torch.load(os.path.join('Model_Parameters', f'{name}_best_parameters.pth')))
    classifier.load_state_dict(torch.load(os.path.join('Model_Parameters', f'{name}_best_classifier.pth')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    classifier.to(device)

    # Dataset:
    dataset_test = CustomTimeSeriesDataset(dataset_test_path, NetType=nettype)
    # dataset_test = EmbeddingDataset("Embeddings/ptbdb_test_embeddings.npz")
    test_loader = DataLoader(dataset_test, batch_size=2910, shuffle=False)

    # Test loop:
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if isinstance(dataset_test, EmbeddingDataset):
                outputs = classifier(inputs).detach().cpu().numpy()
            else:
                outputs = encoder(inputs)
                outputs = classifier(outputs).detach().cpu().numpy()

            labels = labels.detach().cpu().numpy()

    # Compute scores:
    balanced_acc = balanced_accuracy_score(labels, np.rint(outputs))
    f1 = f1_score(labels, np.rint(outputs), average='weighted')
    pr, recall, thresh = precision_recall_curve(labels, outputs)
    pr_auc_rf = auc(recall, pr)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, outputs)
    roc_auc = round(roc_auc_score(labels, outputs), 4)

    print(name)
    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"PR-AUC: {pr_auc_rf}")
    print()

    plt.plot(fpr, tpr, lw=2, label=f'ROC AUC {plot_labels[ind]}: {roc_auc}')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")

# Save all figures
plt.savefig(os.path.join('Figures', f'finetuning_roc_curves_4.png'))
plt.close()
