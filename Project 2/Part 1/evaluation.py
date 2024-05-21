#import libraries
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from MyDataset import CustomTimeSeriesDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch
import os
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, f1_score, auc, balanced_accuracy_score
import matplotlib.pyplot as plt
from ModelZoo import simpleCNN, simpleLSTM, BiLSTM, ResCNN, TransformerWithAttentionOutputted

models = [simpleLSTM(), BiLSTM()]
names = ['finalLSTM', "finalBiLSTM"]
nettypes = ['LSTM', 'LSTM', 'CNN', 'CNN', 'Transformer']
for i,model in enumerate(models):
    #Load lstm, biLSTM
    
    name = names[i]
    best_model = model
    nettype = nettypes[i]
    print(f'Loading {name} model')

    #load best model parameters
    best_model.load_state_dict(torch.load(os.path.join('Model_Parameters', f'{name}_best_parameters.pth')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)

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
        fpr, tpr, thresh = roc_curve(labels, outputs)
        roc_auc = round(roc_auc_score(labels, outputs),4)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,label=f'ROC AUC {name}: {roc_auc}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")

        #Save all figures
        plt.savefig(os.path.join('Figures', f'{name}_ROC_curve.png'))
        plt.close()

        #calculate optimal roc auc threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresh[optimal_idx]
        y_optimal = outputs>optimal_threshold

        #compute f1 score, precision recall AUC, balanced accuracy
        precision, recall, _ = precision_recall_curve(labels, outputs)
        pr_auc = auc(recall, precision)

        f1 = f1_score(labels, y_optimal)
        balanced_accuracy =balanced_accuracy_score(labels, y_optimal)

        #make another figure where you just somehow show the f1 score, precision recall AUC, balanced accuracy
        # Create a figure
        fig, ax = plt.subplots()

        # Create a table with the metrics
        columns = ['F1 Score', 'Precision Recall AUC', 'Balanced Accuracy', 'ROC AUC']
        cell_text = [np.round([f1, pr_auc, balanced_accuracy, roc_auc],4)]
        table = plt.table(cellText=cell_text,
                        colLabels=columns,
                        loc='center')

        # Modify the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Hide axes
        ax.axis('off')

        # Set the title
        plt.title(f'{name} F1 Score, Precision Recall AUC, Balanced Accuracy and ROC AUC')

        # Save the figure
        plt.savefig(os.path.join('Figures', f'{name}_F1_PRAUC_BA_ROCAUC.png'))

        # Close the figure      
        plt.close()