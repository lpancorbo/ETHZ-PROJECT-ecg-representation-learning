#import libraries
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from MyDataset import CustomTimeSeriesDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import os
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt
from ModelZoo import simpleCNN, simpleLSTM, BiLSTM, ResCNN, TransformerWithAttentionOutputted

models = [simpleLSTM(), simpleCNN(), ResCNN(), TransformerWithAttentionOutputted()]
names = ['simpleLSTM','simpleCNN', 'ResCNN', 'TransformerWithAttentionOutputted']
for i,model in enumerate(models):
    #Load lstm, biLSTM, TransformerWithAttentionOutputted, ResCNN, simpleCNN
    print(names[i])
    name = names[i]
    best_model = model

    #load best model parameters
    best_model.load_state_dict(torch.load(os.path.join('Model_Parameters', f'{name}_best_parameters.pth')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)

    #EVALUATION
    #Prepare test dataset
    dataset_test = CustomTimeSeriesDataset('ptbdb_test.csv', NetType='CNN')
    test_loader = DataLoader(dataset_test, batch_size=2910, shuffle=False)

    # Test loop
    #send best model to device
    best_model.to(device)
    best_model.eval()  # Set best model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass and prepare for plot
            if name == "TransformerWithAttentionOutputted":
                outputs = F.sigmoid(best_model(inputs)[0]).detach().cpu().numpy()
            else:
                outputs = F.sigmoid(best_model(inputs)).detach().cpu().numpy()
            labels=labels.detach().cpu().numpy()


    #Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, outputs)

    plt.figure()
    roc_auc = round(roc_auc_score(labels, outputs),4)
    plt.plot(fpr, tpr, color='darkorange', lw=2,label=f'ROC AUC {name}: {roc_auc}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    #Save all figures
    plt.savefig(os.path.join('Figures', f'{name}_ROC_curve.png'))
    plt.close()

    #compute f1 score, precision recall AUC, balanced accuracy
    precision, recall, _ = precision_recall_curve(labels, outputs)
    #calculate the threshold for the outputs that gives best f1 score
    f1 = f1_score(labels, outputs.round())
    pr_auc = auc(recall, precision)
    balanced_accuracy = 0.5*(tpr+(1-fpr))

    #make another figure where you just somehow show the f1 score, precision recall AUC, balanced accuracy
    plt.figure()
    plt.bar(['F1 Score', 'Precision Recall AUC', 'Balanced Accuracy' ], [f1, pr_auc,balanced_accuracy.max()])
    plt.title(f'{name} F1 Score, Precision Recall AUC and Balanced Accuracy')
    plt.savefig(os.path.join('Figures', f'{name}_F1_PR_AUC_BA.png'))

