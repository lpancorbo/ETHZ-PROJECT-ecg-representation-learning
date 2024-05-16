import torch
from torch.utils.data import DataLoader
from dataset_q1 import CustomTimeSeriesDataset, weighted_sampler_dataloader
from model_q4 import get_feature_extractor
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load PTB
nettype = 'CNN'
ptb_train_dataset = CustomTimeSeriesDataset('ptbdb_train.csv', NetType=nettype)
ptb_train_loader = DataLoader(ptb_train_dataset, batch_size=len(ptb_train_dataset), shuffle=False)
ptb_test_dataset = CustomTimeSeriesDataset('ptbdb_test.csv', NetType=nettype)
ptb_test_loader = DataLoader(ptb_test_dataset, batch_size=len(ptb_test_dataset), shuffle=False)

# Get feature extractor model
feature_extractor = get_feature_extractor()

# Freeze all parameters
for param in feature_extractor.parameters():
    param.requires_grad = False

feature_extractor.to(device)

feature_extractor.eval()
features_train = []
labels_train = []
# Extract features of PTB train and test
for inputs, labels in ptb_train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    features_train.append(feature_extractor(inputs).numpy())
    labels_train.append(labels.numpy())
X_train = np.concatenate(features_train, axis=0)
y_train = np.concatenate(labels_train, axis=0).squeeze()

features_test = []
labels_test = []
for inputs, labels in ptb_test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    features_test.append(feature_extractor(inputs).numpy())
    labels_test.append(labels.numpy())
X_test = np.concatenate(features_test, axis=0)
y_test = np.concatenate(labels_test, axis=0).squeeze()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=13)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:,1]

roc_auc_rf = metrics.roc_auc_score(y_test, y_pred)
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_pred)

#plot ROC curves
plt.figure()
plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = %0.4f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Finetuning with Classic ML')
plt.legend(loc="lower right")
plt.show()

plt.savefig('./Figures/ROC_finetuning_classicML.png')
plt.close()

