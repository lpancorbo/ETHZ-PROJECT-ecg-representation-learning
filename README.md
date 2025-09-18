# ECG Representation Learning Project

This repository contains our implementation for the ML4H project. It is organized in two main parts, each addressing different aspects of supervised, transfer, and representation learning on time series data from ECG datasets.

For more context, please refer to the [project report](docs).

---

## Part 1: Supervised Learning on Time Series

Focus: Classification of ECG time series from the PTB Diagnostic ECG Database as healthy or abnormal.

### Exploratory Data Analysis
- **File:** [ExploratoryDataAnalysisPTB.ipynb](src/visualization/ExploratoryDataAnalysisPTB.ipynb)
- Visualize example time series and label distribution.

### Classic Machine Learning Methods
- **Files:** [ClassicMachineLearning_final.ipynb](src/classic_ml/ClassicMachineLearning_final.ipynb)
- Train classic ML classifiers on raw time series.
- Engineer additional features and retrain models.

### Deep Learning Methods
- **Files:** [model_zoo.py](src/models/model_zoo.py), [train_evaluation.py](src/training/train_evaluation.py)

- **Recurrent Neural Networks**: LSTM and bidirectional LSTM models.
- **Convolutional Neural Networks**: vanilla CNN and CNN with residual blocks.
- **Attention and Transformers**: transformer model. Visualize attention maps with [attention_maps_transformer.ipynb](src/visualization/attention_maps_transformer.ipynb)

---

## Part 2: Transfer and Representation Learning

Focus: Leveraging the MIT-BIH database to improve learning on the PTB dataset using transfer and representation learning.

### Supervised Model for Transfer
- **Files:** [model_transfer_supervised.py](src/models/model_transfer_supervised.py), [model_transfer_supervised_training.py](src/training/model_transfer_supervised_training.py)
- Train resCNN on MIT-BIH for arrhythmia classification.
- Use the trained model as a pre-trained encoder for transfer learning.

### Representation Learning Model
- **Files:** [ae_training.py](src/training/ae_training.py), [ae_finetuning.py](src/training/ae_finetuning.py), [evaluate.py](src/evaluation/evaluate.py)
- Pretrain an encoder using unsupervised or self-supervised objectives (autoencoder).
- Evaluate learned representations using classic ML methods.

### Visualising Learned Representations
- **Files:** [visualize_representations.py](src/visualization/visualize_representations.py), [VisualisingLearnedRepresentations.ipynb](src/visualization/VisualisingLearnedRepresentations.ipynb)
- Use encoders to obtain representations for MIT-BIH and PTB datasets.
- Visualize with UMAP and provide quantitative metrics.

### Finetuning Strategies
- **Files:** [FinetuningStrategies_ClassicML.py](src/classic_ml/FinetuningStrategies_ClassicML.py), [FinetuningStrategies_MLP.py](src/classic_ml/FinetuningStrategies_MLP.py), [FinetuningStrategies_MLP_metrics.ipynb](src/classic_ml/FinetuningStrategies_MLP_metrics.ipynb)
- Apply different finetuning strategies to pre-trained encoders:
  - Classic ML on encoder representations
  - MLP output layers: freeze encoder, train all, or staged training

---

## Data, Figures, and Models
- **data:** Datasets must be placed in `data/`. Data is downloaded from [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/) and [MIT-BIH Arrhythmia Database](https://physionet.org/physiobank/database/mitdb/) and processed as follows:
    - Samples are cropped, downsampled, and padded with zeros to a fixed length of 188.
    - Each dataset is split into training and test CSV files, where each row is an example and the last column is the class label.
- **figures:** All plots and results are in `figures/`
- **models:** All saved model weights are in `models/`

---

## How to Run

Create a Conda environment and install requirements:
```bash
conda create -n ecg-dl python=3.11
conda activate ecg-dl
pip install -r requirements.txt
```

---
