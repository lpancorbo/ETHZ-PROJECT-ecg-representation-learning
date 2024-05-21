# Machine Learning For Healthcare - Project 2
## Part 1:

## Part 2:
### Transfer Learning:

### Representation Learning:
- ae_training.py: train a ResCNN autoencoder on the MIT-BIH dataset. Trained model is saved in the Model_Parameters folder. Paths to train and test dataset need to be specified at the top of the script. Test plots are saved in the Figures folder.
- ae_finetuning.py: fine-tune the pre-trained autoencoder with additional output layers on the PTB dataset. Trained model is saved in the Model_Parameters folder (separate files for encoder and classifier MLP). Paths to train and test dataset and model pth file need to be specified at the top of the script.
- evaluate.py: Evaluate fine-tuned model on the PTB test set and compute performance scores (balanced accuracy, F1, ROC-AUC, PR-AUC, ROC curve). Paths to test dataset and model pth file need to be specified at the top of the script.
- visualize_representations.py: Create UMAP visualizations from encoder embeddings of MIT-BIH or PTB test dataset and compute KL-divergences between embedding distributions of different labels/between the two datasets Paths to test dataset and model pth file need to be specified at the top of the script.
- ClassicMachineLearning.ipynb: train classic ML methods such as random forest on encoder embeddings. Paths to train and test embedding npz files has to be specified at the top of the notebook (embedding files are created in visualize_representations.py and stored in the Embeddings folder). 