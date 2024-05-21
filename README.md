# Machine Learning For Healthcare - Project 2
## Part 1: Requires ptb datasets in .csv format in the part 1 subfolder
- ExploratoryDataAnalysisPTB.ipynb: Contains the visualization of label distribution and dataset size for the PTB database
- ClassicMachineLearning_final.ipynb: Contains the training and plotting of performance of classic machine learning models, without and with manually extracted features
- MyDataset.py: Contains the definition of the Dataset object, that receives the .csv file and creates the samples and labels
- train+evaluation.py: Contains the training loop and plots related to training behaviour, on train and val sets, and ROC/ROC AUC, on test test. Plots saved in a .png file. Here, you should define the model to be used (see ModelZoo.py file), and the training parameters. Optimizer used for each model was commented.
- ModelZoo.py: Defines the classes of all deep learning architectures that were trained. Model hyperparameters used are the ones specified inside these classes.
- evaluation.py: Contains the loop over the desired models that allows the computation of different performance metrics in test set, which are saved in a .png figure with a table.
- attention_maps_transformer.py : Contains the code that evaluates test set and computes self-attention weights received by each time point from other points, for different examples from positive and negative classes.

## Part 2:
### Transfer Learning:
- model_transfer_supervided.py: Contains the ResCNN model modified for 5 classes. It also contains the model for transfer learning and the encoder. Once the model is trained on the MITBIH dataset, it needs to be saved in the same directory as this file under the name "./ResCNN_mitbih_best_parameters.pth".
- model_transfer_supervised_training.py: Trains the model on MITBIH and saves the best parameters as "./ResCNN_mitbih_best_parameters.pth".
- VisualisingLearnedRepresentations.ipynb: Extracts the embeddings with the get_feature_extractor function from model_transfer_supervised.py. This notebook performs dimensionality reduction and computes KL divergence
- FinetuningStrategies_ClassicML.py: loads the encoder from model_transfer_supervided.py, extracts the embeddings for PTB and feeds them into a Random Forest.
- FinetuningStrategies_MLP.py: Loads the transfer_model from model_transfer_supervided.py and follows one of the three strategies defines by the TASK variable (A, B or C). Best models were saved as "transfer_model_A_16batch_100epochs_best_parameters.pth", "transfer_model_B_16batch_100epochs_best_parameters.pth", "transfer_model_C_16batch_100epochs_best_parameters.pth".
- FinetuningStrategies_MLP_metrics.py: Compute metrics and joint ROC curve for the 3 strategies

### Representation Learning:
- ae_training.py: train a ResCNN autoencoder on the MIT-BIH dataset. Trained model is saved in the Model_Parameters folder. Paths to train and test dataset need to be specified at the top of the script. Test plots are saved in the Figures folder.
- ae_finetuning.py: fine-tune the pre-trained autoencoder with additional output layers on the PTB dataset. Trained model is saved in the Model_Parameters folder (separate files for encoder and classifier MLP). Paths to train and test dataset and model pth file need to be specified at the top of the script.
- evaluate.py: Evaluate fine-tuned model on the PTB test set and compute performance scores (balanced accuracy, F1, ROC-AUC, PR-AUC, ROC curve). Paths to test dataset and model pth file need to be specified at the top of the script.
- visualize_representations.py: Create UMAP visualizations from encoder embeddings of MIT-BIH or PTB test dataset and compute KL-divergences between embedding distributions of different labels/between the two datasets Paths to test dataset and model pth file need to be specified at the top of the script.
- ClassicMachineLearning.ipynb: train classic ML methods such as random forest on encoder embeddings. Paths to train and test embedding npz files has to be specified at the top of the notebook (embedding files are created in visualize_representations.py and stored in the Embeddings folder). 
