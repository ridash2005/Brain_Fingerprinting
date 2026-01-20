import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging  # Import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.conv_ae import ConvAutoencoder
from src.models.sparse_dictionary_learning import *
from src.utils.config_parser import *
from src.utils.matrix_ops import *

# Argument parsing for selecting the data and task
parser = argparse.ArgumentParser(description='Functional Connectome Task Selection')
parser.add_argument('-data', type=str, required=True, choices=['rest', 'motor', 'wm', 'emotion'], help="Data on which the model was trained: rest, motor, wm, or emotion")
parser.add_argument('-task', type=str, required=True, choices=['rest', 'motor', 'wm', 'emotion'], help="Task to analyze: rest, motor, wm, or emotion")
args = parser.parse_args()

# Basic parameters
basic_parameters = parse_basic_params()
RUN_ID = get_run_timestamp()
LOG_DIR = ensure_dir(os.path.join("logs", "runs", f"{RUN_ID}_optimize_{args.data}_to_{args.task}"))
HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']

# Set up logging
log_file = os.path.join(LOG_DIR, 'accuracy_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', filemode='a')
print(f"Logging optimization results to {log_file}")

# Load the trained model based on the training data
model = ConvAutoencoder()
model.load_state_dict(torch.load(f'./src/models/trained/conv_ae_{args.data}_best_model.pth'))
model.eval()

# Select the appropriate functional connectome (FC) data based on the testing task
fc_task = np.load(f'FC_DATA/fc_{args.task}.npy')
fc_task = fc_task[:, np.newaxis, :, :]
fc_task_tensor = torch.tensor(fc_task, dtype=torch.float32)

# Perform reconstruction
with torch.no_grad():
    fc_task_reconstr = model(fc_task_tensor)

# Calculate the residuals
fc_task_residual = fc_task_tensor.squeeze(1) - fc_task_reconstr.squeeze(1)

# Load train data for comparison
fc_train = np.load(f'FC_DATA/fc_{args.data}.npy')
fc_train_tensor = torch.tensor(fc_train, dtype=torch.float32)


# Correlation analysis function
def calculate_correlation(fc_task_data, fc_train_data):
    return np.corrcoef(fc_task_data.view(N_SUBJECTS, -1).numpy(), fc_train_data.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]

# Initialize arrays for storing accuracies
accuracy_matrix = np.zeros((14, 14))  # For K=2 to 15, and L ranging from K to 15

# Loop over K and L, apply SDL, and compute accuracies
for K in range(15, 16):
    for L in range(2, K + 1):
        # Apply SDL
        print(f'K= {K}, L = {L}')
        Y = np.zeros((int(N_PARCELS * (N_PARCELS - 1) / 2), N_SUBJECTS))
        for i in range(N_SUBJECTS):
            Y[:, i] = fc_task_residual[i][np.tril_indices(fc_task_residual[i].shape[0], k=-1)]

        D, X = k_svd(Y, K, L)
        sdl_retr = np.dot(D, X).transpose()
        DX = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

        for i in range(N_SUBJECTS):
            DX[i, :, :] = reconstruct_symmetric_matrix(sdl_retr[i, :])

        fc_task_refined = fc_task_residual - DX

        # Correlation analysis between task and rest FC after SDL
        corr_task_rest_after_sdl = calculate_correlation(fc_task_refined, fc_train_tensor)

        # Calculate accuracy after SDL
        accuracy_after_sdl = calculate_accuracy(corr_task_rest_after_sdl) * 100  # Convert to percentage
        print(accuracy_after_sdl)

        # Log the accuracy in the specified format
        logging.info(f'K= {K}, L = {L} {accuracy_after_sdl}')

        # Store the accuracy in the matrix
        accuracy_matrix[K - 2, L - 2] = accuracy_after_sdl
