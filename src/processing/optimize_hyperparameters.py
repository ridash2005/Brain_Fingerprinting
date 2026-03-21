import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import logging
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
parser.add_argument('-n_folds', type=int, default=5, help="Number of CV folds (default: 5)")
args = parser.parse_args()

# Basic parameters
basic_parameters = parse_basic_params()
RUN_ID = get_run_timestamp()
LOG_DIR = ensure_dir(os.path.join("logs", "runs", f"{RUN_ID}_optimize_{args.data}_to_{args.task}"))
HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
log_file = os.path.join(LOG_DIR, 'accuracy_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', filemode='a')
print(f"Logging optimization results to {log_file}")

# Load data
fc_task = np.load(f'FC_DATA/fc_{args.task}.npy')
fc_train_data = np.load(f'FC_DATA/fc_{args.data}.npy')

# ==========================================
# NESTED CROSS-VALIDATION (Awareness Fix)
# 
# The hyperparameter search uses K-Fold CV:
# - ConvAE trained on TRAIN subjects per fold
# - SDL applied inductively (train dict → test coding)
# - Accuracy evaluated on HELD-OUT subjects only
# This prevents identifiability leakage where
# the model memorizes subjects across scans.
# ==========================================
n_folds = args.n_folds
indices = np.random.RandomState(42).permutation(N_SUBJECTS)
fold_size = N_SUBJECTS // n_folds

logging.info(f"# {n_folds}-Fold Cross-Validated Hyperparameter Optimization (Awareness Fix)")
logging.info(f"# Data: {args.data}, Task: {args.task}, Subjects: {N_SUBJECTS}")

# Initialize accuracy matrix for heatmap
accuracy_matrix = np.zeros((20, 20))

print(f"Running {n_folds}-Fold CV hyperparameter optimization (awareness.txt fix)")
print(f"Data: {args.data}, Task: {args.task}, Subjects: {N_SUBJECTS}")

# Loop over K and L
for K in range(2, 17, 2):
    for L in range(2, K + 1, 2):
        print(f'K= {K}, L = {L}')
        
        fold_accuracies = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = N_SUBJECTS if fold == n_folds - 1 else (fold + 1) * fold_size
            test_idx = indices[start_idx:end_idx]
            train_idx = np.setdiff1d(indices, test_idx)
            
            n_train = len(train_idx)
            n_test = len(test_idx)
            
            # --- TRAIN ConvAE on train subjects only ---
            train_rest = fc_train_data[train_idx]
            train_tensor = torch.tensor(train_rest[:, np.newaxis, :, :], dtype=torch.float32)
            
            model = ConvAutoencoder().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=16, shuffle=True)
            
            model.train()
            for epoch in range(20):
                for batch, _ in loader:
                    batch = batch.to(DEVICE)
                    optimizer.zero_grad()
                    loss = nn.MSELoss()(model(batch), batch)
                    loss.backward()
                    optimizer.step()
            
            # --- Compute residuals on TEST subjects ---
            model.eval()
            test_task = fc_task[test_idx]
            test_task_tensor = torch.tensor(test_task[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                test_reconstr = model(test_task_tensor).cpu().squeeze(1).numpy()
            test_residual = test_task - test_reconstr
            
            # --- SDL: Learn dictionary from TRAIN residuals, apply to TEST ---
            train_task = fc_task[train_idx]
            train_task_tensor = torch.tensor(train_task[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                train_reconstr = model(train_task_tensor).cpu().squeeze(1).numpy()
            train_residual = train_task - train_reconstr
            
            n_tri = int(N_PARCELS * (N_PARCELS - 1) / 2)
            tril_idx = np.tril_indices(N_PARCELS, k=-1)
            
            Y_train = np.zeros((n_tri, n_train))
            for i in range(n_train):
                Y_train[:, i] = train_residual[i][tril_idx]
            
            D_train, _ = k_svd(Y_train, K, L, n_iter=10, verbose=False, random_state=42)
            
            Y_test = np.zeros((n_tri, n_test))
            for i in range(n_test):
                Y_test[:, i] = test_residual[i][tril_idx]
            
            X_test = omp_sparse_coding(Y_test, D_train, L)
            sdl_retr = np.dot(D_train, X_test).T
            
            DX_test = np.zeros((n_test, N_PARCELS, N_PARCELS))
            for i in range(n_test):
                DX_test[i] = reconstruct_symmetric_matrix(sdl_retr[i])
            
            test_refined = test_residual - DX_test
            
            # --- Accuracy on HELD-OUT subjects ---
            test_rest = fc_train_data[test_idx]
            corr = np.corrcoef(
                test_refined.reshape(n_test, -1),
                test_rest.reshape(n_test, -1)
            )[:n_test, n_test:]
            fold_accuracies.append(calculate_accuracy(corr))
        
        # Average accuracy across folds
        mean_acc = np.mean(fold_accuracies) * 100
        std_acc = np.std(fold_accuracies) * 100
        print(f'  CV Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')
        
        logging.info(f'K= {K}, L = {L} {mean_acc:.2f} ± {std_acc:.2f}')
        accuracy_matrix[K - 2, L - 2] = mean_acc

print(f"\nOptimization complete. Results saved to {log_file}")
