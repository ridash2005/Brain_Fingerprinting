import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.conv_ae import ConvAutoencoder
from src.models.conv_ae_fp import ConvAutoencoderFP
from src.models.sparse_dictionary_learning import *
from src.utils.config_parser import *
from src.utils.matrix_ops import * 

# Argument parsing for task and network selection
parser = argparse.ArgumentParser(description='Functional Connectome Task and Network Selection')
parser.add_argument('-task', type=str, required=True, choices=['motor', 'wm', 'emotion'], help="Task to analyze: motor, wm, or emotion")
parser.add_argument('-network', type=str, required=True, help="Brain network/region to analyze (e.g., Frontoparietal, DMN, etc.)")
parser.add_argument('-n_folds', type=int, default=5, help="Number of CV folds (default: 5)")
args = parser.parse_args()

# Basic parameters
basic_parameters = parse_basic_params()
RUN_ID = get_run_timestamp()
OUTPUT_DIR = ensure_dir(os.path.join("results", "runs", f"{RUN_ID}_refine_network_{args.task}_{args.network}"))

HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Extract the network-specific FC data
regions = np.load(os.path.join(HCP_DIR, 'hcp_rest', 'regions.npy'), allow_pickle=True)
region_names = regions[:, 0].tolist()
network_types = regions[:, 1].tolist()
myelin_values = regions[:, 2].astype(float)

region_info = {
    'name': region_names,
    'network': network_types,
    'myelin': myelin_values
}
network_indices = [i for i, net in enumerate(region_info['network']) if net == args.network]
n_network_parcels = len(network_indices)

# Load data
fc_task_raw = np.load(f'FC_DATA/fc_{args.task}_{args.network}.npy')
fc_rest = np.load(f'FC_DATA/fc_rest_{args.network}.npy')

# ==========================================
# K-FOLD CROSS-VALIDATION (Awareness Fix)
# ConvAE trained on TRAIN subjects only,
# residuals + SDL computed on TEST subjects.
# ==========================================
n_folds = args.n_folds
indices = np.random.RandomState(42).permutation(N_SUBJECTS)
fold_size = N_SUBJECTS // n_folds

residuals_all = np.zeros((N_SUBJECTS, n_network_parcels, n_network_parcels))
refined_all = np.zeros((N_SUBJECTS, n_network_parcels, n_network_parcels))
reconstructed_all = np.zeros((N_SUBJECTS, n_network_parcels, n_network_parcels))
fold_accuracies_convae = []
fold_accuracies_sdl = []
fold_accuracies_before = []

print(f"Running {n_folds}-Fold CV to prevent double-dipping (awareness.txt fix)")
print(f"Task: {args.task}, Network: {args.network}, Subjects: {N_SUBJECTS}, Network Parcels: {n_network_parcels}")

for fold in range(n_folds):
    start_idx = fold * fold_size
    end_idx = N_SUBJECTS if fold == n_folds - 1 else (fold + 1) * fold_size
    test_idx = indices[start_idx:end_idx]
    train_idx = np.setdiff1d(indices, test_idx)
    
    n_train = len(train_idx)
    n_test = len(test_idx)
    
    print(f"\n--- Fold {fold+1}/{n_folds} | Train: {n_train} | Test: {n_test} ---")
    
    # --- TRAIN ConvAE on REST data of TRAIN subjects only ---
    train_rest = fc_rest[train_idx]
    train_tensor = torch.tensor(train_rest[:, np.newaxis, :, :], dtype=torch.float32)
    
    model = ConvAutoencoderFP().to(DEVICE)
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
    
    # --- EVALUATE on TEST subjects (never seen during training) ---
    model.eval()
    test_task = fc_task_raw[test_idx]
    test_task_tensor = torch.tensor(test_task[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    test_rest = fc_rest[test_idx]
    
    with torch.no_grad():
        test_reconstr = model(test_task_tensor).cpu().squeeze(1).numpy()
    
    test_residual = test_task - test_reconstr
    
    for i, idx in enumerate(test_idx):
        reconstructed_all[idx] = test_reconstr[i]
        residuals_all[idx] = test_residual[i]
    
    # --- SDL: Learn dictionary from TRAIN residuals, apply to TEST ---
    train_task = fc_task_raw[train_idx]
    train_task_tensor = torch.tensor(train_task[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        train_reconstr = model(train_task_tensor).cpu().squeeze(1).numpy()
    train_residual = train_task - train_reconstr
    
    n_tri = int(n_network_parcels * (n_network_parcels - 1) / 2)
    tril_idx = np.tril_indices(n_network_parcels, k=-1)
    
    Y_train = np.zeros((n_tri, n_train))
    for i in range(n_train):
        Y_train[:, i] = train_residual[i][tril_idx]
    
    D_train, _ = k_svd(Y_train, 15, 12, n_iter=10, verbose=False, random_state=42)
    
    Y_test = np.zeros((n_tri, n_test))
    for i in range(n_test):
        Y_test[:, i] = test_residual[i][tril_idx]
    
    X_test = omp_sparse_coding(Y_test, D_train, 12)
    sdl_retr = np.dot(D_train, X_test).T
    
    DX_test = np.zeros((n_test, n_network_parcels, n_network_parcels))
    for i in range(n_test):
        DX_test[i] = reconstruct_symmetric_matrix(sdl_retr[i], n_network_parcels)
    
    test_refined = test_residual - DX_test
    
    for i, idx in enumerate(test_idx):
        refined_all[idx] = test_refined[i]
    
    # --- Fold-level accuracy on HELD-OUT subjects ---
    test_rest_flat = test_rest.reshape(n_test, -1)
    
    corr_before = np.corrcoef(test_task.reshape(n_test, -1), test_rest_flat)[:n_test, n_test:]
    fold_accuracies_before.append(calculate_accuracy(corr_before))
    
    corr_convae = np.corrcoef(test_residual.reshape(n_test, -1), test_rest_flat)[:n_test, n_test:]
    fold_accuracies_convae.append(calculate_accuracy(corr_convae))
    
    corr_sdl = np.corrcoef(test_refined.reshape(n_test, -1), test_rest_flat)[:n_test, n_test:]
    fold_accuracies_sdl.append(calculate_accuracy(corr_sdl))
    
    print(f"  Fold {fold+1} Acc — Before: {fold_accuracies_before[-1]:.4f} | ConvAE: {fold_accuracies_convae[-1]:.4f} | ConvAE+SDL: {fold_accuracies_sdl[-1]:.4f}")

# ==========================================
# AGGREGATE CROSS-VALIDATED RESULTS
# ==========================================
accuracy_before = np.mean(fold_accuracies_before)
accuracy_after_convae = np.mean(fold_accuracies_convae)
accuracy_after_sdl = np.mean(fold_accuracies_sdl)

print(f"\n{'='*60}")
print(f"CROSS-VALIDATED RESULTS ({n_folds}-Fold)")
print(f"{'='*60}")
print(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy before: {accuracy_before:.4f} ± {np.std(fold_accuracies_before):.4f}')
print(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy ConvAE: {accuracy_after_convae:.4f} ± {np.std(fold_accuracies_convae):.4f}')
print(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy ConvAE+SDL: {accuracy_after_sdl:.4f} ± {np.std(fold_accuracies_sdl):.4f}')

# ==========================================
# VISUALIZATION
# ==========================================
first_sample = fc_task_raw[0, :, :]
reconstructed_sample = reconstructed_all[0]
residual_sample = residuals_all[0]
refined_sample = refined_all[0]

plt.figure(figsize=(16, 6))

plt.subplot(1, 4, 1)
plt.imshow(first_sample, cmap='viridis')
plt.title(f'Original {args.task.capitalize()} Sample')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(reconstructed_sample, cmap='viridis')
plt.title(f'Reconstructed {args.task.capitalize()} Sample')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(residual_sample, cmap='viridis')
plt.title(f'Residual {args.task.capitalize()} Sample')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(refined_sample, cmap='viridis')
plt.title(f'Refined {args.task.capitalize()} Sample')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{args.task}_{args.network}_fcs.png"))
plt.close()

# Correlation heatmaps
fc_rest_flat = fc_rest.reshape(N_SUBJECTS, -1)
corr_before_all = np.corrcoef(fc_task_raw.reshape(N_SUBJECTS, -1), fc_rest_flat)[:N_SUBJECTS, N_SUBJECTS:]
corr_convae_all = np.corrcoef(residuals_all.reshape(N_SUBJECTS, -1), fc_rest_flat)[:N_SUBJECTS, N_SUBJECTS:]
corr_sdl_all = np.corrcoef(refined_all.reshape(N_SUBJECTS, -1), fc_rest_flat)[:N_SUBJECTS, N_SUBJECTS:]

plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.heatmap(corr_before_all, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (Before ConvAE)")

plt.subplot(1, 3, 2)
sns.heatmap(corr_convae_all, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After ConvAE)")

plt.subplot(1, 3, 3)
sns.heatmap(corr_sdl_all, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After ConvAE+SDL)")

plt.savefig(os.path.join(OUTPUT_DIR, f"{args.task}_{args.network}_corr.png"))
plt.close()

# Save results
with open(os.path.join(OUTPUT_DIR, f'accuracy_results.txt'), 'w') as file:
    file.write(f"# {n_folds}-Fold Cross-Validated Results (Awareness Fix)\n")
    file.write(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy before: {accuracy_before:.4f} ± {np.std(fold_accuracies_before):.4f}\n')
    file.write(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy ConvAE: {accuracy_after_convae:.4f} ± {np.std(fold_accuracies_convae):.4f}\n')
    file.write(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy ConvAE+SDL: {accuracy_after_sdl:.4f} ± {np.std(fold_accuracies_sdl):.4f}\n')
    file.write(f'\nPer-fold accuracies (Before): {[f"{a:.4f}" for a in fold_accuracies_before]}\n')
    file.write(f'Per-fold accuracies (ConvAE): {[f"{a:.4f}" for a in fold_accuracies_convae]}\n')
    file.write(f'Per-fold accuracies (ConvAE+SDL): {[f"{a:.4f}" for a in fold_accuracies_sdl]}\n')

print(f'Accuracy results saved in "{OUTPUT_DIR}".')
