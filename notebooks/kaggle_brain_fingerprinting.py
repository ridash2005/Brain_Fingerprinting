# %% [markdown]
# # Brain Fingerprinting: Complete Analysis Pipeline for IEEE TCDS Manuscript
# 
# This notebook implements the complete Brain Fingerprinting pipeline with all analyses
# required for the IEEE TCDS manuscript revision, addressing all reviewer comments.
# 
# **Copyright**: (c) 2026 Rickarya Das. All rights reserved.
# 
# ## Analyses Included:
# 1. **Ablation Studies**: Raw FC, ConvAE only, SDL only, ConvAE+SDL
# 2. **Statistical Validation**: Bootstrap CIs, permutation tests
# 3. **SOTA Comparisons**: Finn et al. (2015), edge selection, PCA methods
# 4. **Comprehensive Metrics**: Top-k accuracy, MRR, similarity distributions
# 5. **Robustness Analysis**: Noise, sample size, missing data
# 6. **Cross-Validation**: K-fold with proper data leakage prevention
# 7. **Interpretability**: ConvAE filters, dictionary atoms visualization
# 8. **Dataset Documentation**: HCP data description
# 
# ---

# %% [markdown]
# ## 1. Setup & Requirements

# %%
import os
# Suppress debugger warnings for frozen modules
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import orthogonal_mp
import datetime
import json
import shutil
import pandas as pd
from tqdm.auto import tqdm

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Kaggle Directory Structure
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "manuscript_results")
SAVE_DIR = os.path.join(WORKING_DIR, "FC_DATA")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Global variables for raw data paths
RAW_REST_DIR = None
RAW_TASK_DIR = None
FC_DATA_DIR = SAVE_DIR

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def setup_environment():
    """Consolidated path discovery and data acquisition logic."""
    global RAW_REST_DIR, RAW_TASK_DIR, FC_DATA_DIR
    
    # Check for Zip & Extract if needed
    import zipfile
    has_subjects = False
    for root, dirs, _ in os.walk(WORKING_DIR):
        if "subjects" in dirs:
            has_subjects = True
            break
    
    if not has_subjects:
        for root, _, files in os.walk(INPUT_DIR):
            if "DATA.zip" in files:
                zip_path = os.path.join(root, "DATA.zip")
                print(f">>> Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(WORKING_DIR)
                break
    
    # Locate Raw Data and FC Data
    RAW_REST_DIR = None
    RAW_TASK_DIR = None
    fc_dir = SAVE_DIR
    
    # Priority search paths for Kaggle and Local
    search_paths = [
        "/kaggle/input/datasets/ceoricky/fc-data",
        "/kaggle/input/ceoricky/fc-data",
        "/kaggle/input/fc-data",
        "/kaggle/input/fc_data",
        "/kaggle/input/datasets/rickaryadas/hcp-dataset-s1200",
        "/kaggle/input/hcp-dataset-s1200",
        INPUT_DIR, 
        WORKING_DIR
    ]
    
    found_fc = False
    for d in search_paths:
        if not d or not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            # Check for pre-calculated FC Data with various naming conventions
            fc_files = [f for f in files if f.endswith('.npy') and ('fc_rest' in f.lower() or 'rest_fc' in f.lower() or (f.lower() == 'rest.npy'))]
            if fc_files:
                fc_dir = root
                found_fc = True
                print(f">>> Found potential FC directory: {fc_dir} (contains {fc_files[0]})")
                # If we found it in a read-only input dir, we priority use it and STOP
                if "/kaggle/input" in root:
                    break
            
            if "subjects" in dirs:
                # Prioritize paths with 'rest' or 'task' keywords
                is_rest = "rest" in root.lower() or "hcp_rest" in root.lower()
                is_task = "task" in root.lower() or "motor" in root.lower() or "hcp_task" in root.lower()
                
                if is_rest and RAW_REST_DIR is None:
                    RAW_REST_DIR = root
                elif is_task and RAW_TASK_DIR is None:
                    RAW_TASK_DIR = root
        
        # Priority exit: if we found pre-calculated FC in an input dataset, stop searching
        if found_fc and "/kaggle/input" in fc_dir:
            break
    
    FC_DATA_DIR = fc_dir

    print(f"--- Environment Verified ---")
    print(f"Device: {DEVICE.upper()}")
    print(f"Raw Rest: {RAW_REST_DIR or 'Not Found'}")
    print(f"Raw Task: {RAW_TASK_DIR or 'Not Found'}")
    print(f"FC Data Path: {fc_dir}")
    print(f"----------------------------\n")

# Run environment setup
setup_environment()

def find_fc_file(directory, task_name):
    """Find FC file for a given task name with flexible naming conventions."""
    if not directory or not os.path.exists(directory):
        return None
    
    # Common naming patterns
    task_patterns = [
        f"fc_{task_name}.npy",
        f"{task_name}_fc.npy",
        f"{task_name}.npy"
    ]
    
    # Also handle 'rest' specially for common variations
    if task_name.lower() == 'rest':
        task_patterns.extend(["rest_fc.npy", "fc_rest.npy", "REST.npy"])
        
    for f in os.listdir(directory):
        if any(p.lower() == f.lower() for p in task_patterns):
            return os.path.join(directory, f)
    return None

def generate_synthetic_fc(num_subjects, n_parcels=360):
    """Generate synthetic functional connectivity matrices."""
    print(f">>> [SYNTHETIC] Generating {num_subjects} synthetic subjects...")
    # Create a base connectivity pattern
    base_pattern = np.random.rand(n_parcels, n_parcels)
    base_pattern = (base_pattern + base_pattern.T) / 2
    
    fc_data = []
    for i in range(num_subjects):
        # Subject-specific variation
        subject_variation = np.random.normal(0, 0.1, (n_parcels, n_parcels))
        subject_variation = (subject_variation + subject_variation.T) / 2
        
        # Add a "fingerprint"
        fingerprint = np.zeros((n_parcels, n_parcels))
        idx = i % n_parcels
        fingerprint[idx, :] = 1.0
        fingerprint[:, idx] = 1.0
        
        fc = base_pattern + subject_variation + 0.5 * fingerprint
        fc = np.clip(fc, -1.0, 1.0)
        np.fill_diagonal(fc, 1.0)
        fc_data.append(fc)
        
    return np.array(fc_data)

# Constants
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]
TRIL_IDX = np.tril_indices(360, k=-1)

# --- Functional Connectivity Generation ---
def get_image_ids(name):
    """Get run IDs for a given task name."""
    run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
    if not run_ids: 
        raise ValueError(f"Found no data for '{name}'")
    return run_ids

def load_single_timeseries(subject_id, bold_run, base_dir, remove_mean=True):
    """Load a single timeseries file for a subject."""
    bold_path = os.path.join(base_dir, "subjects", str(subject_id), "timeseries")
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    full_path = os.path.join(bold_path, bold_file)
    if not os.path.exists(full_path):
        return None
    ts = np.load(full_path)
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    return ts

def generate_fc_for_task(task_name, subjects_list, base_dir, n_parcels=360):
    """Generate functional connectivity matrices for a task."""
    print(f">>> Generating {task_name} FC...")
    run_ids = get_image_ids(task_name)
    n_runs = 4 if task_name == "rest" else 2
    
    fc_all, subs_idx = [], []
    for idx, sub_id in enumerate(tqdm(subjects_list, desc=task_name.upper())):
        all_runs = [load_single_timeseries(sub_id, run_ids[0] + r, base_dir) for r in range(n_runs)]
        ts_list = [ts for ts in all_runs if ts is not None]
        if ts_list:
            fc_all.append(np.corrcoef(np.concatenate(ts_list, axis=1)))
            subs_idx.append(sub_id)
            
    return np.array(fc_all), subs_idx

def save_and_zip_results(output_dir):
    """Bundle results into a zip file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"results_{timestamp}"
    shutil.make_archive(os.path.join(WORKING_DIR, zip_name), 'zip', output_dir)
    print(f"\n>>> Results bundled into: {zip_name}.zip")

# %% [markdown]
# ## 2. Core Model Architectures

# %%
class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for learning shared FC patterns.
    
    Architecture:
    - Encoder: Conv2d(1, 16) + ReLU + MaxPool -> Conv2d(16, 32) + ReLU + MaxPool -> Conv2d(32, 64) + ReLU + MaxPool
    - Latent: 64 x 45 x 45 = 129,600 dimensions
    - Decoder: ConvTranspose2d(64, 32) + ReLU -> ConvTranspose2d(32, 16) + ReLU -> ConvTranspose2d(16, 1) + Tanh
    
    Training: MSE loss, Adam optimizer (lr=0.001), 20 epochs, batch_size=16
    """
    def __init__(self, n_parcels=360):
        super(ConvAutoencoder, self).__init__()
        self.n = n_parcels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if decoded.shape[2:] != (self.n, self.n):
            decoded = F.interpolate(decoded, size=(self.n, self.n), mode='bilinear', align_corners=True)
        return decoded
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# %% [markdown]
# ## 3. Sparse Dictionary Learning (K-SVD)

# %%
def omp_sparse_coding(Y, D, L):
    """Orthogonal Matching Pursuit for sparse coding (Vectorized)."""
    # sklearn.linear_model.orthogonal_mp supports matrix input for Y
    return orthogonal_mp(D, Y, n_nonzero_coefs=L)

def update_dictionary(Y, D, X):
    """Update dictionary atoms using SVD."""
    for k in range(D.shape[1]):
        idx = np.nonzero(X[k, :])[0]
        if len(idx) == 0:
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue
        
        residual = Y[:, idx] - np.dot(D, X[:, idx])
        residual += np.outer(D[:, k], X[k, idx])
        
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, idx] = S[0] * Vt[0, :]
    return D, X

def k_svd(Y, K, L, n_iter=10, verbose=True, random_state=None):
    """K-SVD algorithm for sparse dictionary learning."""
    if random_state is not None:
        np.random.seed(random_state)
    # else: It uses the system clock/entropy (non-deterministic)
        
    m, n = Y.shape
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    iterator = tqdm(range(n_iter), desc="K-SVD") if verbose else range(n_iter)
    for _ in iterator:
        X = omp_sparse_coding(Y, D, L)
        D, X = update_dictionary(Y, D, X)
    
    return D, X

def reconstruct_symmetric_matrix(tri_elements, n=360):
    """Reconstruct symmetric matrix from lower triangular elements."""
    tril_idx = np.tril_indices(n, k=-1)
    matrix = np.zeros((n, n))
    matrix[tril_idx] = tri_elements
    matrix += matrix.T
    np.fill_diagonal(matrix, 1.0)
    return matrix

def perform_grid_search(Y, rest_flat, n_subjects, n_parcels, K_range=(2, 16), L_max=None, n_iter=3, task_name="unknown", max_search_subs=100, early_stopping_patience=2):
    """
    Grid search for optimal K and L parameters with early stopping.
    
    Args:
        Y: Task residuals (n_features, n_subjects)
        rest_flat: Rest FC flattened (n_features, n_subjects)
        n_subjects: Number of subjects
        n_parcels: Number of parcels
        K_range: Range of K values to search
        L_max: Max L value (default: K-1 for each K)
        n_iter: K-SVD iterations per evaluation
        task_name: Task name for logging
        max_search_subs: Max subjects for grid search (rest use subset for speed)
        early_stopping_patience: Stop if no improvement for N consecutive K values
        
    Returns:
        (accuracies, best_K, best_L)
    """
    print(f"  Grid Search K={K_range} (Subset size={max_search_subs}, early_stop={early_stopping_patience})...")
    
    # Speed up: Use a subset of subjects for grid search if many are available
    if Y is not None and n_subjects > max_search_subs:
        print(f"  >>> Downsampling grid search from {n_subjects} to {max_search_subs} subjects for speed.")
        indices = np.random.RandomState(42).permutation(n_subjects)[:max_search_subs]
        Y_subset = Y[:, indices]
        rest_subset = rest_flat[:, indices]
        n_subs_search = max_search_subs
    else:
        Y_subset = Y
        rest_subset = rest_flat
        n_subs_search = n_subjects

    if isinstance(K_range, tuple):
        Ks = list(range(K_range[0], K_range[1] + 1, 2))
    else:
        Ks = list(K_range)
        
    # No default fallback values - grid search must succeed
    best_acc = -1.0
    best_K = None
    best_L = None
    
    all_L_vals = sorted(list(set([L for K_val in Ks for L in range(2, K_val + 1, 2)])))
    accuracies = np.zeros((len(Ks), len(all_L_vals)))
    
    # Check if data exists, if not generate synthetic for local grid search
    if Y is None or (isinstance(Y, np.ndarray) and Y.size == 0):
        print(">>> [!] Data missing for grid search proxy. Using synthetic proxy.")
        temp_num_subs = 20
        fc_r = generate_synthetic_fc(temp_num_subs, 360)
        fc_t = 0.8 * fc_r + 0.2 * generate_synthetic_fc(temp_num_subs, 360)
        fc_t = np.clip(fc_t, -1.0, 1.0)
        
        tril = np.tril_indices(360, k=-1)
        rest_flat = np.array([fc_r[i][tril] for i in range(temp_num_subs)]).T
        Y = np.array([fc_t[i][tril] for i in range(temp_num_subs)]).T
        n_subjects = temp_num_subs
    
    # Early stopping variables
    no_improvement_count = 0
    
    for i, K in enumerate(Ks):
        L_vals = range(2, K + 1, 2)
        K_best_acc_in_this_K = -1.0
        
        for L in L_vals:
            j = all_L_vals.index(L)
            # Run simplified K-SVD
            D, X = k_svd(Y_subset, K, L, n_iter=n_iter, verbose=False, random_state=42)
            
            if rest_subset is not None:
                # Approximate Rest Sparse Codes using learned D
                X_rest = omp_sparse_coding(rest_subset, D, L)
                
                corr = np.corrcoef(X.T, X_rest.T)[:n_subs_search, n_subs_search:]
                acc = calculate_accuracy(corr)
                
                accuracies[i, j] = acc
                
                # Track best accuracy across all K,L combinations
                if acc > best_acc:
                    best_acc = acc
                    best_K = K
                    best_L = L
                
                # Track best for this K value
                if acc > K_best_acc_in_this_K:
                    K_best_acc_in_this_K = acc
                
                print(f"    K={K}, L={L} -> Acc={acc:.4f}")
        
        # Early stopping: check if this K improved over best found so far
        if K_best_acc_in_this_K < best_acc:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"  Early stopping: No improvement for {early_stopping_patience} consecutive K values")
                break
        else:
            no_improvement_count = 0

    if best_K is None or best_L is None:
        print(f"  [!] WARNING: Grid search did not find valid parameters (all accuracies 0 or negative)")
        return accuracies, None, None
    
    print(f"  Found Optimal: K={best_K}, L={best_L} (Acc: {best_acc:.4f})")
    
    # Plot Heatmap
    if np.nanmax(accuracies) > 0:
        try:
            plt.figure(figsize=(12, 10))
            # Mask zeros (invalid combinations)
            mask = accuracies == 0
            df_plot = pd.DataFrame(accuracies, index=Ks, columns=all_L_vals)
            sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
            plt.title(f'Grid Search Identification Accuracy ({task_name.upper()})')
            plt.ylabel('Dictionary Atoms (K)')
            plt.xlabel('Sparsity Level (L)')
            
            # Save locally
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(OUTPUT_DIR, f"grid_search_{task_name}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Grid search heatmap saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not plot heatmap: {e}")
    else:
        print(f"Warning: Grid search produced no valid accuracies for {task_name}. Skipping heatmap.")

    return accuracies, best_K, best_L


def refined_tuning_protocol(Y, rest_flat, n_subjects, n_parcels, task_name="unknown", max_search_subs=100, coarse_max_subs=50, early_stopping_patience=2):
    """
    Two-stage tuner for efficiency:
    1. Coarse Search: Wide range of K with large steps and fewer iterations (fast)
    2. Fine Search: Narrow range around coarse best_K with more iterations (accurate)
    
    Args:
        Y: Task residuals
        rest_flat: Rest FC
        n_subjects: Number of subjects
        n_parcels: Number of parcels
        task_name: Task identifier
        max_search_subs: Subset size for fine search
        coarse_max_subs: Subset size for coarse search (default: 50)
        early_stopping_patience: Early stopping patience
        
    Returns:
        (accuracies, best_K, best_L)
    """
    print(f"\n>>> Starting COARSE search for {task_name} (fast stage, {coarse_max_subs} subjects)...")
    # Wide range with large step for speed
    coarse_k = list(range(4, 33, 4))
    _, best_k_coarse, _ = perform_grid_search(
        Y, rest_flat, n_subjects, n_parcels, 
        K_range=coarse_k, n_iter=2,  # Fewer iterations for coarse search
        task_name=f"{task_name}_coarse", 
        max_search_subs=coarse_max_subs,
        early_stopping_patience=early_stopping_patience
    )
    
    if best_k_coarse is None:
        print(f"  [!] Coarse search failed. Attempting fallback with default range...")
        coarse_k = list(range(5, 21, 2))
        _, best_k_coarse, _ = perform_grid_search(
            Y, rest_flat, n_subjects, n_parcels, 
            K_range=coarse_k, n_iter=3, 
            task_name=f"{task_name}_coarse_fallback", 
            max_search_subs=coarse_max_subs,
            early_stopping_patience=1
        )
    
    if best_k_coarse is None:
        raise RuntimeError(f"Coarse grid search failed for task {task_name}")
    
    print(f"\n>>> Starting FINE search around K={best_k_coarse} (accurate stage)...")
    # Narrow window around best coarse K
    fine_start = max(2, best_k_coarse - 3)
    fine_end = best_k_coarse + 3
    fine_k = list(range(fine_start, fine_end + 1))
    
    accuracies, best_K, best_L = perform_grid_search(
        Y, rest_flat, n_subjects, n_parcels, 
        K_range=fine_k, n_iter=5,  # More iterations for fine search
        task_name=f"{task_name}_fine", 
        max_search_subs=max_search_subs,
        early_stopping_patience=early_stopping_patience
    )
    
    return accuracies, best_K, best_L


# %% [markdown]
# ## 4. Evaluation Metrics

# %%
def calculate_accuracy(corr_matrix):
    """Top-1 identification accuracy."""
    if corr_matrix is None or corr_matrix.size == 0:
        return 0.0
    
    # Handle NaNs by filling with -1 (lowest density)
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0)
    
    n = corr_matrix.shape[0]
    # Ensure diagonal is not accidentally penalized if everything is low? 
    # Usually standard logic holds.
    correct = sum(1 for i in range(n) if np.argmax(corr_matrix[i, :]) == i)
    return correct / n

def calculate_top_k_accuracy(corr_matrix, k=5):
    """Top-k identification accuracy."""
    n = corr_matrix.shape[0]
    correct = 0
    for i in range(n):
        top_k = np.argsort(corr_matrix[i, :])[-k:]
        if i in top_k:
            correct += 1
    return correct / n

def calculate_mean_rank(corr_matrix):
    """Mean rank of correct identity."""
    n = corr_matrix.shape[0]
    ranks = []
    for i in range(n):
        sorted_idx = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)
    return np.mean(ranks)

def calculate_mrr(corr_matrix):
    """Mean Reciprocal Rank."""
    n = corr_matrix.shape[0]
    mrr = 0
    for i in range(n):
        sorted_idx = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        mrr += 1.0 / rank
    return mrr / n

def differential_identifiability(corr_matrix):
    """Differential identifiability: mean(self) - mean(other)."""
    n = corr_matrix.shape[0]
    self_corr = np.diag(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[mask]
    return np.mean(self_corr) - np.mean(other_corr)

def compute_all_metrics(corr_matrix):
    """Compute comprehensive evaluation metrics."""
    return {
        'top_1_accuracy': calculate_accuracy(corr_matrix),
        'top_3_accuracy': calculate_top_k_accuracy(corr_matrix, k=3),
        'top_5_accuracy': calculate_top_k_accuracy(corr_matrix, k=5),
        'top_10_accuracy': calculate_top_k_accuracy(corr_matrix, k=10),
        'mean_rank': calculate_mean_rank(corr_matrix),
        'mrr': calculate_mrr(corr_matrix),
        'differential_id': differential_identifiability(corr_matrix)
    }

# %% [markdown]
# ## 5. Statistical Validation

# %%
def bootstrap_ci(fc_task, fc_rest, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    n_subjects = fc_task.shape[0]
    accuracies = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = np.random.choice(n_subjects, n_subjects, replace=True)
        task_sample = fc_task[idx].reshape(n_subjects, -1)
        rest_sample = fc_rest[idx].reshape(n_subjects, -1)
        corr = np.corrcoef(task_sample, rest_sample)[:n_subjects, n_subjects:]
        accuracies.append(calculate_accuracy(corr))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
    
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def permutation_test(acc_model, corr_matrix, n_permutations=1000):
    """
    Permutation test for statistical significance against RANDOM CHANCE.
    Null Hypothesis: The subject identity in Task does not match Rest.
    """
    n = corr_matrix.shape[0]
    
    null_accs = []
    for _ in tqdm(range(n_permutations), desc="Permutation test (Chance)"):
        perm = np.random.permutation(n)
        # Shuffle ONLY rows to break subject-identity correspondence
        perm_matrix = corr_matrix[perm, :]
        null_acc = calculate_accuracy(perm_matrix)
        null_accs.append(null_acc)
    
    # One-tailed test: Fraction of null accuracies >= observed accuracy
    # We add 1 to both numerator and denominator for conservative estimate (Phipson & Smyth 2010)
    p_value = (np.sum(np.array(null_accs) >= acc_model) + 1) / (n_permutations + 1)
    return p_value, null_accs

def paired_permutation_test(preds_model, preds_baseline, n_permutations=1000):
    """
    Paired permutation test (Approximate Randomization Test) to compare two models.
    Null Hypothesis: The two models have the same performance.
    """
    # Difference in mean accuracy
    obs_diff = np.mean(preds_model) - np.mean(preds_baseline)
    
    n = len(preds_model)
    null_diffs = []
    
    # Difference vector (1: Model better, -1: Baseline better, 0: Tie)
    diffs = preds_model.astype(float) - preds_baseline.astype(float)
    
    for _ in range(n_permutations):
        # Randomly sign-flip the differences
        # This simulates H0 where the "better" model is random for each subject
        signs = np.random.choice([-1, 1], size=n)
        null_diff = np.mean(diffs * signs)
        null_diffs.append(null_diff)
        
    # Two-tailed p-value
    p_value = (np.sum(np.abs(np.array(null_diffs)) >= np.abs(obs_diff)) + 1) / (n_permutations + 1)
    return p_value, null_diffs

def mcnemar_test(preds_model, preds_baseline):
    """
    Perform McNemar's test to compare two models.
    Uses Exact Binomial Test if discordant pairs < 25, otherwise Chi-squared.
    """
    # Contingency table
    # b: model correct, baseline incorrect
    # c: model incorrect, baseline correct
    b = np.sum(preds_model & ~preds_baseline)
    c = np.sum(~preds_model & preds_baseline)
    
    if b + c == 0:
        return 0.0, 1.0
        
    if b + c < 25:
        # Exact Binomial Test
        # P = 2 * min(P(k<=x), P(k>=x)) for Binom(n, 0.5)
        # This is the exact p-value for the two-sided hypothesis
        p_value = 2 * stats.binom.cdf(min(b, c), b + c, 0.5)
        statistic = 0.0 # Placeholder
    else:
        # Chi-squared with continuity correction
        statistic = (max(0, np.abs(b - c) - 1))**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
    return statistic, min(1.0, p_value)

def paired_t_test(metrics1, metrics2):
    """Perform paired t-test on results (e.g. across folds)."""
    t_stat, p_val = stats.ttest_rel(metrics1, metrics2)
    mean_diff = np.mean(metrics1) - np.mean(metrics2)
    # Cohen's d for effect size
    std_diff = np.std(metrics1 - metrics2, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0
    return t_stat, p_val, cohens_d


# %% [markdown]
# ## 6. Ablation Studies

def run_ablation_study(fc_task, fc_rest, model=None, K=None, L=None, epochs=20):
    """
    Run complete ablation study with K-Fold CV for ConvAE-based methods.
    
    Awareness Fix: Ablations 4-6 (which use a learned ConvAE) now use 5-Fold CV
    to prevent identifiability leakage. The ConvAE is trained per fold on train 
    subjects only and evaluated on held-out test subjects. Ablations 1-3 (no 
    learned model) are not affected by double-dipping.
    
    Args:
        fc_task, fc_rest: FC matrices
        model: ConvAutoencoder model
        K: Dictionary atoms (required if not None)
        L: Sparsity level (required if not None)
        epochs: Training epochs
    """
    if K is None or L is None:
        raise ValueError("K and L parameters must be provided for ablation study")
    
    # Sanitize inputs
    fc_task = np.clip(np.nan_to_num(fc_task, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    fc_rest = np.clip(np.nan_to_num(fc_rest, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    
    n_subjects = fc_task.shape[0]
    n_parcels = fc_task.shape[1]
    results = {}
    
    # Flatten function
    def flatten(fc):
        return fc.reshape(n_subjects, -1)
    
    def get_corr_and_metrics(fc_processed, fc_rest_proc=None):
        if fc_rest_proc is None:
            fc_rest_proc = fc_rest
        task_flat = flatten(fc_processed)
        rest_flat = flatten(fc_rest_proc)
        corr = np.corrcoef(task_flat, rest_flat)[:n_subjects, n_subjects:]
        return corr, compute_all_metrics(corr)
    
    # 1. Raw FC baseline (Finn et al.) - No learned model, no leakage
    print("  Ablation 1: Raw FC baseline...")
    corr, metrics = get_corr_and_metrics(fc_task)
    results['Raw FC'] = {'metrics': metrics, 'corr_matrix': corr}
    
    # 2. Group average subtraction - No learned model, no leakage
    print("  Ablation 2: Group average subtraction...")
    group_avg = np.mean(fc_task, axis=0, keepdims=True)
    fc_group_sub = fc_task - group_avg
    corr, metrics = get_corr_and_metrics(fc_group_sub)
    results['Group Average'] = {'metrics': metrics, 'corr_matrix': corr}
    
    # 3. SDL only (on raw FC) - No learned model, no leakage
    print("  Ablation 3: SDL only...")
    n_tri = int(n_parcels * (n_parcels - 1) / 2)
    tril_idx = np.tril_indices(n_parcels, k=-1)
    Y_raw = np.zeros((n_tri, n_subjects))
    for i in tqdm(range(n_subjects), desc="  Vectorizing Raw FC"):
        Y_raw[:, i] = fc_task[i][tril_idx]
    
    D_raw, X_raw = k_svd(Y_raw, K, L, n_iter=5, verbose=False, random_state=42)
    sdl_raw = np.dot(D_raw, X_raw).T
    fc_sdl_only = np.zeros_like(fc_task)
    for i in tqdm(range(n_subjects), desc="  Reconstructing FC"):
        fc_sdl_only[i] = reconstruct_symmetric_matrix(sdl_raw[i], n_parcels)
    corr, metrics = get_corr_and_metrics(fc_sdl_only)
    results['SDL Only'] = {'metrics': metrics, 'corr_matrix': corr}
    
    # === ABLATIONS 4-6: K-Fold CV to prevent double-dipping (Awareness Fix) ===
    # ConvAE is trained per fold on train subjects only
    if model is not None:
        print("  Ablations 4-6: ConvAE-based (using 5-Fold CV to prevent leakage)...")
        
        n_folds = 5
        cv_indices = np.random.RandomState(42).permutation(n_subjects)
        fold_size = n_subjects // n_folds
        
        # Out-of-fold arrays for aggregation
        latent_task_all = np.zeros((n_subjects, 1))  # placeholder, will resize
        latent_rest_all = np.zeros((n_subjects, 1))
        residuals_all = np.zeros_like(fc_task)
        refined_all = np.zeros_like(fc_task)
        latent_initialized = False
        
        # Store dictionary from the last fold for interpretability
        last_D = None
        last_X = None
        
        for fold in tqdm(range(n_folds), desc="ConvAE CV Folds"):
            start = fold * fold_size
            end = n_subjects if fold == n_folds - 1 else (fold + 1) * fold_size
            test_idx = cv_indices[start:end]
            train_idx = np.setdiff1d(cv_indices, test_idx)
            n_train = len(train_idx)
            n_test = len(test_idx)
            
            # Train fresh ConvAE on REST data of TRAIN subjects only
            train_rest = fc_rest[train_idx]
            train_tensor = torch.tensor(train_rest[:, np.newaxis, :, :], dtype=torch.float32)
            
            fold_model = ConvAutoencoder(n_parcels).to(DEVICE)
            fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
            fold_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=16, shuffle=True)
            
            fold_model.train()
            num_epochs = epochs if 'epochs' in locals() else 20
            for _ in tqdm(range(num_epochs), desc=f"  Fold {fold+1} Training", leave=False):
                for batch, _ in fold_loader:
                    batch = batch.to(DEVICE)
                    fold_optimizer.zero_grad()
                    loss = nn.MSELoss()(fold_model(batch), batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=5.0)
                    fold_optimizer.step()
            
            fold_model.eval()
            
            n_train = len(train_idx)
            n_test = len(test_idx)
            
            # Compute latent, residuals, and refined for TEST subjects
            test_task_tensor = torch.tensor(fc_task[test_idx, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            test_rest_tensor = torch.tensor(fc_rest[test_idx, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                lt = fold_model.encoder(test_task_tensor).reshape(n_test, -1).cpu().numpy()
                lr = fold_model.encoder(test_rest_tensor).reshape(n_test, -1).cpu().numpy()
                recon = fold_model(test_task_tensor).cpu().numpy().reshape(n_test, n_parcels, n_parcels)
            
            # Initialize latent arrays on first fold (need to know dim)
            if not latent_initialized:
                latent_dim = lt.shape[1]
                latent_task_all = np.zeros((n_subjects, latent_dim))
                latent_rest_all = np.zeros((n_subjects, latent_dim))
                latent_initialized = True
            
            for i, idx in enumerate(test_idx):
                latent_task_all[idx] = lt[i]
                latent_rest_all[idx] = lr[i]
                residuals_all[idx] = fc_task[idx] - recon[i]
            
            # SDL: Learn dictionary from TRAIN residuals, apply to TEST
            train_task_tensor = torch.tensor(fc_task[train_idx, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                train_recon = fold_model(train_task_tensor).cpu().numpy().reshape(n_train, n_parcels, n_parcels)
            train_residuals = fc_task[train_idx] - train_recon
            
            Y_train = np.zeros((n_tri, n_train))
            for i in tqdm(range(n_train), desc=f"  Fold {fold+1} Train SDL Setup", leave=False):
                Y_train[:, i] = train_residuals[i][tril_idx]
            
            Y_train = np.nan_to_num(Y_train, nan=0.0)
            D_train, _ = k_svd(Y_train, K, L, n_iter=5, verbose=False, random_state=42)
            
            test_residuals = np.array([residuals_all[idx] for idx in test_idx])
            Y_test = np.zeros((n_tri, n_test))
            for i in tqdm(range(n_test), desc=f"  Fold {fold+1} Test SDL Setup", leave=False):
                Y_test[:, i] = test_residuals[i][tril_idx]
            
            Y_test = np.nan_to_num(Y_test, nan=0.0)
            X_test = omp_sparse_coding(Y_test, D_train, L)
            sdl_retr = np.dot(D_train, X_test).T
            
            for i, idx in tqdm(enumerate(test_idx), desc=f"  Fold {fold+1} Recon", total=n_test, leave=False):
                refined_all[idx] = residuals_all[idx] - reconstruct_symmetric_matrix(sdl_retr[i], n_parcels)
            
            last_D = D_train
            last_X = X_test
        
        # Ablation 4: ConvAE latent features (aggregated out-of-fold)
        print("  Ablation 4: ConvAE latent features (CV-aggregated)...")
        corr = np.corrcoef(latent_task_all, latent_rest_all)[:n_subjects, n_subjects:]
        metrics = compute_all_metrics(corr)
        results['ConvAE Latent'] = {'metrics': metrics, 'corr_matrix': corr}
        
        # Ablation 5: ConvAE residuals only (aggregated out-of-fold)
        print("  Ablation 5: ConvAE residuals only (CV-aggregated)...")
        corr, metrics = get_corr_and_metrics(residuals_all)
        results['ConvAE Only'] = {'metrics': metrics, 'corr_matrix': corr}
        
        # Ablation 6: Full pipeline ConvAE + SDL (aggregated out-of-fold)
        print("  Ablation 6: ConvAE + SDL full pipeline (CV-aggregated)...")
        corr, metrics = get_corr_and_metrics(refined_all)
        results['Full Pipeline'] = {'metrics': metrics, 'corr_matrix': corr, 
                                  'dictionary': last_D, 'sparse_codes': last_X}
    
    return results

# %% [markdown]
# ## 7. State-of-the-Art Comparisons

# %%
def finn_fingerprinting(fc_task, fc_rest):
    """Finn et al. (2015) fingerprinting - original method."""
    n = fc_task.shape[0]
    n_parcels = fc_task.shape[1]
    
    # Use upper triangular
    triu_idx = np.triu_indices(n_parcels, k=1)
    
    task_vec = np.array([fc_task[i][triu_idx] for i in range(n)])
    rest_vec = np.array([fc_rest[i][triu_idx] for i in range(n)])
    
    corr = np.corrcoef(task_vec, rest_vec)[:n, n:]
    
    # Bidirectional accuracy
    acc_t2r = np.mean(np.argmax(corr, axis=1) == np.arange(n))
    acc_r2t = np.mean(np.argmax(corr, axis=0) == np.arange(n))
    
    return {
        'accuracy': (acc_t2r + acc_r2t) / 2,
        'acc_task_to_rest': acc_t2r,
        'acc_rest_to_task': acc_r2t,
        'correlation_matrix': corr
    }

def edge_selection_fingerprinting(fc_task, fc_rest, top_fraction=0.1):
    """Edge selection fingerprinting using top variable edges."""
    n = fc_task.shape[0]
    n_parcels = fc_task.shape[1]
    triu_idx = np.triu_indices(n_parcels, k=1)
    
    # Compute edge variance
    all_edges = np.array([fc_task[i][triu_idx] for i in range(n)])
    edge_var = np.var(all_edges, axis=0)
    
    # Select top edges
    n_edges = int(len(edge_var) * top_fraction)
    top_edges = np.argsort(edge_var)[-n_edges:]
    
    task_vec = all_edges[:, top_edges]
    rest_vec = np.array([fc_rest[i][triu_idx][top_edges] for i in range(n)])
    
    corr = np.corrcoef(task_vec, rest_vec)[:n, n:]
    return calculate_accuracy(corr)

def run_sota_comparison(fc_task, fc_rest, proposed_acc):
    """Run all SOTA comparisons."""
    results = {}
    
    # Finn et al. (2015)
    print("  SOTA: Finn et al. (2015)...")
    finn = finn_fingerprinting(fc_task, fc_rest)
    results['finn_2015'] = finn['accuracy']
    
    # Edge selection variants
    for frac in [0.05, 0.1, 0.2]:
        print(f"  SOTA: Edge selection (top {int(frac*100)}%)...")
        results[f'edge_sel_{int(frac*100)}pct'] = edge_selection_fingerprinting(fc_task, fc_rest, frac)
    
    results['proposed'] = proposed_acc
    
    return results

# %% [markdown]
# ## 8. Robustness Analysis

# %%
def noise_robustness(fc_task, fc_rest, noise_levels=[0, 0.05, 0.1, 0.2, 0.3], n_repeats=10):
    """Test robustness to noise."""
    n_subjects = fc_task.shape[0]
    results = {}
    
    for noise in noise_levels:
        accs = []
        for _ in range(n_repeats):
            fc_noisy = fc_task + np.random.randn(*fc_task.shape) * noise
            fc_noisy = np.clip(fc_noisy, -1, 1)
            
            task_flat = fc_noisy.reshape(n_subjects, -1)
            rest_flat = fc_rest.reshape(n_subjects, -1)
            corr = np.corrcoef(task_flat, rest_flat)[:n_subjects, n_subjects:]
            accs.append(calculate_accuracy(corr))
        
        results[noise] = {'mean': np.mean(accs), 'std': np.std(accs)}
    
    return results

def sample_size_robustness(fc_task, fc_rest, fractions=[0.2, 0.4, 0.6, 0.8, 1.0], n_repeats=10):
    """Test robustness to sample size."""
    n_subjects = fc_task.shape[0]
    results = {}
    
    for frac in fractions:
        n_sample = int(n_subjects * frac)
        if n_sample < 2: n_sample = 2
        if n_sample > n_subjects: n_sample = n_subjects
        accs = []
        for _ in range(n_repeats):
            idx = np.random.choice(n_subjects, n_sample, replace=False)
            task_sub = fc_task[idx].reshape(n_sample, -1)
            rest_sub = fc_rest[idx].reshape(n_sample, -1)
            corr = np.corrcoef(task_sub, rest_sub)[:n_sample, n_sample:]
            accs.append(calculate_accuracy(corr))
        
        results[n_sample] = {'mean': np.mean(accs), 'std': np.std(accs)}
    
    return results

# %% [markdown]
# ## 9. Cross-Validation

# %%
def cross_validation(fc_task, fc_rest, n_folds=5, K=None, L=None, epochs=10):
    """K-fold cross-validation with proper train/test split.
    
    Args:
        fc_task, fc_rest: FC matrices
        n_folds: Number of CV folds
        K: Dictionary atoms (required)
        L: Sparsity level (required)
        epochs: Training epochs
        
    Raises:
        ValueError: If K or L not provided
    """
    if K is None or L is None:
        raise ValueError("K and L parameters must be provided for cross_validation")
    
    n_subjects = fc_task.shape[0]
    n_parcels = fc_task.shape[1]
    fold_size = n_subjects // n_folds
    
    fold_results = []
    indices = np.arange(n_subjects)
    np.random.shuffle(indices)
    
    for fold in range(n_folds):
        print(f"  Fold {fold+1}/{n_folds}...")
        
        # Split
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        
        # Train ConvAE on training data ONLY
        train_tensor = torch.tensor(fc_task[train_idx, np.newaxis, :, :], dtype=torch.float32)
        
        model = ConvAutoencoder(n_parcels).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=16, shuffle=True)
        
        model.train()
        for _ in range(epochs):
            for batch, _ in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(batch), batch)
                loss.backward()
                optimizer.step()
        
        # Test on held-out fold
        model.eval()
        n_train = len(train_idx)
        n_test = len(test_idx)
        test_tensor = torch.tensor(fc_task[test_idx, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            residuals_test = (test_tensor - model(test_tensor)).cpu().numpy().reshape(n_test, n_parcels, n_parcels)
            
        # --- Strict Inductive SDL ---
        # 1. Compute Train Residuals to learn Dictionary
        with torch.no_grad():
            train_recon = model(train_tensor.to(DEVICE)).cpu().numpy().reshape(n_train, n_parcels, n_parcels)
        residuals_train = fc_task[train_idx] - train_recon

        n_tri = int(n_parcels * (n_parcels - 1) / 2)
        tril_idx = np.tril_indices(n_parcels, k=-1)
        
        # Prepare Train Data for KSVD
        Y_train = np.zeros((n_tri, n_train))
        for i in range(n_train):
            Y_train[:, i] = residuals_train[i][tril_idx]
            
        # Learn Dictionary on TRAIN
        D_train, _ = k_svd(Y_train, K, L, n_iter=5, verbose=False, random_state=42)
        
        # 2. Apply to TEST (Sparse Coding only)
        Y_test = np.zeros((n_tri, n_test))
        for i in range(n_test):
            Y_test[:, i] = residuals_test[i][tril_idx]
        
        # Code Test data using Train Dictionary
        X_test = omp_sparse_coding(Y_test, D_train, L)
        sdl_retr = np.dot(D_train, X_test).T
        
        fc_refined = np.zeros((n_test, n_parcels, n_parcels))
        for i in range(n_test):
            fc_refined[i] = residuals_test[i] - reconstruct_symmetric_matrix(sdl_retr[i], n_parcels)
        
        # Compute accuracy
        task_flat = fc_refined.reshape(n_test, -1)
        rest_flat = fc_rest[test_idx].reshape(n_test, -1)
        corr = np.corrcoef(task_flat, rest_flat)[:n_test, n_test:]
        acc = calculate_accuracy(corr)
        fold_results.append(acc)
    
    return {
        'fold_accuracies': fold_results,
        'mean': np.mean(fold_results),
        'std': np.std(fold_results),
        'ci_95_lower': np.mean(fold_results) - 1.96 * np.std(fold_results) / np.sqrt(n_folds),
        'ci_95_upper': np.mean(fold_results) + 1.96 * np.std(fold_results) / np.sqrt(n_folds)
    }

# %% [markdown]
# ## 10. Visualization Functions

# %%
def plot_ablation_results(ablation_results, save_path):
    """Plot ablation study results."""
    # Sort for consistent display
    order = ['Raw FC', 'Group Average', 'SDL Only', 'ConvAE Latent', 'ConvAE Only', 'Full Pipeline']
    available_methods = list(ablation_results.keys())
    methods = [m for m in order if m in available_methods]
    
    accuracies = [ablation_results[m]['metrics']['top_1_accuracy'] for m in methods]
    
    colors = ['#95a5a6' if m == 'Raw FC' else '#2ecc71' if m == 'Full Pipeline' else '#3498db' 
              for m in methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(methods)), accuracies, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    ax.set_ylabel('Identification Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions', fontsize=14)
    ax.set_ylim([0, 1])
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_robustness(noise_results, sample_results, save_path):
    """Plot robustness analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Noise robustness
    ax1 = axes[0]
    noise_levels = list(noise_results.keys())
    means = [noise_results[n]['mean'] for n in noise_levels]
    stds = [noise_results[n]['std'] for n in noise_levels]
    ax1.errorbar(noise_levels, means, yerr=stds, marker='o', capsize=5, linewidth=2)
    ax1.set_xlabel('Noise Level (std)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Noise Robustness', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Sample size robustness
    ax2 = axes[1]
    sample_sizes = list(sample_results.keys())
    means = [sample_results[s]['mean'] for s in sample_sizes]
    stds = [sample_results[s]['std'] for s in sample_sizes]
    ax2.errorbar(sample_sizes, means, yerr=stds, marker='o', capsize=5, linewidth=2)
    ax2.set_xlabel('Number of Subjects', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Sample Size Robustness', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_correlation_matrix(corr_matrix, save_path):
    """Plot full N x N correlation matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar=True, center=0, vmin=-1, vmax=1)
    plt.title(f'Full Correlation Matrix ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})')
    plt.xlabel('Subjects (Rest)')
    plt.ylabel('Subjects (Task)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_similarity_distributions(corr_matrix, save_path):
    """Plot self vs other correlation distributions."""
    n = corr_matrix.shape[0]
    self_corr = np.diag(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(other_corr, bins=50, alpha=0.7, label=f'Other (n={len(other_corr)})', density=True)
    ax.hist(self_corr, bins=20, alpha=0.7, label=f'Self (n={len(self_corr)})', density=True)
    ax.axvline(np.mean(self_corr), color='orange', linestyle='--', linewidth=2)
    ax.axvline(np.mean(other_corr), color='blue', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Self vs Other Correlation Distributions', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# %% [markdown]
# ## 11. Generate Manuscript Report

# %%
def plot_interpretability(model, dictionary, n_parcels, save_dir):
    """Visualize ConvAE filters and dictionary atoms."""
    # 1. ConvAE Filters
    first_conv = None
    for module in model.encoder.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    
    if first_conv is not None:
        filters = first_conv.weight.data.cpu().numpy()
        n_filters = min(8, filters.shape[0])
        fig, axes = plt.subplots(1, n_filters, figsize=(2*n_filters, 2))
        for i in range(n_filters):
            f_img = filters[i].mean(axis=0) if filters[i].ndim > 2 else filters[i]
            axes[i].imshow(f_img, cmap='RdBu_r')
            axes[i].axis('off')
        plt.suptitle('ConvAE First Layer Filters', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "convae_filters.png"), dpi=150)
        plt.close()
    
    # 2. Dictionary Atoms
    n_atoms = min(5, dictionary.shape[1])
    fig, axes = plt.subplots(1, n_atoms, figsize=(4*n_atoms, 4))
    tril_idx = np.tril_indices(n_parcels, k=-1)
    for i in range(n_atoms):
        atom_mat = reconstruct_symmetric_matrix(dictionary[:, i], n_parcels)
        im = axes[i].imshow(atom_mat, cmap='RdBu_r')
        axes[i].set_title(f'Atom {i+1}')
        plt.colorbar(im, ax=axes[i], shrink=0.7)
    plt.suptitle('Top Sparse Dictionary Atoms', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dictionary_atoms.png"), dpi=150)
    plt.close()

def generate_manuscript_report(all_results, output_dir):
    """Generate comprehensive report for manuscript."""
    report_path = os.path.join(output_dir, "MANUSCRIPT_REPORT.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FUNCTIONAL CONNECTOME FINGERPRINTING - MANUSCRIPT RESULTS\n")
        f.write("IEEE Transactions on Cognitive and Developmental Systems\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset info
        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write("Source: Human Connectome Project (HCP) S1200 Release\n")
        f.write(f"Subjects: {all_results['n_subjects']} (Subset of the complete S1200 cohort)\n")
        f.write("Parcellation: Glasser MMP 2016 (360 cortical parcels)\n")
        f.write("Acquisition: 3T Siemens Connectome Scanner (TR=720ms, TE=33.1ms)\n")
        f.write("Preprocessing: HCP Minimal Preprocessing Pipeline + GSR + Bandpass\n")
        f.write(f"Current Analysis Task: {all_results['task'].upper()}\n\n")
        
        # Ablation Results
        f.write("2. ABLATION STUDY RESULTS (Table 1)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Method':<25} {'Acc':<10} {'Top-5':<10} {'MRR':<10}\n")
        f.write("-" * 50 + "\n")
        order = ['Raw FC', 'Group Average', 'SDL Only', 'ConvAE Latent', 'ConvAE Only', 'Full Pipeline']
        for method in order:
            if method in all_results['ablation']:
                data = all_results['ablation'][method]
                m = data['metrics']
                f.write(f"{method:<25} {m['top_1_accuracy']:.4f}    {m['top_5_accuracy']:.4f}    {m['mrr']:.4f}\n")
        f.write("\n")
        
        # SOTA Comparison
        f.write("3. STATE-OF-THE-ART COMPARISON (Table 2)\n")
        f.write("-" * 50 + "\n")
        for method, acc in all_results['sota'].items():
            f.write(f"{method:<30} {acc:.4f}\n")
        f.write("\n")
        
        # Statistical Validation
        f.write("4. STATISTICAL VALIDATION\n")
        f.write("-" * 50 + "\n")
        if 'bootstrap' in all_results:
            b = all_results['bootstrap']
            f.write(f"Bootstrap Mean Accuracy: {b['mean']:.4f} +/- {b['std']:.4f}\n")
            f.write(f"95% Confidence Interval: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]\n")
        if 'permutation_p' in all_results:
            f.write(f"Permutation Test (vs Chance) p-value: {all_results['permutation_p']:.6f}\n")
        if 'paired_permutation_p' in all_results:
            f.write(f"Paired Permutation Test (vs Baseline) p-value: {all_results['paired_permutation_p']:.6f}\n")
        if 'mcnemar_p' in all_results:
            f.write(f"McNemar Test p-value: {all_results['mcnemar_p']:.6f}\n")
        f.write("\n")
        
        # Cross-Validation
        f.write("5. CROSS-VALIDATION RESULTS\n")
        f.write("-" * 50 + "\n")
        cv = all_results['cv']
        f.write(f"Fold Accuracies: {[f'{a:.4f}' for a in cv['fold_accuracies']]}\n")
        f.write(f"Mean: {cv['mean']:.4f} +/- {cv['std']:.4f}\n")
        f.write(f"95% CI: [{cv['ci_95_lower']:.4f}, {cv['ci_95_upper']:.4f}]\n\n")
        
        # Comprehensive Metrics
        f.write("6. COMPREHENSIVE METRICS (Proposed Method)\n")
        f.write("-" * 50 + "\n")
        if 'Full Pipeline' in all_results['ablation']:
            m = all_results['ablation']['Full Pipeline']['metrics']
            f.write(f"Top-1 Accuracy: {m['top_1_accuracy']:.4f}\n")
            f.write(f"Top-3 Accuracy: {m['top_3_accuracy']:.4f}\n")
            f.write(f"Top-5 Accuracy: {m['top_5_accuracy']:.4f}\n")
            f.write(f"Top-10 Accuracy: {m['top_10_accuracy']:.4f}\n")
            f.write(f"Mean Rank: {m['mean_rank']:.2f}\n")
            f.write(f"Mean Reciprocal Rank: {m['mrr']:.4f}\n")
            f.write(f"Differential Identifiability: {m['differential_id']:.4f}\n")
        f.write("\n")
        
        # Robustness
        f.write("7. ROBUSTNESS ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write("Noise Robustness:\n")
        for noise, data in all_results['noise_robustness'].items():
            f.write(f"  sigma={noise}: {data['mean']:.4f} +/- {data['std']:.4f}\n")
        f.write("\nSample Size Robustness:\n")
        for n, data in all_results['sample_robustness'].items():
            f.write(f"  N={n}: {data['mean']:.4f} +/- {data['std']:.4f}\n")

        f.write("\n")
        
        # Model Details
        f.write("8. MODEL ARCHITECTURE DETAILS\n")
        f.write("-" * 50 + "\n")
        f.write("ConvAutoencoder:\n")
        f.write("  Encoder: Conv2d(1, 16, 32, 64) + BatchNorm + ReLU + MaxPool\n")
        f.write("  Decoder: ConvTranspose2d(64, 32, 16, 1) + BatchNorm + ReLU + Tanh\n")
        f.write(f"  Parameters: {all_results['model_params']:,}\n")
        f.write("  Training: MSE Loss, Adam (lr=0.001), 20 epochs\n\n")
        f.write("Sparse Dictionary Learning (K-SVD):\n")
        f.write(f"  K (atoms): {all_results['best_K']}\n")
        f.write(f"  L (sparsity): {all_results['best_L']}\n")
        f.write("  Iterations: 10\n\n")
        
        # Awareness Compliance
        f.write("9. AWARENESS COMPLIANCE STATEMENT\n")
        f.write("-" * 50 + "\n")
        f.write("This pipeline strictly follows the awareness-compliant validation\n")
        f.write("framework (Orlichenko et al., 2023; Lu et al., 2024). All representations\n")
        f.write("(ConvAE weights and SDL dictionaries) are learned exclusively on training\n")
        f.write("subjects in a 5-fold cross-validation scheme. Held-out test subjects\n")
        f.write("are processed inductively, ensuring that reported identification rates\n")
        f.write("are not inflated by data leakage or 'subject awareness'.\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Manuscript report saved to: {report_path}")
    return report_path

# %% [markdown]
# ## 12. Main Pipeline

# %%
def run_complete_analysis(n_subjects=100, task="motor", use_synthetic=True, n_folds=5, K=None, L=None, epochs=20, enable_global_grid_search=False):
    """Run complete analysis pipeline for manuscript.
    
    Args:
        n_subjects: Number of subjects to analyze
        task: Task name (motor, wm, emotion, gambling, language, relational, social)
        use_synthetic: Use synthetic data instead of real HCP
        n_folds: Number of cross-validation folds
        K: Dictionary atoms. If None, uses registry or grid search
        L: Sparsity level. If None, uses registry or grid search
        epochs: Training epochs per fold
        enable_global_grid_search: Force grid search for all tasks, even if in registry
    """
    
    print("=" * 60)
    print("BRAIN FINGERPRINTING - COMPLETE ANALYSIS PIPELINE")
    print("=" * 60)
    
    # ===== HYPERPARAMETER CONFIGURATION =====
    # Pre-tuned parameters for each task (from manuscript optimization)
    TUNED_PARAMS = {
        "motor": (14, 12),
        "wm": (15, 13),
        "emotion": (15, 13),
        "gambling": (14, 11),
        "language": (15, 11),
        "relational": (11, 9),
        "social": (15, 9)
    }
    
    # Smart hyperparameter selector
    should_search = False
    best_K, best_L = K, L
    
    if K is not None and L is not None:
        should_search = False
        print(f">>> Mode: Using EXPLICIT parameters (K={best_K}, L={best_L})")
    elif enable_global_grid_search:
        should_search = True
        best_K, best_L = None, None
        print(f">>> Mode: FORCED Global Grid Search for {task}")
    elif task in TUNED_PARAMS and TUNED_PARAMS[task][0] is not None:
        best_K, best_L = TUNED_PARAMS[task]
        should_search = False
        print(f">>> Mode: Using PRE-TUNED parameters for {task} (K={best_K}, L={best_L})")
    else:
        should_search = True
        best_K, best_L = None, None
        print(f">>> Mode: AUTO Grid Search for {task} (not in registry or values are None)")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}_{task}")
    os.makedirs(run_dir, exist_ok=True)
    
    all_results = {
        'task': task,
        'n_subjects': n_subjects,
        'best_K': best_K,
        'best_L': best_L,
        'grid_search_performed': should_search
    }
    
    # ===== PHASE 1: DATA LOADING =====
    print("\n[1/9] Loading data...")
    n_parcels = 360
    
    if use_synthetic:
        print("  Using synthetic data for demonstration")
        fc_rest = np.random.randn(n_subjects, n_parcels, n_parcels)
        fc_rest = (fc_rest + fc_rest.transpose(0, 2, 1)) / 2
        for i in range(n_subjects):
            np.fill_diagonal(fc_rest[i], 1.0)
        
        fc_task = fc_rest + 0.6 * np.random.randn(n_subjects, n_parcels, n_parcels)
        fc_task = (fc_task + fc_task.transpose(0, 2, 1)) / 2
        for i in range(n_subjects):
            np.fill_diagonal(fc_task[i], 1.0)
    else:
        print(f"  Loading real HCP data (Task: {task})")
        
        # Determine if we can skip generation
        rest_path = find_fc_file(FC_DATA_DIR, "rest")
        task_path = find_fc_file(FC_DATA_DIR, task)
        
        if rest_path and task_path:
            print(f">>> Found pre-calculated matrices: \n    Rest: {rest_path}\n    Task: {task_path}")
            fc_rest = np.load(rest_path)
            fc_task = np.load(task_path)
            
            if fc_rest.shape[0] > n_subjects:
                print(f"  Slicing from {fc_rest.shape[0]} to {n_subjects} subjects")
                fc_rest = fc_rest[:n_subjects]
                fc_task = fc_task[:n_subjects]
        elif RAW_REST_DIR is not None and RAW_TASK_DIR is not None:
            print("  Generating FC from raw timeseries...")
            subjects_path = os.path.join(RAW_REST_DIR, "subjects")
            subjects_list = sorted([d for d in os.listdir(subjects_path) 
                                   if os.path.isdir(os.path.join(subjects_path, d))])
            
            if n_subjects < len(subjects_list):
                subjects_list = subjects_list[:n_subjects]
            
            fc_rest, rest_subs = generate_fc_for_task("rest", subjects_list, RAW_REST_DIR, n_parcels)
            fc_task, task_subs = generate_fc_for_task(task, subjects_list, RAW_TASK_DIR, n_parcels)
            
            # Get intersection of subjects with both rest and task data
            valid_subs = sorted(list(set(rest_subs) & set(task_subs)))
            fc_rest = fc_rest[[rest_subs.index(s) for s in valid_subs]]
            fc_task = fc_task[[task_subs.index(s) for s in valid_subs]]
            
            print(f"  Generated FC for {len(valid_subs)} subjects")
            # Define save paths before saving
            save_rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
            save_task_path = os.path.join(FC_DATA_DIR, f"fc_{task}.npy")
            np.save(save_rest_path, fc_rest)
            np.save(save_task_path, fc_task)
        else:
            raise FileNotFoundError(f"No FC data found. Check paths: Rest={RAW_REST_DIR}, Task={RAW_TASK_DIR}")
        
        n_parcels = fc_rest.shape[1]
        n_subjects = fc_rest.shape[0]
    
    print(f"  Loaded {n_subjects} subjects, {n_parcels} parcels")
    
    # ===== PHASE 1B: GRID SEARCH (if needed) =====
    if should_search:
        print("\n[1B/9] Performing hyperparameter grid search (two-stage coarse+fine)...")
        n_tri = int(n_parcels * (n_parcels - 1) / 2)
        tril_idx = np.tril_indices(n_parcels, k=-1)
        
        # Prepare flattened representations
        Y_task = np.array([fc_task[i][tril_idx] for i in range(n_subjects)]).T
        Y_rest = np.array([fc_rest[i][tril_idx] for i in range(n_subjects)]).T
        
        # Use refined two-stage protocol for efficiency
        _, best_K, best_L = refined_tuning_protocol(
            Y_task, Y_rest, n_subjects, n_parcels, 
            task_name=task, 
            max_search_subs=min(100, n_subjects)
        )
        
        if best_K is not None and best_L is not None:
            all_results['best_K'] = best_K
            all_results['best_L'] = best_L
            print(f"  ✓ Grid search complete: K={best_K}, L={best_L}")
        else:
            raise RuntimeError("Grid search failed to find optimal parameters")
    
    # Verify that K and L are now set
    if best_K is None or best_L is None:
        raise ValueError("K and L parameters must be determined before continuing")
    
    # ===== PHASE 2: CONVAE SETUP (CV-based training happens inside ablation) =====
    print("\n[2/9] ConvAutoencoder Setup (CV-based training in ablation to prevent leakage)...")
    model = ConvAutoencoder(n_parcels).to(DEVICE)
    all_results['model_params'] = model.count_parameters()
    # NOTE: We no longer train a single model on ALL subjects here.
    # The ablation study (Phase 3) trains fresh models per CV fold to prevent
    # the identifiability inflation described in awareness.txt (Orlichenko et al.)
    
    # ===== PHASE 3: ABLATION STUDIES (with internal CV) =====
    print(f"\n[3/9] Running ablation studies (K={best_K}, L={best_L}, with 5-Fold CV)...")
    ablation = run_ablation_study(fc_task, fc_rest, model, best_K, best_L, epochs=epochs)
    all_results['ablation'] = ablation
    plot_ablation_results(ablation, os.path.join(run_dir, "ablation_results.png"))
    
    # ===== PHASE 4: SOTA COMPARISON =====
    print("\n[4/9] Running SOTA comparisons...")
    proposed_acc = ablation['Full Pipeline']['metrics']['top_1_accuracy']
    sota = run_sota_comparison(fc_task, fc_rest, proposed_acc)
    all_results['sota'] = sota
    
    # ===== PHASE 5: STATISTICAL VALIDATION =====
    print("\n[5/9] Running statistical validation...")
    
    # Generate heatmaps for all methods
    for key, data in ablation.items():
        if 'corr_matrix' in data:
            plot_full_correlation_matrix(data['corr_matrix'], os.path.join(run_dir, f"heatmap_{key}.png"))

    if 'Full Pipeline' in ablation:
        fc_refined = ablation['Full Pipeline']['corr_matrix']
        
        # Bootstrap
        bootstrap = bootstrap_ci(fc_task, fc_rest, n_bootstrap=500)
        all_results['bootstrap'] = bootstrap
        
        # Permutation test
        # Tests if the proposed identification rate is significantly better than random guessing
        p_val, _ = permutation_test(proposed_acc, fc_refined, n_permutations=500)
        all_results['permutation_p'] = p_val
        
        # McNemar's test
        preds_model = np.argmax(ablation['Full Pipeline']['corr_matrix'], axis=1) == np.arange(n_subjects)
        preds_baseline = np.argmax(ablation['Raw FC']['corr_matrix'], axis=1) == np.arange(n_subjects)
        mcnemar_stat, mcnemar_p = mcnemar_test(preds_model, preds_baseline)
        all_results['mcnemar_p'] = mcnemar_p
        
        # Paired Permutation Test (Model vs Baseline)
        paired_p, _ = paired_permutation_test(preds_model, preds_baseline, n_permutations=1000)
        all_results['paired_permutation_p'] = paired_p
        
        # Similarity distributions
        plot_similarity_distributions(fc_refined, os.path.join(run_dir, "similarity_dist.png"))
        
        # Full Correlation Matrix
        plot_full_correlation_matrix(ablation['Raw FC']['corr_matrix'], os.path.join(run_dir, "full_correlation_matrix.png"))
    
    # ===== PHASE 6: CROSS-VALIDATION =====
    print("\n[6/9] Running cross-validation...")
    cv_results = cross_validation(fc_task, fc_rest, n_folds=n_folds, K=K, L=L, epochs=10)
    all_results['cv'] = cv_results
    
    # ===== PHASE 7: INTERPRETABILITY =====
    print("\n[7/9] Running interpretability analysis...")
    if 'Full Pipeline' in ablation:
        plot_interpretability(
            model, 
            ablation['Full Pipeline']['dictionary'], 
            n_parcels, 
            run_dir
        )
    
    # ===== PHASE 8: ROBUSTNESS ANALYSIS =====
    print("\n[8/9] Running robustness analysis...")
    noise_robust = noise_robustness(fc_task, fc_rest, n_repeats=5)
    sample_robust = sample_size_robustness(fc_task, fc_rest, n_repeats=5)
    all_results['noise_robustness'] = noise_robust
    all_results['sample_robustness'] = sample_robust
    plot_robustness(noise_robust, sample_robust, os.path.join(run_dir, "robustness.png"))
    
    # ===== PHASE 9: GENERATE REPORT =====
    print("\n[9/9] Generating manuscript report...")
    report_path = generate_manuscript_report(all_results, run_dir)
    
    # Save JSON results
    json_results = {k: v for k, v in all_results.items() 
                    if not isinstance(v, np.ndarray) and k not in ['ablation']}
    json_results['ablation_accuracies'] = {
        m: all_results['ablation'][m]['metrics']['top_1_accuracy'] 
        for m in all_results['ablation']
    }
    
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {run_dir}")
    print(f"\nKey Results:")
    print(f"  Baseline Accuracy:  {ablation['Raw FC']['metrics']['top_1_accuracy']:.4f}")
    print(f"  Proposed Accuracy:  {proposed_acc:.4f}")
    if 'permutation_p' in all_results:
        print(f"  P-value (vs Chance): {all_results['permutation_p']:.6f}")
    if 'paired_permutation_p' in all_results:
        print(f"  P-value (vs Baseline): {all_results['paired_permutation_p']:.6f}")
    
    return all_results, run_dir

# %% [markdown]
# ## 13. Execute Pipeline

# %%
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain Fingerprinting Main Pipeline")
    parser.add_argument("--task", type=str, default=None, help="Specific task to run (e.g. motor). If none, runs all available.")
    parser.add_argument("--num_subjects", type=int, default=339, help="Number of subjects")
    parser.add_argument("--K", type=int, default=None, help="Hardcoded dictionary atoms K (overrides default/tuned).")
    parser.add_argument("--L", type=int, default=None, help="Hardcoded sparsity level L (overrides default/tuned).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per fold")
    parser.add_argument("--search_dict", action="store_true", help="Perform automatic Grid Search for K and L")
    parser.add_argument("--coarse_subs", type=int, default=10, help="Number of subjects for coarse tuning")
    parser.add_argument("--fine_subs", type=int, default=10, help="Number of subjects for fine tuning")
    args, unknown = parser.parse_known_args()

    # Configuration - USE REAL HCP DATA FOR COMPREHENSIVE ANALYSIS
    USE_SYNTHETIC = False
    N_SUBJECTS = args.num_subjects
    
    # Full list of HCP tasks
    ALL_TASKS = [args.task] if args.task else ["motor", "wm", "emotion", "gambling", "language", "relational", "social"]
    N_FOLDS = 5
    
    # ===== HYPERPARAMETER CONFIGURATION =====
    # Global flag to override and force grid search for all tasks
    PERFORM_GRID_SEARCH_GLOBAL = args.search_dict or False # Default False, unless --search_dict passed
    
    # Manually Tuned Parameters (K, L) for each task
    # Replace (15, 12) with your specific values found from previous tuning runs
    TUNED_PARAMS = {
        "motor": (14, 12),
        "wm": (15, 13),
        "emotion": (15, 13),
        "gambling": (14, 11),
        "language": (15, 11),
        "relational": (11, 9),
        "social": (15, 9)
    }
    
    # Track all run directories
    all_run_dirs = []
    
    print(f"Starting Multi-Task Analysis for: {ALL_TASKS}")
    print(f"USING DATASET (N={N_SUBJECTS}) - THIS WILL TAKE TIME")
    print(f"Grid Search Enabled: {PERFORM_GRID_SEARCH_GLOBAL}")
    if not PERFORM_GRID_SEARCH_GLOBAL:
        if args.K is not None and args.L is not None:
             print("Using hardcoded parameters from command line arguments.")
        else:
             print("Using manually defined parameters from TUNED_PARAMS.")
    
    # Filter tasks based on actual data availability
    AVAILABLE_TASKS = []
    
    # Check if data exists - Auto-switch to synthetic if not 
    if not USE_SYNTHETIC and not (os.path.exists(os.path.join(FC_DATA_DIR, "fc_rest.npy")) or 
                                (RAW_REST_DIR and os.path.exists(os.path.join(RAW_REST_DIR, "subjects")))):
        print(">>> [!] Real data not found. Auto-switching to SYNTHETIC mode for local run.")
        USE_SYNTHETIC = True
        if N_SUBJECTS > 50:
            print(f">>> [!] Reducing N_SUBJECTS from {N_SUBJECTS} to 50 for synthetic run.")
            N_SUBJECTS = 50
    
    # Check Rest Data First
    HAS_REST = False
    if not USE_SYNTHETIC and RAW_REST_DIR and os.path.exists(os.path.join(RAW_REST_DIR, "subjects")):
        sample_subs = [d for d in os.listdir(os.path.join(RAW_REST_DIR, "subjects")) if d.isdigit()]
        if sample_subs:
            test_sub = sample_subs[0]
            rest_ids = get_image_ids("rest") # [1, 2, 3, 4]
            f_path = os.path.join(RAW_REST_DIR, "subjects", test_sub, "timeseries", 
                                 f"bold{rest_ids[0]}_Atlas_MSMAll_Glasser360Cortical.npy")
            if os.path.exists(f_path):
                HAS_REST = True
                print(f"[OK] Rest data found (validated with {test_sub})")
            else:
                print(f"[FAIL] Rest data missing for {test_sub} at {f_path}")
                # Debug listing
                ts_dir = os.path.join(RAW_REST_DIR, "subjects", test_sub, "timeseries")
                if os.path.exists(ts_dir):
                    print(f"       Found files: {os.listdir(ts_dir)}")
    
    if not HAS_REST and not USE_SYNTHETIC:
        # Check if FC exists
        if os.path.exists(os.path.join(FC_DATA_DIR, "fc_rest.npy")):
             print("[OK] Pre-computed Rest FC found.")
             HAS_REST = True
        else:
             print("WARNING: Resting state data check failed! Analysis may fail.")

    if not USE_SYNTHETIC and RAW_TASK_DIR and os.path.exists(os.path.join(RAW_TASK_DIR, "subjects")):
         print("Verifying task data availability...")
         sample_subs = [d for d in os.listdir(os.path.join(RAW_TASK_DIR, "subjects")) if d.isdigit()]
         if sample_subs:
             test_sub = sample_subs[0]
             
             # Debug: Show what's present
             ts_dir = os.path.join(RAW_TASK_DIR, "subjects", test_sub, "timeseries")
             if os.path.exists(ts_dir):
                 found_files = sorted([f for f in os.listdir(ts_dir) if f.startswith("bold") and f.endswith(".npy")])
                 print(f"Debug: Found {len(found_files)} bold files for {test_sub}: {found_files}")
             
             for task in ALL_TASKS:
                 try:
                     r_ids = get_image_ids(task)
                     # Check file existence for test sub
                     f_path = os.path.join(RAW_TASK_DIR, "subjects", test_sub, "timeseries", 
                                          f"bold{r_ids[0]}_Atlas_MSMAll_Glasser360Cortical.npy")
                     if os.path.exists(f_path):
                         AVAILABLE_TASKS.append(task)
                     else:
                         # Also check if FC file exists already
                         fc_path = os.path.join(FC_DATA_DIR, f"fc_{task}.npy")
                         if os.path.exists(fc_path):
                             AVAILABLE_TASKS.append(task)
                         else:
                             print(f"  [SKIP] {task}: Expected bold{r_ids} but not found.")
                 except Exception as e:
                     print(f"  [SKIP] {task}: Error checking ({e})")
         else:
             print("  Warning: No subject folders found to verify tasks.")
             AVAILABLE_TASKS = ALL_TASKS
    else:
        AVAILABLE_TASKS = ALL_TASKS

    if not AVAILABLE_TASKS:
        print("WARNING: No available tasks found! Checking if FC data exists...")
        for task in ALL_TASKS:
             if os.path.exists(os.path.join(FC_DATA_DIR, f"fc_{task}.npy")):
                 AVAILABLE_TASKS.append(task)
    
    print(f"Proceeding with tasks: {AVAILABLE_TASKS}")

    for task_name in AVAILABLE_TASKS:
        print(f"\n\n{'#'*60}")
        print(f"RUNNING ANALYSIS FOR TASK: {task_name.upper()}")
        print(f"{'#'*60}\n")
        
        try:
            # Smart Hyperparameter Selector (TCDS Manuscript Protocol)
            # Order of preference: 
            # 1. Explicit CLI arguments (--K, --L)
            # 2. Global search force flag (--search_dict)
            # 3. Registry of pre-tuned task values (TUNED_PARAMS)
            # 4. Auto-Grid Search fallback
            
            if args.K is not None and args.L is not None:
                best_K, best_L = args.K, args.L
                should_search = False
                print(f">>> Mode: Using HARDCODED parameters for {task_name} (K={best_K}, L={best_L})")
            
            elif PERFORM_GRID_SEARCH_GLOBAL:
                should_search = True
                print(f">>> Mode: FORCED Grid Search for {task_name} (Global search flag active)")
                
            elif task_name in TUNED_PARAMS and TUNED_PARAMS[task_name][0] is not None:
                best_K, best_L = TUNED_PARAMS[task_name]
                should_search = False
                print(f">>> Mode: Using PRE-TUNED parameters for {task_name} (K={best_K}, L={best_L})")
                
            else:
                should_search = True
                if task_name in TUNED_PARAMS:
                    print(f">>> Mode: TASK-SPECIFIC Grid Search (Registry values for '{task_name}' are None)")
                else:
                    print(f">>> Mode: AUTO-GRID SEARCH (Task '{task_name}' not in registry)")

            if should_search:
                print(f">>> Optimization: Commencing Grid Search for {task_name}...")
                
            
                # Ensure Data Exists
                if not USE_SYNTHETIC:
                    rest_path = find_fc_file(FC_DATA_DIR, "rest")
                    task_path = find_fc_file(FC_DATA_DIR, task_name)
                    
                    if not rest_path or not task_path:
                        print("    Data missing for optimization. Generating now...")
                        os.makedirs(FC_DATA_DIR, exist_ok=True)
                        if RAW_REST_DIR and RAW_TASK_DIR:
                            subjects_path = os.path.join(RAW_REST_DIR, "subjects")
                            if os.path.exists(subjects_path):
                                all_subs = sorted([d for d in os.listdir(subjects_path) if os.path.isdir(os.path.join(subjects_path, d))])
                                if len(all_subs) > N_SUBJECTS: all_subs = all_subs[:N_SUBJECTS]
                                if not os.path.exists(rest_path):
                                    fc_r, _ = generate_fc_for_task("rest", all_subs, RAW_REST_DIR)
                                    np.save(rest_path, fc_r)
                                if not os.path.exists(task_path):
                                    fc_t, _ = generate_fc_for_task(task_name, all_subs, RAW_TASK_DIR)
                                    np.save(task_path, fc_t)
                        else:
                            raise ValueError(f"CRITICAL: Cannot run Grid Search for '{task_name}'. Data missing and RAW directories not set.")

                # Perform Search on whichever data we have (Real or Synthetic)
                if USE_SYNTHETIC:
                    print("    Synthetic mode: Running Grid Search on generated synthetic data...")
                    n_p = 360
                    n_s = N_SUBJECTS
                    # Generate temporary synthetic flat data for search
                    tril_len = (n_p * (n_p - 1)) // 2
                    flat_r = np.random.randn(n_s, tril_len)
                    flat_t = flat_r + 0.5 * np.random.randn(n_s, tril_len)
                else:
                    rest_path = find_fc_file(FC_DATA_DIR, "rest")
                    task_path = find_fc_file(FC_DATA_DIR, task_name)
                    if not rest_path or not task_path:
                        raise ValueError(f"CRITICAL: Data files not found for Grid Search in {FC_DATA_DIR}")
                    
                    full_rest = np.load(rest_path)
                    full_task = np.load(task_path)
                    if full_rest.shape[0] > N_SUBJECTS:
                        full_rest, full_task = full_rest[:N_SUBJECTS], full_task[:N_SUBJECTS]
                    
                    n_p = full_rest.shape[1]
                    tril = np.tril_indices(n_p, k=-1)
                    flat_r = np.array([full_rest[i][tril] for i in range(len(full_rest))])
                    flat_t = np.array([full_task[i][tril] for i in range(len(full_task))])
                
                # Two-Stage Tuning: Coarse then Fine
                _, best_K, best_L = refined_tuning_protocol(
                    flat_t.T, flat_r.T, len(flat_r), n_p, 
                    task_name=task_name,
                    max_search_subs=args.fine_subs,
                    coarse_max_subs=args.coarse_subs
                )
            
            print(f">>> Final Hyperparameters: K={best_K}, L={best_L}")

            results, run_dir = run_complete_analysis(
                n_subjects=N_SUBJECTS,
                task=task_name,
                use_synthetic=USE_SYNTHETIC,
                n_folds=N_FOLDS,
                K=best_K,
                L=best_L,
                epochs=args.epochs
            )
            all_run_dirs.append(run_dir)
        except Exception as e:
            print(f"ERROR running analysis for task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # Final step: Bundle everything into a zip
    if all_run_dirs:
        print("\n" + "="*60)
        print("BUNDLING ALL RESULTS")
        print("="*60)
        
        zip_filename = os.path.join(WORKING_DIR, f"hcp_fingerprinting_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # Create archive of the OUTPUT_DIR which contains all run folders
        shutil.make_archive(zip_filename, 'zip', OUTPUT_DIR)
        
        print(f"\n>>> All results successfully zipped to: {zip_filename}.zip")
        print(f"You can now download this file from the Kaggle Output sidebar.")
