#!/usr/bin/env python3
"""
CVAE + SDL Baseline for HCP Dataset
Lu et al., NeuroImage 295 (2024) 120651

Production-ready implementation for comparing against custom architectures.
Implements exact methodology from published paper with HCP S1200 data support.
"""

import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import orthogonal_mp
import datetime
import json
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
import gc

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "cvae_sdl_results")

# FC_DATA_DIR resolution: check multiple Kaggle dataset locations first
FC_DATA_DIR = None
REST_DATA_DIR = None  # Separate location for rest data if needed

if os.path.exists("/kaggle/input"):
    # Check common Kaggle dataset paths
    kaggle_candidates = [
        "/kaggle/input/datasets/ceoricky/fc-data",
        "/kaggle/input/fc-data",
        "/kaggle/input/hcp-fc-data",
    ]
    for cand in kaggle_candidates:
        if os.path.exists(cand):
            FC_DATA_DIR = cand
            break
    
    # Check for alternate HCP dataset with rest data
    if not FC_DATA_DIR:
        hcp_candidates = [
            "/kaggle/input/datasets/rickaryadas/hcp-dataset-s1200",
            "/kaggle/input/hcp-dataset-s1200",
        ]
        for cand in hcp_candidates:
            if os.path.exists(cand):
                FC_DATA_DIR = cand
                REST_DATA_DIR = cand
                break

# Fallback to local paths
if not FC_DATA_DIR:
    data_candidates = [
        "/data/FC_DATA",
        "../../../DATA/FC_DATA",
        "../../DATA/FC_DATA",
    ]
    for cand in data_candidates:
        if os.path.exists(cand):
            FC_DATA_DIR = cand
            break

# Use current directory as last resort
if not FC_DATA_DIR:
    FC_DATA_DIR = "./FC_DATA"

if not REST_DATA_DIR:
    REST_DATA_DIR = FC_DATA_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*70}")
print(f"Device: {DEVICE}")
print(f"FC Data Directory: {FC_DATA_DIR}")
if REST_DATA_DIR != FC_DATA_DIR:
    print(f"REST Data Directory: {REST_DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"{'='*70}")

# ==========================================
# HCP STATE CONFIGURATION
# ==========================================
# 9 fMRI states from HCP S1200 release
HCP_STATES = {
    "rest1": "rfMRI_REST1_LR",
    "rest2": "rfMRI_REST2_LR", 
    "gambling": "tfMRI_GAMBLING_LR",
    "motor": "tfMRI_MOTOR_LR",
    "wm": "tfMRI_WM_LR",
    "emotion": "tfMRI_EMOTION_LR",
    "language": "tfMRI_LANGUAGE_LR",
    "relational": "tfMRI_RELATIONAL_LR",
    "social": "tfMRI_SOCIAL_LR"
}
ALL_STATES = list(HCP_STATES.keys())
STATE_IDX = {state: idx for idx, state in enumerate(ALL_STATES)}
print(f"\nHCP States ({len(ALL_STATES)}): {', '.join(ALL_STATES)}")

# ==========================================
# CONDITIONAL VAE (Lu et al. Architecture)
# ==========================================
class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder with fully-connected architecture.
    
    Exact implementation from Lu et al., NeuroImage 295 (2024):
    - Architecture: d-3000-2000-1000-500-100-500-1000-2000-3000-d
    - d = 64,620 (lower-triangular FC from 360 parcels)
    - Condition: 9-dimensional one-hot for fMRI state
    - Latent: 100 dimensions
    - Activation: Tanh throughout
    """
    def __init__(self, n_features=64620, n_states=9, latent_dim=100):
        super(ConditionalVAE, self).__init__()
        self.n_features = n_features
        self.n_states = n_states
        self.latent_dim = latent_dim
        
        # Encoder: input + condition -> features -> latent parameters
        self.encoder = nn.Sequential(
            nn.Linear(n_features + n_states, 3000),
            nn.BatchNorm1d(3000),
            nn.Tanh(),
            nn.Linear(3000, 2000),
            nn.BatchNorm1d(2000),
            nn.Tanh(),
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.Tanh(),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.Tanh()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(500, latent_dim)
        self.fc_logvar = nn.Linear(500, latent_dim)
        
        # Decoder: latent + condition -> features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_states, 500),
            nn.BatchNorm1d(500),
            nn.Tanh(),
            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.Tanh(),
            nn.Linear(1000, 2000),
            nn.BatchNorm1d(2000),
            nn.Tanh(),
            nn.Linear(2000, 3000),
            nn.BatchNorm1d(3000),
            nn.Tanh(),
            nn.Linear(3000, n_features),
            nn.Tanh()
        )

    def encode(self, x, c):
        """Encode with condition."""
        x_cond = torch.cat([x, c], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        mu = torch.clamp(mu, min=-1000.0, max=1000.0)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decode with condition."""
        z_cond = torch.cat([z, c], dim=1)
        return self.decoder(z_cond)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HCPDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that avoids big tensor copies."""
    def __init__(self, fc_data, subject_indices, sorted_states):
        self.fc_data = fc_data
        self.indices = subject_indices
        self.states = sorted_states
        self.n_subjects = len(subject_indices)
        self.n_states = len(sorted_states)
        
    def __len__(self):
        return self.n_subjects * self.n_states
    
    def __getitem__(self, idx):
        # Calculate subject and state indices from flat index
        state_idx = idx // self.n_subjects
        sub_local_idx = idx % self.n_subjects
        actual_sub_idx = self.indices[sub_local_idx]
        
        state = self.states[state_idx]
        
        # Load data (already in RAM)
        x = self.fc_data[state][actual_sub_idx]
        
        # One-hot condition
        c = np.zeros(self.n_states, dtype=np.float32)
        c[state_idx] = 1.0
        
        return torch.from_numpy(x).float(), torch.from_numpy(c).float()


def clean_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  [INFO] GPU cache cleared.")


# ==========================================
# LOSS & METRICS
# ==========================================
def cvae_loss(recon_x, x, mu, logvar, beta=1.0):
    """CVAE loss: Reconstruction + KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def calculate_accuracy(corr_matrix):
    """Top-1 identification accuracy."""
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0)
    n_rows, n_cols = corr_matrix.shape
    n = min(n_rows, n_cols)
    if n == 0:
        return 0.0
    correct = sum(1 for i in range(n) if np.argmax(corr_matrix[i, :]) == i)
    return correct / n


def compute_all_metrics(corr_matrix):
    """Comprehensive metrics matching main pipeline."""
    n_rows, n_cols = corr_matrix.shape
    n = min(n_rows, n_cols)
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0)
    
    if n == 0:
        return {
            'top_1_accuracy': 0.0,
            'top_5_accuracy': 0.0,
            'top_10_accuracy': 0.0,
            'mean_rank': 0.0,
            'mrr': 0.0,
            'differential_identifiability': 0.0
        }
    
    # Accuracy metrics
    top_1 = calculate_accuracy(corr_matrix)
    top_5 = np.mean([1 if i < n_cols and i in np.argsort(corr_matrix[i, :])[-5:] else 0 for i in range(n)])
    top_10 = np.mean([1 if i < n_cols and i in np.argsort(corr_matrix[i, :])[-10:] else 0 for i in range(n)])
    
    # Rank metrics
    ranks = []
    for i in range(n):
        if i >= n_cols:
            ranks.append(n_cols)
            continue
        sorted_indices = np.argsort(corr_matrix[i, :])[::-1]
        match_pos = np.where(sorted_indices == i)[0]
        if len(match_pos) > 0:
            ranks.append(match_pos[0] + 1)
        else:
            ranks.append(n_cols)
            
    mean_rank = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])
    
    # Differential identifiability
    self_corr = np.diag(corr_matrix[:n, :n])
    other_mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[:n, :n][other_mask]
    diff_id = np.mean(self_corr) - np.mean(other_corr)
    
    return {
        'top_1_accuracy': top_1,
        'top_5_accuracy': top_5,
        'top_10_accuracy': top_10,
        'mean_rank': mean_rank,
        'mrr': mrr,
        'differential_identifiability': diff_id
    }


# ==========================================
# SPARSE DICTIONARY LEARNING (K-SVD + OMP)
# ==========================================
def omp_sparse_coding(Y, D, L):
    """Orthogonal Matching Pursuit for sparse coding."""
    return orthogonal_mp(D, Y, n_nonzero_coefs=L)


def update_dictionary(Y, D, X):
    """Update dictionary atoms using SVD."""
    for k in range(D.shape[1]):
        idx = np.nonzero(X[k, :])[0]
        if len(idx) == 0:
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue
        residual = Y[:, idx] - np.dot(D, X[:, idx]) + np.outer(D[:, k], X[k, idx])
        U, s, Vt = np.linalg.svd(residual, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, idx] = s[0] * Vt[0, :]
    return D, X


def k_svd(Y, K, L, n_iter=10, verbose=True, random_state=None):
    """K-SVD algorithm for dictionary learning."""
    if random_state is not None:
        np.random.seed(random_state)
    m, n = Y.shape
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    iterator = tqdm(range(n_iter), desc="K-SVD", disable=not verbose) if verbose else range(n_iter)
    for _ in iterator:
        X = omp_sparse_coding(Y, D, L)
        D, X = update_dictionary(Y, D, X)
    return D, X


# ==========================================
# DATA LOADING
# ==========================================
def find_fc_file(directory, task_name):
    """Find FC file for a given task name with flexible naming conventions.
    
    Matches the kaggle_brainfingerprinting.py pattern for consistency.
    """
    if not directory or not os.path.exists(directory):
        return None
    
    # Common naming patterns
    task_patterns = [
        f"fc_{task_name}.npy",
        f"{task_name}_fc.npy",
        f"{task_name}.npy"
    ]
    
    # Special handling for rest
    if task_name.lower() == 'rest':
        task_patterns.extend(["rest_fc.npy", "fc_rest.npy", "REST.npy"])
    elif task_name.lower().startswith('rest'):
        # For rest1, rest2, also check combined rest file
        task_patterns.extend(["fc_rest.npy", "rest_fc.npy"])
    
    try:
        for f in os.listdir(directory):
            if any(p.lower() == f.lower() for p in task_patterns):
                return os.path.join(directory, f)
    except (PermissionError, OSError):
        pass
    
    return None


def load_hcp_fc_data(fc_dir, states, max_subjects=None):
    """
    Load HCP FC matrices for specified states.
    Follows kaggle_brain_fingerprinting.py pattern.
    
    Returns:
        fc_data: Dictionary {state: (n_subjects, 64620)}
        n_loaded: Number of subjects loaded
    """
    fc_data = {}
    
    if not os.path.exists(fc_dir):
        print(f"\n[ERROR] FC directory not found: {fc_dir}")
        return {}, 0
    
    print(f"\nLoading FC data from: {fc_dir}")
    
    # First, check for combined rest file
    combined_rest_file = find_fc_file(fc_dir, "rest")
    combined_rest_data = None
    
    if combined_rest_file:
        print(f"  Found combined rest file: {os.path.basename(combined_rest_file)}")
        combined_rest_data = np.load(combined_rest_file)
        if max_subjects and combined_rest_data.shape[0] > max_subjects:
            combined_rest_data = combined_rest_data[:max_subjects]
    
    for state in states:
        # Try to find individual file first
        individual_file = None
        if state.startswith('rest'):
            # For rest states, look for combined file or individual files
            if combined_rest_data is not None:
                individual_file = "combined"
            else:
                individual_file = find_fc_file(fc_dir, state)
        else:
            # For task states, look for specific state file
            individual_file = find_fc_file(fc_dir, state)
        
        if individual_file == "combined":
            data = combined_rest_data.copy()
            print(f"  Loading {state}: {os.path.basename(combined_rest_file)} (from combined)", end="")
        elif individual_file:
            print(f"  Loading {state}: {os.path.basename(individual_file)}", end="")
            data = np.load(individual_file)
            if max_subjects and data.shape[0] > max_subjects:
                data = data[:max_subjects]
        else:
            print(f"  [SKIP] No data found for {state}")
            continue
        
        # Vectorize if 3D (n_subjects, n_parcels, n_parcels) -> (n_subjects, n_features)
        if data.ndim == 3:
            n_subjects, n_parcels, _ = data.shape
            n_features = (n_parcels * (n_parcels - 1)) // 2
            vectorized = np.zeros((n_subjects, n_features))
            # Extract lower triangle indices
            lower_tri_idx = np.tril_indices(n_parcels, k=-1)
            for i in range(n_subjects):
                vectorized[i, :] = data[i, lower_tri_idx[0], lower_tri_idx[1]]
            data = vectorized
        
        # Clip to [-1, 1] range
        fc_data[state] = np.clip(np.nan_to_num(data, nan=0.0), -1.0, 1.0)
        print(f" -> {fc_data[state].shape}")
    
    if len(fc_data) == 0:
        print("[ERROR] No FC data loaded!")
        return {}, 0
    
    n_loaded = fc_data[list(fc_data.keys())[0]].shape[0]
    return fc_data, n_loaded


# ==========================================
# MAIN PIPELINE
# ==========================================
def run_cvae_sdl_baseline(num_subjects=339, epochs=50, learning_rate=1e-4, 
                          lr_decay=0.9991, K=50, L=20, skip_sdl=False):
    """
    CVAE+SDL Baseline for HCP Dataset.
    
    Production pipeline comparing against custom architectures.
    
    Args:
        num_subjects: Max subjects to use (default 339, max 997)
        epochs: Training epochs (default 50, paper uses 300)
        learning_rate: SGD initial learning rate
        lr_decay: Learning rate exponential decay factor
        K: K-SVD dictionary atoms
        L: OMP sparsity level
    """
    print("\n" + "="*80)
    print(" CVAE+SDL BASELINE - HCP Dataset Evaluation")
    print(" Lu et al., NeuroImage 295 (2024) 120651")
    print("="*80)
    
    global DEVICE
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"cvae_sdl_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Configuration
    n_parcels = 360
    n_features = (n_parcels * (n_parcels - 1)) // 2  # 64,620
    
    print(f"\n[CONFIG]")
    print(f"  Subjects: {num_subjects}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate} (decay: {lr_decay})")
    print(f"  K-SVD: K={K}, L={L} (skip_sdl={skip_sdl})")
    print(f"  Output: {run_dir}")
    
    # ==========================================
    # LOAD DATA
    # ==========================================
    print(f"\n[1/7] Loading HCP FC data...")
    fc_data, n_subjects = load_hcp_fc_data(FC_DATA_DIR, ALL_STATES, max_subjects=num_subjects)
    
    if len(fc_data) < len(ALL_STATES):
        print(f"\n[WARNING] Only {len(fc_data)}/{len(ALL_STATES)} states available")
        if len(fc_data) < 2:
            print("[ERROR] Need at least 2 states. Aborting.")
            return None, None
    
    # Truncate to common size
    n_subjects = min([fc_data[s].shape[0] for s in fc_data.keys()])
    for state in fc_data:
        fc_data[state] = fc_data[state][:n_subjects]
    
    print(f"  Loaded {n_subjects} subjects, {len(fc_data)} states")
    
    # ==========================================
    # TRAIN-TEST SPLIT (50-50)
    # ==========================================
    print(f"\n[2/7] Creating 50-50 train-test split...")
    n_train = n_subjects // 2
    indices = np.random.RandomState(42).permutation(n_subjects)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
    
    # ==========================================
    # CVAE TRAINING
    # ==========================================
    print(f"\n[3/7] Training CVAE ({epochs} epochs, SGD lr={learning_rate})...")
    
    # Memory optimization: clear previous runs
    clean_memory()
    
    # Prevent OOM by deleting global 'model' or 'cvae' if they exist in notebook
    # This is a common issue in Jupyter environments
    for var in ['model', 'cvae', 'results']:
        if var in globals():
            print(f"  [INFO] Found existing variable '{var}', deleting to free memory...")
            del globals()[var]
    clean_memory()
    
    try:
        cvae = ConditionalVAE(n_features=n_features, n_states=len(fc_data), latent_dim=100).to(DEVICE)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "!"*60)
            print("  [ERROR] CUDA Out of Memory during model initialization.")
            print("  Attempting to initialize on CPU instead...")
            print("!"*60 + "\n")
            clean_memory()
            DEVICE = "cpu"
            cvae = ConditionalVAE(n_features=n_features, n_states=len(fc_data), latent_dim=100).to(DEVICE)
        else:
            raise e
            
    print(f"  Model parameters: {cvae.count_parameters():,}")
    
    optimizer = optim.SGD(cvae.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    # Memory-efficient Data Loading
    sorted_states = sorted(fc_data.keys())
    train_dataset = HCPDataset(fc_data, train_idx, sorted_states)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=(DEVICE=="cuda"))
    
    # Training loop
    cvae.train()
    losses = []
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0
        n_batches = 0
        
        for batch_x, batch_c in dataloader:
            batch_x, batch_c = batch_x.to(DEVICE), batch_c.to(DEVICE)
            optimizer.zero_grad()
            
            recon, mu, logvar = cvae(batch_x, batch_c)
            loss = cvae_loss(recon, batch_x, mu, logvar, beta=1.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    # ==========================================
    # COMPUTE RESIDUALS & LEARN DICTIONARY
    # ==========================================
    print(f"\n[4/7] Computing residuals and learning K-SVD dictionary...")
    
    cvae.eval()
    residuals_train = {}
    
    with torch.no_grad():
        for state_idx, state in enumerate(sorted(fc_data.keys())):
            state_vectors = fc_data[state][train_idx]
            x_state = torch.tensor(state_vectors, dtype=torch.float32).to(DEVICE)
            
            c_state = torch.zeros((len(train_idx), len(fc_data)), dtype=torch.float32).to(DEVICE)
            c_state[:, state_idx] = 1
            
            recon, _, _ = cvae(x_state, c_state)
            residuals_train[state] = (x_state - recon).cpu().numpy()
    
    # Learn dictionary
    if skip_sdl:
        print(f"  [SKIP] Skipping Dictionary Learning as skip_sdl=True")
        D = None
    else:
        # Stack residuals from all states: (n_subjects, n_features) per state
        # Result after vstack: (n_subjects*n_states, n_features) 
        # Transpose to: (n_features, n_subjects*n_states) for K-SVD
        Y_train = np.vstack([residuals_train[s] for s in sorted(fc_data.keys())]).T
        D, _ = k_svd(Y_train, K=K, L=L, n_iter=10, verbose=True, random_state=42)
        print(f"  Dictionary learned: K={K} atoms, L={L} sparsity")
        print(f"  Dictionary shape: D = ({D.shape[0]}, {D.shape[1]})")
    
    # ==========================================
    # COMPUTE TEST RESIDUALS
    # ==========================================
    print(f"\n[5/7] Computing test residuals...")
    
    residuals_test = {}
    with torch.no_grad():
        for state_idx, state in enumerate(sorted(fc_data.keys())):
            state_vectors = fc_data[state][test_idx]
            x_state = torch.tensor(state_vectors, dtype=torch.float32).to(DEVICE)
            
            c_state = torch.zeros((len(test_idx), len(fc_data)), dtype=torch.float32).to(DEVICE)
            c_state[:, state_idx] = 1
            
            recon, _, _ = cvae(x_state, c_state)
            residuals_test[state] = (x_state - recon).cpu().numpy()
    
    # ==========================================
    # EVALUATION
    # ==========================================
    # Pre-compute representations for all test scans
    print(f"\n[5.5/7] Coding test representations (skip_sdl={skip_sdl})...")
    test_representations = {}
    for state in tqdm(sorted_states, desc="Representations"):
        Y_test = residuals_test[state].T
        if skip_sdl:
            test_representations[state] = Y_test
        else:
            test_representations[state] = omp_sparse_coding(Y_test, D, L=L)
            
    print(f"\n[6/7] Evaluating on state combinations...")
    
    results_by_pair = {}
    
    for train_state in sorted_states:
        for test_state in sorted_states:
            if train_state == test_state:
                continue
            
            # Evaluate identification on HELD-OUT subjects (test_idx)
            # Match Scan 1 (train_state) with Scan 2 (test_state) for the same subjects
            X_gallery = test_representations[train_state] # (dim, n_test)
            X_probe = test_representations[test_state]     # (dim, n_test)
            
            # Correlation matrix (n_test, n_test)
            corr = np.corrcoef(X_probe.T, X_gallery.T)[:len(test_idx), len(test_idx):]
            metrics = compute_all_metrics(corr)
            
            key = f"{train_state}->{test_state}"
            results_by_pair[key] = metrics
    
    # ==========================================
    # RESULTS SUMMARY
    # ==========================================
    print(f"\n[7/7] Summarizing results...")
    
    print("\n" + "="*80)
    print(" RESULTS SUMMARY")
    print("="*80)
    
    # Best results
    best_acc = max([m['top_1_accuracy'] for m in results_by_pair.values()])
    best_pair = [k for k, m in results_by_pair.items() if m['top_1_accuracy'] == best_acc][0]
    
    print(f"\nBest accuracy: {best_acc:.4f} ({best_pair})")
    
    # Average results
    avg_top1 = np.mean([m['top_1_accuracy'] for m in results_by_pair.values()])
    avg_top5 = np.mean([m['top_5_accuracy'] for m in results_by_pair.values()])
    avg_mrr = np.mean([m['mrr'] for m in results_by_pair.values()])
    avg_diff_id = np.mean([m['differential_identifiability'] for m in results_by_pair.values()])
    
    print(f"\nAverage metrics across all {len(results_by_pair)} state pairs:")
    print(f"  Top-1 Accuracy:     {avg_top1:.4f}")
    print(f"  Top-5 Accuracy:     {avg_top5:.4f}")
    print(f"  Mean Reciprocal Rank: {avg_mrr:.4f}")
    print(f"  Differential ID:    {avg_diff_id:.4f}")
    
    # Sample pairs
    print(f"\nSample results:")
    for pair in list(results_by_pair.keys())[:5]:
        m = results_by_pair[pair]
        print(f"  {pair}: Top-1={m['top_1_accuracy']:.4f}, MRR={m['mrr']:.4f}")
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print(f"\nSaving results to {run_dir}...")
    
    # Save metrics as JSON
    results_json = {
        'config': {
            'num_subjects': n_subjects,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'epochs': epochs,
            'learning_rate': learning_rate,
            'K': K,
            'L': L,
            'skip_sdl': skip_sdl,
            'timestamp': timestamp
        },
        'training': {'final_loss': float(losses[-1])},
        'average_metrics': {
            'top_1_accuracy': float(avg_top1),
            'top_5_accuracy': float(avg_top5),
            'mrr': float(avg_mrr),
            'differential_identifiability': float(avg_diff_id)
        },
        'best_pair': {
            'pair': best_pair,
            'accuracy': float(best_acc)
        },
        'all_pairs': {k: {kk: float(vv) if isinstance(vv, (int, np.integer, float, np.floating)) else vv 
                          for kk, vv in v.items()} 
                      for k, v in results_by_pair.items()}
    }
    
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save training losses
    np.save(os.path.join(run_dir, "training_losses.npy"), np.array(losses))
    
    # Save model
    torch.save(cvae.state_dict(), os.path.join(run_dir, "cvae_model.pt"))
    
    # Save dictionary
    if D is not None:
        np.save(os.path.join(run_dir, "dictionary_D.npy"), D)
        print(f"  ✓ dictionary_D.npy")
    
    print(f"  ✓ results.json")
    print(f"  ✓ training_losses.npy")
    print(f"  ✓ cvae_model.pt")
    print(f"  ✓ dictionary_D.npy")
    
    print("\n" + "="*80)
    print(f"Baseline complete! Results saved to:\n  {run_dir}")
    print("="*80 + "\n")
    
    return results_json, cvae


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CVAE+SDL Baseline for HCP")
    parser.add_argument("--subjects", type=int, default=339, help="Number of subjects")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--K", type=int, default=50, help="Dictionary atoms")
    parser.add_argument("--L", type=int, default=20, help="Sparsity level")
    parser.add_argument("--skip-sdl", action="store_true", help="Skip Sparse Dictionary Learning")
    
    args, _ = parser.parse_known_args()
    
    results, model = run_cvae_sdl_baseline(
        num_subjects=args.subjects,
        epochs=args.epochs,
        learning_rate=args.lr,
        K=args.K,
        L=args.L,
        skip_sdl=args.skip_sdl
    )
    
    if results is not None and model is not None:
        print("\n✓ Baseline generation complete!")
        print(f"✓ Ready for comparison with custom architecture")
    else:
        print("\n✗ Baseline generation failed. Please check data paths and try again.")
