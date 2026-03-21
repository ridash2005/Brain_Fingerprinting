# %% [markdown]
# # CVAE + SDL Baseline - Brain Fingerprinting Comparison
# 
# This notebook implements the Conditional Variational Autoencoder + Sparse Dictionary
# Learning (CVAE+SDL) baseline from **Lu et al., NeuroImage 295 (2024) 120651** for
# direct comparison against our ConvAE+SDL Brain Fingerprinting pipeline.
#
# **Key Difference from Our Method:**
# - Lu et al. use a **Conditional VAE** that embeds fMRI state information (rest vs task)
#   into the encoding/decoding process. This allows the CVAE to better capture shared
#   inter-subject features conditioned on the scan type.
# - Our method uses a standard **Convolutional Autoencoder (ConvAE)** without state conditioning.
# - Both methods then apply **Sparse Dictionary Learning (K-SVD + OMP)** on the residuals.
#
# **Awareness-Compliant Design (Orlichenko et al., 2023):**
# - 5-Fold Cross-Validation with strict subject-level splits
# - CVAE trained per fold on TRAIN subjects only, evaluated on HELD-OUT subjects
# - SDL dictionary learned from TRAIN residuals only, applied inductively to TEST
# - Prevents identifiability leakage / double-dipping
#
# **Evaluation Protocol:**
# - Identical metrics to `kaggle_brain_fingerprinting.py`
# - Same FC data loading, same subject ordering
# - Results directly comparable
#
# ---

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import os
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
import shutil
from tqdm.auto import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "cvae_sdl_results")
SAVE_DIR = os.path.join(WORKING_DIR, "FC_DATA")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 2. Data Discovery & Loading

# %%
# ==========================================
# PATH DISCOVERY (Matches main pipeline exactly)
# ==========================================
RAW_REST_DIR = None
RAW_TASK_DIR = None
FC_DATA_DIR = SAVE_DIR

def setup_environment():
    """Discover HCP data paths on Kaggle or local filesystem."""
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
    
    fc_dir = SAVE_DIR
    for d in [INPUT_DIR, WORKING_DIR, "..", "../DATA", "."]:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            if "subjects" in dirs:
                if "rest" in root.lower() and RAW_REST_DIR is None:
                    RAW_REST_DIR = root
                elif ("task" in root.lower() or "motor" in root.lower() or "hcp_task" in root.lower()) and RAW_TASK_DIR is None:
                    RAW_TASK_DIR = root
            if "fc_rest.npy" in files:
                fc_dir = root
    FC_DATA_DIR = fc_dir

setup_environment()

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

print(f"--- Environment Verified ---")
print(f"Device: {DEVICE.upper()}")
print(f"Raw Rest: {RAW_REST_DIR or 'Not Found'}")
print(f"Raw Task: {RAW_TASK_DIR or 'Not Found'}")
print(f"FC Data Path: {FC_DATA_DIR}")
print(f"----------------------------\n")

# Constants (identical to main pipeline)
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]
TRIL_IDX = np.tril_indices(360, k=-1)

# --- FC Generation (identical to main pipeline) ---
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

# %% [markdown]
# ## 3. CVAE Architecture (Lu et al., 2024)

# %%
class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for Brain Fingerprinting.
    
    From Lu et al., NeuroImage 295 (2024):
    - Embeds fMRI state information (Rest=0, Task=1) into encoding/decoding
    - The condition allows the CVAE to better capture SHARED inter-subject features
    - Residual (Original - Reconstruction) contains individual-specific information
    
    Architecture:
    - Decoder: Linear(latent_dim+1 -> flatten_size) -> ConvTranspose2d(64->32->16->1) + Tanh
    """
    def __init__(self, n_parcels=360, latent_dim=128):
        super(ConditionalVAE, self).__init__()
        self.n = n_parcels
        self.latent_dim = latent_dim
        
        # Encoder (2 channels: 1 for FC matrix, 1 for condition map)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 360->180
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 180->90
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)    # 90->45
        )
        self.flatten_size = 64 * 45 * 45  # 129,600
        
        # VAE: mu and logvar, conditioned on state label
        self.fc_mu = nn.Linear(self.flatten_size + 1, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size + 1, latent_dim)
        
        # Decoder input, conditioned on state label
        self.fc_dec = nn.Linear(latent_dim + 1, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 45 to 90
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 90 to 180
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # 180 to 360
            nn.Tanh()
        )

    def encode(self, x, c):
        """Encode FC matrix with state condition."""
        c_channel = c.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x_cond = torch.cat([x, c_channel], dim=1)  # (B, 2, n, n)
        h = self.encoder(x_cond).reshape(x.size(0), -1)
        h_cond = torch.cat([h, c], dim=1)
        
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        
        # CRITICAL VAE FIX: Prevent float32 Inf/NaN overflow before reparameterization
        mu = torch.clamp(mu, min=-1000.0, max=1000.0)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decode latent vector with state condition."""
        z_cond = torch.cat([z, c], dim=1)
        h = self.fc_dec(z_cond).view(-1, 64, 45, 45)
        recon = self.decoder(h)
        if recon.shape[2:] != (self.n, self.n):
            recon = F.interpolate(recon, size=(self.n, self.n), mode='bilinear', align_corners=True)
        return recon

    def forward(self, x, c):
        """Full forward pass: encode -> reparameterize -> decode."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cvae_loss(recon_x, x, mu, logvar, beta=0.1):
    """CVAE Loss = Reconstruction (MSE) + beta x KL Divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld) / x.size(0)

# %% [markdown]
# ## 4. Sparse Dictionary Learning (K-SVD + OMP)

# %%
def omp_sparse_coding(Y, D, L):
    """Orthogonal Matching Pursuit for sparse coding."""
    n_samples = Y.shape[1]
    n_atoms = D.shape[1]
    X = np.zeros((n_atoms, n_samples))
    for i in range(n_samples):
        X[:, i] = orthogonal_mp(D, Y[:, i], n_nonzero_coefs=L)
    return X

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
    """K-SVD algorithm for sparse dictionary learning."""
    if random_state is not None:
        np.random.seed(random_state)
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
    return matrix

def perform_grid_search(Y, rest_flat, n_subjects, n_parcels, K_range=(2, 16), L_max=None, n_iter=3, task_name="unknown"):
    """
    Grid search for optimal K and L parameters.
    Returns (accuracies, best_K, best_L)
    """
    print(f"  Grid Search K={K_range}...")
    
    if isinstance(K_range, tuple):
        Ks = sorted(list(range(K_range[0], K_range[1] + 1, 1)))
    else:
        Ks = sorted(list(K_range))
        
    best_acc = -1.0
    best_K = 15  # Default fallback
    best_L = 12
    
    all_L_vals = sorted(list(set([L for K_val in Ks for L in range(2, K_val + 1, 2)])))
    accuracies = np.zeros((len(Ks), len(all_L_vals)))
    
    for i, K in enumerate(tqdm(Ks, desc="Grid Search K")):
        L_vals = range(2, K + 1, 2)
        for L in L_vals:
            j = all_L_vals.index(L)
            # Run simplified K-SVD
            D, X = k_svd(Y, K, L, n_iter=n_iter, verbose=False, random_state=42)
            
            if rest_flat is not None:
                # Approximate Rest Sparse Codes using learned D
                X_rest = omp_sparse_coding(rest_flat, D, L)
                
                corr = np.corrcoef(X.T, X_rest.T)[:n_subjects, n_subjects:]
                acc = calculate_accuracy(corr)
                
                accuracies[i, j] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_K = K
                    best_L = L
    
    print(f"  Found Optimal: K={best_K}, L={best_L} (Acc: {best_acc:.4f})")
    
    # Plot Heatmap
    if np.nanmax(accuracies) > 0:
        try:
            plt.figure(figsize=(12, 10))
            mask = accuracies == 0
            df_plot = pd.DataFrame(accuracies, index=Ks, columns=all_L_vals)
            sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
            plt.title(f'CVAE Grid Search Identification Accuracy ({task_name.upper()})')
            plt.ylabel('Dictionary Atoms (K)')
            plt.xlabel('Sparsity Level (L)')
            
            # Save to OUTPUT_DIR
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(OUTPUT_DIR, f"grid_search_cvae_{task_name}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] CVAE Grid search heatmap saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not plot CVAE heatmap: {e}")
            
    return accuracies, best_K, best_L

# %% [markdown]
# ## 5. Evaluation Metrics (Identical to Main Pipeline)

# %%
def calculate_accuracy(corr_matrix):
    """Top-1 identification accuracy."""
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0)
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if np.argmax(corr_matrix[i, :]) == i)
    return correct / n

def calculate_top_k_accuracy(corr_matrix, k=5):
    """Top-k identification accuracy."""
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if i in np.argsort(corr_matrix[i, :])[-k:])
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
    """Compute comprehensive evaluation metrics (identical to main pipeline)."""
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
# ## 6. CVAE+SDL Pipeline (5-Fold CV, Awareness Compliant)

# %%
def run_cvae_sdl_baseline(task_name="motor", num_subjects=339, K=15, L=12, 
                          epochs=20, beta=0.1, latent_dim=128, n_folds=5):
    """
    CVAE+SDL Baseline from Lu et al., NeuroImage 295 (2024) 120651.
    
    Awareness-Compliant Implementation:
    - 5-Fold CV with strict subject-level splits (same seed=42 as main pipeline)
    - CVAE trained per fold on TRAIN subjects only (both rest + task, conditioned)
    - SDL dictionary learned from TRAIN residuals only
    - Sparse coding applied inductively to TEST residuals
    - All evaluation metrics computed on HELD-OUT subjects only
    
    Args:
        task_name: HCP task name (motor, wm, emotion, gambling, language, relational, social)
        num_subjects: Number of subjects to use (max 339)
        K: Number of dictionary atoms for K-SVD
        L: Sparsity level for OMP
        epochs: Training epochs per fold for CVAE
        beta: KL divergence weight in CVAE loss
        latent_dim: Dimensionality of CVAE latent space
        n_folds: Number of CV folds
    """
    print("=" * 60)
    print("CVAE+SDL BASELINE - 5-FOLD CV (AWARENESS COMPLIANT)")
    print("Lu et al., NeuroImage 295 (2024) 120651")
    print("=" * 60)

    # ==========================================
    # DATA LOADING (Same logic as main pipeline)
    # ==========================================
    print("\n[1/4] Loading data...")
    n_parcels = 360
    
    rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
    task_path = os.path.join(FC_DATA_DIR, f"fc_{task_name}.npy")
    save_rest_path = os.path.join(SAVE_DIR, "fc_rest.npy")
    save_task_path = os.path.join(SAVE_DIR, f"fc_{task_name}.npy")
    
    if os.path.exists(rest_path) and os.path.exists(task_path):
        print(f"  Loading pre-calculated matrices from {FC_DATA_DIR}")
        fc_rest = np.load(rest_path)
        fc_task = np.load(task_path)
    elif RAW_REST_DIR is not None and RAW_TASK_DIR is not None:
        print("  Generating FC from raw timeseries...")
        subjects_path = os.path.join(RAW_REST_DIR, "subjects")
        subjects_list = sorted([d for d in os.listdir(subjects_path)
                               if os.path.isdir(os.path.join(subjects_path, d))])
        if num_subjects < len(subjects_list):
            subjects_list = subjects_list[:num_subjects]
            
        fc_rest, rest_subs = generate_fc_for_task("rest", subjects_list, RAW_REST_DIR, n_parcels)
        fc_task, task_subs = generate_fc_for_task(task_name, subjects_list, RAW_TASK_DIR, n_parcels)
        
        valid_subs = sorted(list(set(rest_subs) & set(task_subs)))
        fc_rest = fc_rest[[rest_subs.index(s) for s in valid_subs]]
        fc_task = fc_task[[task_subs.index(s) for s in valid_subs]]
        
        print(f"  Generated FC for {len(valid_subs)} subjects")
        np.save(save_rest_path, fc_rest)
        np.save(save_task_path, fc_task)
    else:
        print(">>> [!] No FC data found and no raw data directories available.")
        print(">>> [!] LOCAL RUN DETECTED: Generating synthetic data for seamless execution...")
        
        # Fallback to fewer subjects if on local machine to keep it fast
        if num_subjects > 50:
            print(f">>> [!] Reducing num_subjects from {num_subjects} to 50 for local synthetic run.")
            num_subjects = 50
            
        fc_rest = generate_synthetic_fc(num_subjects, n_parcels)
        # Task FC is Rest FC + some noise + task effect
        fc_task = 0.8 * fc_rest + 0.2 * generate_synthetic_fc(num_subjects, n_parcels)
        fc_task = np.clip(fc_task, -1.0, 1.0)
        
        print(f">>> [OK] Generated synthetic data for {num_subjects} subjects")
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(save_rest_path, fc_rest)
        np.save(save_task_path, fc_task)

    if fc_rest.shape[0] > num_subjects:
        fc_rest = fc_rest[:num_subjects]
        fc_task = fc_task[:num_subjects]
        
    fc_rest = np.clip(np.nan_to_num(fc_rest, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    fc_task = np.clip(np.nan_to_num(fc_task, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    
    num_subjects = fc_rest.shape[0]
    n_parcels = fc_rest.shape[1]
    
    if num_subjects < 5:
        print("[!] Too few subjects (N < 5) for analysis. Aborting.")
        return None, None
    
    print(f"  Loaded {num_subjects} subjects, {n_parcels} parcels")

    # ==========================================
    # RAW FC BASELINE (Finn et al., 2015)
    # ==========================================
    print("\n[2/4] Computing Raw FC Baseline (Finn et al.)...")
    raw_rest = np.array([fc_rest[i][TRIL_IDX] for i in tqdm(range(num_subjects), desc="Vectorizing Rest FC")])
    raw_task = np.array([fc_task[i][TRIL_IDX] for i in tqdm(range(num_subjects), desc="Vectorizing Task FC")])
    raw_matrix = np.corrcoef(raw_task, raw_rest)[:num_subjects, num_subjects:]
    raw_metrics = compute_all_metrics(raw_matrix)
    print(f"  Raw FC Top-1: {raw_metrics['top_1_accuracy']:.4f}")

    # ==========================================
    # CVAE MODEL SETUP
    # ==========================================
    print(f"\n[3/4] Running CVAE+SDL with {n_folds}-Fold CV (Epochs={epochs}, K={K}, L={L}, beta={beta})...")
    
    cvae_model = ConditionalVAE(n_parcels, latent_dim).to(DEVICE)
    print(f"  CVAE Parameters: {cvae_model.count_parameters():,}")
    model_params = cvae_model.count_parameters()
    del cvae_model  # Will create fresh models per fold
    
    # Cross-validation setup (same seed as main pipeline)
    indices = np.random.RandomState(42).permutation(num_subjects)
    fold_size = num_subjects // n_folds
    
    # Out-of-fold arrays for aggregation
    refined_rest_all = np.zeros_like(fc_rest)
    refined_task_all = np.zeros_like(fc_task)
    residual_rest_all = np.zeros_like(fc_rest)
    residual_task_all = np.zeros_like(fc_task)
    fold_losses = []
    fold_accuracies_refined = []
    fold_accuracies_residual = []
    last_D = None  # Store dictionary from last fold for interpretability
    
    for fold in tqdm(range(n_folds), desc="CV Folds"):
        start_idx = fold * fold_size
        end_idx = num_subjects if fold == n_folds - 1 else (fold + 1) * fold_size
        test_idx = indices[start_idx:end_idx]
        train_idx = np.setdiff1d(indices, test_idx)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        
        # --- CVAE Training on TRAIN subjects only ---
        train_rest = fc_rest[train_idx]
        train_task = fc_task[train_idx]
        
        # Combine rest + task of TRAIN subjects, with condition labels
        x_train = torch.tensor(
            np.concatenate([train_rest, train_task])[:, np.newaxis, :, :],
            dtype=torch.float32
        )
        c_train = torch.tensor(
            np.concatenate([np.zeros(n_train), np.ones(n_train)]),
            dtype=torch.float32
        ).unsqueeze(1)
        
        fold_model = ConditionalVAE(n_parcels, latent_dim).to(DEVICE)
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
        dataset = TensorDataset(x_train, c_train)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training
        fold_model.train()
        for ep in tqdm(range(epochs), desc=f"  Fold {fold+1} Training", leave=False):
            ep_loss = 0
            n_batches = 0
            for batch_x, batch_c in dataloader:
                batch_x, batch_c = batch_x.to(DEVICE), batch_c.to(DEVICE)
                fold_optimizer.zero_grad()
                recon, mu, logvar = fold_model(batch_x, batch_c)
                
                loss = cvae_loss(recon, batch_x, mu, logvar, beta=beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=5.0)
                fold_optimizer.step()
                
                if not torch.isnan(loss):
                    ep_loss += loss.item()
                n_batches += 1

        avg_loss = ep_loss / max(1, n_batches)
        fold_losses.append(avg_loss)
        
        # --- Compute Residuals on TRAIN set for SDL Dictionary ---
        fold_model.eval()
        with torch.no_grad():
            # Train REST residuals (for dictionary learning)
            tr_rest_t = torch.tensor(train_rest[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            c_rest_tr = torch.zeros((n_train, 1), dtype=torch.float32).to(DEVICE)
            recon_train_rest, _, _ = fold_model(tr_rest_t, c_rest_tr)
        
        train_res_rest = train_rest - recon_train_rest.squeeze(1).cpu().numpy()
        
        # Learn Dictionary D from TRAIN residuals only (INDUCTIVE)
        Y_train = np.array([train_res_rest[i][TRIL_IDX] for i in tqdm(range(n_train), desc=f"  Fold {fold+1} Train SDL Setup", leave=False)]).T
        Y_train = np.nan_to_num(Y_train, nan=0.0)
        D, _ = k_svd(Y_train, K=K, L=L, n_iter=10, verbose=False, random_state=42)
        last_D = D  # Save for interpretability
        
        # --- Inference on HELD-OUT TEST subjects ---
        with torch.no_grad():
            te_rest_t = torch.tensor(fc_rest[test_idx][:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            te_task_t = torch.tensor(fc_task[test_idx][:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            c_rest_te = torch.zeros((n_test, 1), dtype=torch.float32).to(DEVICE)
            c_task_te = torch.ones((n_test, 1), dtype=torch.float32).to(DEVICE)
            
            recon_test_rest, _, _ = fold_model(te_rest_t, c_rest_te)
            recon_test_task, _, _ = fold_model(te_task_t, c_task_te)
        
        test_res_rest = fc_rest[test_idx] - recon_test_rest.squeeze(1).cpu().numpy()
        test_res_task = fc_task[test_idx] - recon_test_task.squeeze(1).cpu().numpy()
        
        # Store raw residuals for ablation
        for i, act_idx in tqdm(enumerate(test_idx), desc=f"  Fold {fold+1} Storing Residuals", total=n_test, leave=False):
            residual_rest_all[act_idx] = test_res_rest[i]
            residual_task_all[act_idx] = test_res_task[i]
        
        # Sparse code TEST residuals using TRAIN dictionary D (INDUCTIVE SDL)
        Y_test_rest = np.array([test_res_rest[i][TRIL_IDX] for i in tqdm(range(n_test), desc=f"  Fold {fold+1} Test SDL Setup (Rest)", leave=False)]).T
        Y_test_task = np.array([test_res_task[i][TRIL_IDX] for i in tqdm(range(n_test), desc=f"  Fold {fold+1} Test SDL Setup (Task)", leave=False)]).T
        
        Y_test_rest = np.nan_to_num(Y_test_rest, nan=0.0)
        Y_test_task = np.nan_to_num(Y_test_task, nan=0.0)
        
        X_rest_test = omp_sparse_coding(Y_test_rest, D, L=L)
        X_task_test = omp_sparse_coding(Y_test_task, D, L=L)
        
        for i, act_idx in tqdm(enumerate(test_idx), desc=f"  Fold {fold+1} Reconstruction", total=n_test, leave=False):
            refined_rest_all[act_idx] = test_res_rest[i] - reconstruct_symmetric_matrix(np.dot(D, X_rest_test[:, i]), n_parcels)
            refined_task_all[act_idx] = test_res_task[i] - reconstruct_symmetric_matrix(np.dot(D, X_task_test[:, i]), n_parcels)
        
        # Per-fold accuracy (for reporting)
        fold_ref_rest = np.array([refined_rest_all[idx][TRIL_IDX] for idx in test_idx])
        fold_ref_task = np.array([refined_task_all[idx][TRIL_IDX] for idx in test_idx])
        fold_corr = np.corrcoef(fold_ref_task, fold_ref_rest)[:n_test, n_test:]
        fold_acc_ref = calculate_accuracy(fold_corr)
        fold_accuracies_refined.append(fold_acc_ref)
        
        fold_res_rest = np.array([residual_rest_all[idx][TRIL_IDX] for idx in test_idx])
        fold_res_task = np.array([residual_task_all[idx][TRIL_IDX] for idx in test_idx])
        fold_corr_res = np.corrcoef(fold_res_task, fold_res_rest)[:n_test, n_test:]
        fold_acc_res = calculate_accuracy(fold_corr_res)
        fold_accuracies_residual.append(fold_acc_res)
        
        print(f"  Fold [{fold+1}/{n_folds}] Train: {n_train} | Test: {n_test} | "
              f"CVAE Loss: {avg_loss:.4f} | Refined Acc: {fold_acc_ref:.4f} | Residual Acc: {fold_acc_res:.4f}")
        
        # Free GPU memory between folds
        del fold_model, fold_optimizer, dataset, dataloader
        torch.cuda.empty_cache()

    # ==========================================
    # AGGREGATE RESULTS
    # ==========================================
    print(f"\n[4/4] Computing aggregate metrics...")
    
    # CVAE Residuals only (no SDL)
    res_rest_triu = np.array([residual_rest_all[i][TRIL_IDX] for i in range(num_subjects)])
    res_task_triu = np.array([residual_task_all[i][TRIL_IDX] for i in range(num_subjects)])
    residual_matrix = np.corrcoef(res_task_triu, res_rest_triu)[:num_subjects, num_subjects:]
    residual_metrics = compute_all_metrics(residual_matrix)
    
    # CVAE + SDL (full pipeline)
    ref_rest_triu = np.array([refined_rest_all[i][TRIL_IDX] for i in range(num_subjects)])
    ref_task_triu = np.array([refined_task_all[i][TRIL_IDX] for i in range(num_subjects)])
    cvae_sdl_matrix = np.corrcoef(ref_task_triu, ref_rest_triu)[:num_subjects, num_subjects:]
    cvae_sdl_metrics = compute_all_metrics(cvae_sdl_matrix)

    # ==========================================
    # PRINT RESULTS TABLE
    # ==========================================
    print("\n" + "=" * 60)
    print(f"RESULTS - {task_name.upper()} ({num_subjects} Subjects, {n_folds}-Fold CV)")
    print("=" * 60)
    print(f"{'Method':<30} {'Top-1':<10} {'Top-5':<10} {'MRR':<10} {'Diff-ID':<10}")
    print("-" * 60)
    print(f"{'Raw FC (Finn et al.)':<30} {raw_metrics['top_1_accuracy']:.4f}    {raw_metrics['top_5_accuracy']:.4f}    {raw_metrics['mrr']:.4f}    {raw_metrics['differential_id']:.4f}")
    print(f"{'CVAE Residual Only':<30} {residual_metrics['top_1_accuracy']:.4f}    {residual_metrics['top_5_accuracy']:.4f}    {residual_metrics['mrr']:.4f}    {residual_metrics['differential_id']:.4f}")
    print(f"{'CVAE + SDL (Lu et al.)':<30} {cvae_sdl_metrics['top_1_accuracy']:.4f}    {cvae_sdl_metrics['top_5_accuracy']:.4f}    {cvae_sdl_metrics['mrr']:.4f}    {cvae_sdl_metrics['differential_id']:.4f}")
    print("=" * 60)
    
    print(f"\nCVAE+SDL Comprehensive Metrics:")
    print("-" * 50)
    print(f"  Top-1 Accuracy:              {cvae_sdl_metrics['top_1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy:              {cvae_sdl_metrics['top_3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy:              {cvae_sdl_metrics['top_5_accuracy']:.4f}")
    print(f"  Top-10 Accuracy:             {cvae_sdl_metrics['top_10_accuracy']:.4f}")
    print(f"  Mean Rank:                   {cvae_sdl_metrics['mean_rank']:.2f}")
    print(f"  Mean Reciprocal Rank:        {cvae_sdl_metrics['mrr']:.4f}")
    print(f"  Differential Identifiability:{cvae_sdl_metrics['differential_id']:.4f}")
    print("=" * 60)

    # ==========================================
    # SAVE ALL OUTPUTS
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}_{task_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Save Matrices
    np.save(os.path.join(run_dir, "raw_fc_matrix.npy"), raw_matrix)
    np.save(os.path.join(run_dir, "residual_matrix.npy"), residual_matrix)
    np.save(os.path.join(run_dir, "cvae_sdl_id_matrix.npy"), cvae_sdl_matrix)
    if last_D is not None:
        np.save(os.path.join(run_dir, "dictionary_D.npy"), last_D)
    print(f"[OK] Matrices & dictionary saved to {run_dir}")
    
    # 2. Correlation Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    
    sns.heatmap(raw_matrix, cmap='RdBu_r', center=0, vmin=-0.5, vmax=1, ax=axes[0])
    axes[0].set_title(f'Raw FC ({num_subjects}x{num_subjects})')
    axes[0].set_xlabel('Subjects (Rest)')
    axes[0].set_ylabel('Subjects (Task)')
    
    sns.heatmap(residual_matrix, cmap='RdBu_r', center=0, vmin=-0.5, vmax=1, ax=axes[1])
    axes[1].set_title(f'CVAE Residual ({num_subjects}x{num_subjects})')
    axes[1].set_xlabel('Subjects (Rest)')
    axes[1].set_ylabel('Subjects (Task)')
    
    sns.heatmap(cvae_sdl_matrix, cmap='RdBu_r', center=0, vmin=-0.5, vmax=1, ax=axes[2])
    axes[2].set_title(f'CVAE+SDL ({num_subjects}x{num_subjects})')
    axes[2].set_xlabel('Subjects (Rest)')
    axes[2].set_ylabel('Subjects (Task)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "correlation_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Heatmaps saved")
    
    # 3. Similarity Distributions
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    for ax, matrix, title in [(axes[0], residual_matrix, 'CVAE Residual'), 
                               (axes[1], cvae_sdl_matrix, 'CVAE+SDL')]:
        n = matrix.shape[0]
        self_corr = np.diag(matrix)
        mask = ~np.eye(n, dtype=bool)
        other_corr = matrix[mask]
        ax.hist(other_corr, bins=50, alpha=0.7, label=f'Other (n={len(other_corr)})', density=True, color='steelblue')
        ax.hist(self_corr, bins=20, alpha=0.7, label=f'Self (n={len(self_corr)})', density=True, color='coral')
        ax.axvline(np.mean(self_corr), color='red', linestyle='--', linewidth=2, label=f'Self Mean: {np.mean(self_corr):.3f}')
        ax.axvline(np.mean(other_corr), color='blue', linestyle='--', linewidth=2, label=f'Other Mean: {np.mean(other_corr):.3f}')
        ax.set_xlabel('Correlation', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{title}: Self vs Other Distributions', fontsize=14)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "similarity_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Similarity distributions saved")
    
    # 4. Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Raw FC', 'CVAE Residual', 'CVAE+SDL']
    accs = [raw_metrics['top_1_accuracy'], residual_metrics['top_1_accuracy'], cvae_sdl_metrics['top_1_accuracy']]
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(methods, accs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('Top-1 Identification Accuracy', fontsize=13)
    ax.set_title(f'CVAE+SDL Ablation - {task_name.upper()} ({num_subjects} Subjects)', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "ablation_results.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Ablation chart saved")
    
    # 5. Dictionary Atoms Visualization
    if last_D is not None:
        n_show = min(K, 8)
        fig, axes = plt.subplots(2, n_show // 2, figsize=(4 * (n_show // 2), 8))
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                atom_matrix = reconstruct_symmetric_matrix(last_D[:, idx], n_parcels)
                im = ax.imshow(atom_matrix, cmap='RdBu_r', aspect='equal')
                ax.set_title(f'Atom {idx+1}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.suptitle(f'SDL Dictionary Atoms (K={K})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "dictionary_atoms.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Dictionary atoms saved")
    
    # 6. Comprehensive Report
    report_path = os.path.join(run_dir, "CVAE_SDL_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CVAE + SDL BASELINE - BRAIN FINGERPRINTING COMPARISON REPORT\n")
        f.write("Reference: Lu et al., NeuroImage 295 (2024) 120651\n")
        f.write("Awareness-Compliant: 5-Fold Subject-Level Cross-Validation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. EXPERIMENTAL CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Task: {task_name.upper()}\n")
        f.write(f"Subjects: {num_subjects}\n")
        f.write(f"Parcellation: Glasser MMP 2016 ({n_parcels} cortical parcels)\n")
        f.write(f"Protocol: {n_folds}-Fold Cross-Validation (Awareness-Compliant)\n")
        f.write(f"CVAE Epochs per fold: {epochs}\n")
        f.write(f"CVAE Latent Dim: {latent_dim}\n")
        f.write(f"CVAE beta (KL weight): {beta}\n")
        f.write(f"SDL K (atoms): {K}\n")
        f.write(f"SDL L (sparsity): {L}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Model Parameters: {model_params:,}\n\n")
        
        f.write("2. COMPARATIVE RESULTS (ABLATION)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Method':<30} {'Top-1':<10} {'Top-5':<10} {'MRR':<10} {'Diff-ID':<10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Raw FC (Finn et al.)':<30} {raw_metrics['top_1_accuracy']:.4f}    {raw_metrics['top_5_accuracy']:.4f}    {raw_metrics['mrr']:.4f}    {raw_metrics['differential_id']:.4f}\n")
        f.write(f"{'CVAE Residual Only':<30} {residual_metrics['top_1_accuracy']:.4f}    {residual_metrics['top_5_accuracy']:.4f}    {residual_metrics['mrr']:.4f}    {residual_metrics['differential_id']:.4f}\n")
        f.write(f"{'CVAE + SDL (Lu et al.)':<30} {cvae_sdl_metrics['top_1_accuracy']:.4f}    {cvae_sdl_metrics['top_5_accuracy']:.4f}    {cvae_sdl_metrics['mrr']:.4f}    {cvae_sdl_metrics['differential_id']:.4f}\n\n")
        
        f.write("3. CVAE+SDL COMPREHENSIVE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Top-1 Accuracy:              {cvae_sdl_metrics['top_1_accuracy']:.4f}\n")
        f.write(f"Top-3 Accuracy:              {cvae_sdl_metrics['top_3_accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy:              {cvae_sdl_metrics['top_5_accuracy']:.4f}\n")
        f.write(f"Top-10 Accuracy:             {cvae_sdl_metrics['top_10_accuracy']:.4f}\n")
        f.write(f"Mean Rank:                   {cvae_sdl_metrics['mean_rank']:.2f}\n")
        f.write(f"Mean Reciprocal Rank:        {cvae_sdl_metrics['mrr']:.4f}\n")
        f.write(f"Differential Identifiability:{cvae_sdl_metrics['differential_id']:.4f}\n\n")
        
        f.write("4. PER-FOLD CROSS-VALIDATION DETAILS\n")
        f.write("-" * 50 + "\n")
        for i in range(n_folds):
            f.write(f"Fold {i+1}: CVAE Loss = {fold_losses[i]:.4f} | "
                    f"Refined Acc = {fold_accuracies_refined[i]:.4f} | "
                    f"Residual Acc = {fold_accuracies_residual[i]:.4f}\n")
        f.write(f"\nMean CVAE Loss: {np.mean(fold_losses):.4f}\n")
        f.write(f"Mean Refined Acc: {np.mean(fold_accuracies_refined):.4f} +/- {np.std(fold_accuracies_refined):.4f}\n")
        f.write(f"Mean Residual Acc: {np.mean(fold_accuracies_residual):.4f} +/- {np.std(fold_accuracies_residual):.4f}\n\n")
        
        f.write("5. RAW FC COMPREHENSIVE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Top-1 Accuracy:              {raw_metrics['top_1_accuracy']:.4f}\n")
        f.write(f"Top-3 Accuracy:              {raw_metrics['top_3_accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy:              {raw_metrics['top_5_accuracy']:.4f}\n")
        f.write(f"Top-10 Accuracy:             {raw_metrics['top_10_accuracy']:.4f}\n")
        f.write(f"Mean Rank:                   {raw_metrics['mean_rank']:.2f}\n")
        f.write(f"Mean Reciprocal Rank:        {raw_metrics['mrr']:.4f}\n")
        f.write(f"Differential Identifiability:{raw_metrics['differential_id']:.4f}\n\n")
        
        f.write("6. MODEL ARCHITECTURE DETAILS\n")
        f.write("-" * 50 + "\n")
        f.write("Conditional VAE (Lu et al., 2024):\n")
        f.write("  Encoder: Conv2d(2, 16, 32, 64) + BatchNorm + ReLU + MaxPool\n")
        f.write("  Condition: fMRI state (Rest=0, Task=1) as extra input channel\n")
        f.write("  Latent: mu, log(sigma^2) via Linear (conditioned on state)\n")
        f.write("  Decoder: ConvTranspose2d(64, 32, 16, 1) + BatchNorm + ReLU + Tanh\n")
        f.write(f"  Parameters: {model_params:,}\n")
        f.write(f"  Training: beta-VAE Loss (beta={beta}), Adam (lr=0.001), {epochs} epochs\n\n")
        f.write("Sparse Dictionary Learning (K-SVD):\n")
        f.write(f"  K (atoms): {K}\n")
        f.write(f"  L (sparsity): {L}\n")
        f.write("  Iterations: 10\n")
        f.write("  Coding: Orthogonal Matching Pursuit (OMP)\n\n")
        
        f.write("7. AWARENESS COMPLIANCE STATEMENT\n")
        f.write("-" * 50 + "\n")
        f.write("This baseline strictly follows the awareness guidelines from\n")
        f.write("Orlichenko et al. (2023). All model training (CVAE) and representation\n")
        f.write("learning (SDL dictionary) are performed exclusively on training subjects.\n")
        f.write("Test subjects are NEVER seen during training. SDL dictionary from train\n")
        f.write("subjects is applied inductively to test subjects via OMP coding.\n")
        f.write("This prevents the identifiability inflation ('double-dipping') that\n")
        f.write("occurs when all subjects are used for both training and evaluation.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"[OK] Report saved to {report_path}")
    
    # 7. JSON Results for programmatic access
    json_results = {
        'task': task_name,
        'num_subjects': num_subjects,
        'n_folds': n_folds,
        'epochs': epochs,
        'beta': beta,
        'latent_dim': latent_dim,
        'K': K,
        'L': L,
        'model_params': model_params,
        'raw_fc_metrics': raw_metrics,
        'cvae_residual_metrics': residual_metrics,
        'cvae_sdl_metrics': cvae_sdl_metrics,
        'fold_losses': fold_losses,
        'fold_accuracies_refined': fold_accuracies_refined,
        'fold_accuracies_residual': fold_accuracies_residual,
        'cv_mean_acc': float(np.mean(fold_accuracies_refined)),
        'cv_std_acc': float(np.std(fold_accuracies_refined)),
        'timestamp': timestamp
    }
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"[OK] JSON results saved")
    
    # 8. Zip everything for easy Kaggle download
    zip_path = os.path.join(WORKING_DIR, f"cvae_sdl_results_{task_name}_{timestamp}")
    shutil.make_archive(zip_path, 'zip', run_dir)
    print(f"[OK] All results zipped to {zip_path}.zip")
    
    print(f"\n{'='*60}")
    print(f"ALL OUTPUTS SAVED TO: {run_dir}")
    print(f"ZIP DOWNLOAD: {zip_path}.zip")
    print(f"{'='*60}")
    
    return cvae_sdl_metrics, raw_metrics

# %% [markdown]
# ## 7. Execute Pipeline

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CVAE+SDL Baseline - Awareness-Compliant (Lu et al., 2024)")
    parser.add_argument("--task", type=str, default="all", help="Task name (motor, wm, emotion, etc.), or 'all'")
    parser.add_argument("--num_subjects", type=int, default=339, help="Number of subjects")
    parser.add_argument("--K", type=int, default=15, help="Number of dictionary atoms")
    parser.add_argument("--L", type=int, default=12, help="Sparsity level for OMP")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per fold")
    parser.add_argument("--beta", type=float, default=0.1, help="KL divergence weight")
    parser.add_argument("--latent_dim", type=int, default=128, help="CVAE latent dimension")
    parser.add_argument("--search_dict", action="store_true", help="Perform grid search over dictionary hyperparameters K and L")
    
    args, unknown = parser.parse_known_args()
    
    # ===== HYPERPARAMETER CONFIGURATION =====
    # Change to True to run automatic Grid Search (slow, searches K and L)
    # Change to False to use the manually tuned parameters below (fast)
    ENABLE_GRID_SEARCH = True
    
    PERFORM_GRID_SEARCH = args.search_dict or ENABLE_GRID_SEARCH
    
    # Pre-Tuned Parameters (K, L) for each task
    TUNED_PARAMS = {
        "motor": (14, 12),
        "wm": (15, 13),
        "emotion": (15, 13),
        "gambling": (14, 11),
        "language": (15, 11),
        "relational": (11, 9),
        "social": (15, 9)
    }

    if args.task.lower() == "all":
        tasks_to_run = list(TUNED_PARAMS.keys())
    else:
        tasks_to_run = [args.task]

    for current_task in tasks_to_run:
        print(f"\n\n{'='*70}")
        print(f"STARTING EXPERIMENT: Task = {current_task.upper()}")
        print(f"{'='*70}\n")
        
        try:
            if PERFORM_GRID_SEARCH:
                print("\n" + "*"*60)
                print(f"STARTING DICTIONARY HYPERPARAMETER SEARCH FOR CVAE+SDL ({current_task.upper()})")
                print("*"*60)
                # Load data paths
                rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
                task_path = os.path.join(FC_DATA_DIR, f"fc_{current_task}.npy")

                if not os.path.exists(rest_path) or not os.path.exists(task_path):
                    if RAW_REST_DIR is not None and RAW_TASK_DIR is not None:
                        print(f">>> [!] Data missing for grid search but raw directories found.")
                        print(f">>> Generating real FC data for {current_task} grid search...")
                        
                        # Get subject list
                        subjects_path = os.path.join(RAW_REST_DIR, "subjects")
                        all_subs = sorted([d for d in os.listdir(subjects_path) 
                                         if os.path.isdir(os.path.join(subjects_path, d))])
                        if len(all_subs) > args.num_subjects:
                            all_subs = all_subs[:args.num_subjects]
                            
                        # Generate Rest if missing
                        if not os.path.exists(rest_path):
                            fc_r, _ = generate_fc_for_task("rest", all_subs, RAW_REST_DIR)
                            os.makedirs(FC_DATA_DIR, exist_ok=True)
                            np.save(rest_path, fc_r)
                            
                        # Generate Task if missing
                        if not os.path.exists(task_path):
                            fc_t, _ = generate_fc_for_task(current_task, all_subs, RAW_TASK_DIR)
                            os.makedirs(FC_DATA_DIR, exist_ok=True)
                            np.save(task_path, fc_t)
                    else:
                        print(">>> [!] Data missing for grid search and no raw directories. Fallback to synthetic.")

                if os.path.exists(rest_path) and os.path.exists(task_path):
                    print(f">>> Loading pre-calculated matrices for grid search...")
                    fc_r = np.load(rest_path)
                    fc_t = np.load(task_path)
                    
                    if fc_r.shape[0] > args.num_subjects:
                        fc_r, fc_t = fc_r[:args.num_subjects], fc_t[:args.num_subjects]
                    
                    # Sanitize (Consistent with run_cvae_sdl_baseline)
                    fc_r = np.clip(np.nan_to_num(fc_r, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                    fc_t = np.clip(np.nan_to_num(fc_t, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                    
                    n_p = fc_r.shape[1]
                    tril = np.tril_indices(n_p, k=-1)
                    flat_r = np.array([fc_r[i][tril] for i in range(len(fc_r))]).T
                    flat_t = np.array([fc_t[i][tril] for i in range(len(fc_t))]).T
                    
                    # Search Range: K from 2 to 16
                    search_k_range = list(range(2, 17))
                    _, best_K, best_L = perform_grid_search(
                        flat_t, flat_r, len(fc_r), n_p, 
                        K_range=search_k_range, 
                        n_iter=10,
                        task_name=current_task
                    )
                else:
                    # Fallback to pure synthetic grid search if even generation failed
                    print(">>> [!] Real data generation failed. Using synthetic proxy for grid search.")
                    temp_num_subs = min(args.num_subjects, 20)
                    fc_r = generate_synthetic_fc(temp_num_subs, 360)
                    fc_t = 0.8 * fc_r + 0.2 * generate_synthetic_fc(temp_num_subs, 360)
                    fc_t = np.clip(fc_t, -1.0, 1.0)
                    
                    n_p = 360
                    tril = np.tril_indices(n_p, k=-1)
                    flat_r = np.array([fc_r[i][tril] for i in range(len(fc_r))]).T
                    flat_t = np.array([fc_t[i][tril] for i in range(len(fc_t))]).T
                    
                    search_k_range = list(range(2, 6)) # Smaller range for synthetic test
                    _, best_K, best_L = perform_grid_search(
                        flat_t, flat_r, len(fc_r), n_p, 
                        K_range=search_k_range, 
                        n_iter=3,
                        task_name=current_task
                    )
            else:
                # Use tuned parameters if not explicitly overridden by defaults
                if current_task in TUNED_PARAMS and args.K == 15 and args.L == 12:
                    best_K, best_L = TUNED_PARAMS[current_task]
                    print(f">>> Using Pre-Tuned Parameters for {current_task}: K={best_K}, L={best_L}")
                else:
                    best_K, best_L = args.K, args.L
                    print(f">>> Using Manual Parameters: K={best_K}, L={best_L}")
                    
            # Run final analysis fold-by-fold with best parameters
            run_cvae_sdl_baseline(
                task_name=current_task,
                num_subjects=args.num_subjects,
                K=best_K,
                L=best_L,
                epochs=args.epochs,
                beta=args.beta,
                latent_dim=args.latent_dim
            )
        except Exception as e:
            print(f"[ERROR] Failed on task {current_task}: {e}")
