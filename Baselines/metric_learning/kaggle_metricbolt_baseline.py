# %% [markdown]
# # MetricBolT Baseline - Brain Fingerprinting Comparison
# 
# This notebook implements the MetricBolT (Metric Learning + BolT Transformer) 
# baseline for direct comparison against our ConvAE+SDL Brain Fingerprinting pipeline.
#
# **Key Design Principles (Awareness-Compliant):**
# - 5-Fold Cross-Validation with strict subject-level splits
# - Model trained on TRAIN subjects only, evaluated on HELD-OUT subjects
# - Prevents identifiability leakage (Orlichenko et al., 2023)
#
# **Evaluation Protocol:**
# - Identical metrics to `kaggle_brain_fingerprinting.py`
# - Same data loading, same BOLD naming, same subject intersection
# - Results directly comparable
#
# ---

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "metricbolt_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# CLONE METRICBOLT & INSTALL DEPENDENCIES
# ==========================================
METRICBOLT_REPO_URL = "https://github.com/CAIMI-WEIGroup/MetricBolT.git"
REPO_DIR = os.path.join(WORKING_DIR, "MetricBolT")

if not os.path.exists(REPO_DIR):
    print(f"[*] Cloning MetricBolT from {METRICBOLT_REPO_URL}...")
    subprocess.run(["git", "clone", METRICBOLT_REPO_URL, REPO_DIR], check=True)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

try:
    import pytorch_metric_learning
except ImportError:
    print("[*] Installing pytorch-metric-learning...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-metric-learning", "-q"], check=True)

from utils import Option
from Models.BolT.hyperparams import getHyper_bolT
from Models.BolT.bolT import BolT
from pytorch_metric_learning import losses, distances

print("[OK] All MetricBolT dependencies loaded.")

# %% [markdown]
# ## 2. Data Discovery & Loading

# %%
# ==========================================
# PATH DISCOVERY (Matches main pipeline)
# ==========================================
RAW_REST_DIR = None
RAW_TASK_DIR = None

def setup_environment():
    """Discover HCP data paths on Kaggle or local filesystem."""
    global RAW_REST_DIR, RAW_TASK_DIR
    
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

    for d in [INPUT_DIR, WORKING_DIR, "..", "../DATA", "."]:
        if not os.path.exists(d):
            continue
        for root, dirs, _ in os.walk(d):
            if "subjects" in dirs:
                if "rest" in root.lower() and RAW_REST_DIR is None:
                    RAW_REST_DIR = root
                elif ("task" in root.lower() or "motor" in root.lower()) and RAW_TASK_DIR is None:
                    RAW_TASK_DIR = root

setup_environment()

def generate_synthetic_timeseries(num_subjects, n_parcels=360, length=284):
    """Generate synthetic BOLD timeseries data."""
    print(f">>> [SYNTHETIC] Generating {num_subjects} synthetic subjects (length={length})...")
    ts_data = []
    for i in range(num_subjects):
        # Base signal components
        t = np.linspace(0, 10, length)
        # Shared low-frequency components
        shared = np.sin(0.5 * t) + 0.5 * np.cos(0.2 * t)
        
        # Subject-specific weights and unique signal
        weights = np.random.randn(n_parcels, 1)
        unique_sig = np.sin((1.0 + 0.1 * i) * t)
        
        # Combine: shared + unique + noise
        ts = weights @ shared.reshape(1, -1) + 0.5 * unique_sig.reshape(1, -1) + np.random.normal(0, 0.5, (n_parcels, length))
        
        # Z-score
        ts = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
        ts_data.append(ts)
    return np.array(ts_data)

print(f"Raw Rest Dir: {RAW_REST_DIR or 'Not Found'}")
print(f"Raw Task Dir: {RAW_TASK_DIR or 'Not Found'}")

# ==========================================
# BOLD SCAN DEFINITIONS
# ==========================================
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]

def get_image_ids(name):
    """Get BOLD run IDs for a given task name."""
    run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
    if not run_ids: 
        raise ValueError(f"Found no data for '{name}'")
    return run_ids

def load_concat_timeseries(subject_id, run_idx_list, base_dir, dynamic_length=None):
    """Load and concatenate timeseries for a subject, with z-scoring and optional cropping."""
    ts_list = []
    for run_id in run_idx_list:
        bold_path = os.path.join(base_dir, "subjects", str(subject_id), "timeseries")
        bold_file = f"bold{run_id}_Atlas_MSMAll_Glasser360Cortical.npy"
        full_path = os.path.join(bold_path, bold_file)
        if os.path.exists(full_path):
            ts = np.load(full_path)
            ts = (ts - np.mean(ts, axis=1, keepdims=True)) / (np.std(ts, axis=1, keepdims=True) + 1e-8)
            ts_list.append(ts)
            
    if not ts_list:
        return None
        
    ts_concat = np.concatenate(ts_list, axis=1)
    
    if dynamic_length and ts_concat.shape[1] > dynamic_length:
        ts_concat = ts_concat[:, :dynamic_length]
        
    return ts_concat

# %% [markdown]
# ## 3. Evaluation Metrics (Identical to Main Pipeline)

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
# ## 4. MetricBolT Training & Evaluation (5-Fold CV)

# %%
def run_comparison(task_name="motor", num_subjects=100, dynamic_length=284, epochs=20, device=DEVICE):
    """
    Run MetricBolT baseline comparison with 5-Fold Cross-Validation.
    
    Awareness Fix: Subjects are strictly split between train and test to prevent
    identifiability leakage (Orlichenko et al., 2023). The model is trained per 
    fold on TRAIN subjects only and evaluated on HELD-OUT subjects.
    """
    print("=" * 60)
    print("METRICBOLT BASELINE - 5-FOLD CV (AWARENESS COMPLIANT)")
    print("=" * 60)

    if RAW_REST_DIR is None or RAW_TASK_DIR is None:
        print(">>> [!] Raw REST or TASK directories not found.")
        print(">>> [!] LOCAL RUN DETECTED: Generating synthetic data for seamless execution...")
        
        if num_subjects > 20: 
            print(f">>> [!] Reducing num_subjects from {num_subjects} to 20 for local synthetic run.")
            num_subjects = 20
            
        rest_tensor = torch.tensor(generate_synthetic_timeseries(num_subjects, 360, dynamic_length), dtype=torch.float32)
        # Rest2 is similar to Rest1 but with different noise
        rest_tensor2 = rest_tensor + torch.randn_like(rest_tensor) * 0.1
        # Task is related but different
        task_tensor = 0.7 * rest_tensor + 0.3 * torch.tensor(generate_synthetic_timeseries(num_subjects, 360, dynamic_length), dtype=torch.float32)
        
        # Normalization
        rest_tensor2 = (rest_tensor2 - rest_tensor2.mean(dim=2, keepdim=True)) / (rest_tensor2.std(dim=2, keepdim=True) + 1e-8)
        task_tensor = (task_tensor - task_tensor.mean(dim=2, keepdim=True)) / (task_tensor.std(dim=2, keepdim=True) + 1e-8)
        
        valid_subjects = [f"sub-{i}" for i in range(num_subjects)]
        print(f">>> [OK] Generated synthetic data for {num_subjects} subjects")
    else:
        rest_run_ids = get_image_ids("rest")
        task_run_ids = get_image_ids(task_name)

        subs_rest = set([d for d in os.listdir(os.path.join(RAW_REST_DIR, "subjects")) if d.isdigit()])
        subs_task = set([d for d in os.listdir(os.path.join(RAW_TASK_DIR, "subjects")) if d.isdigit()])
        intersect_subs = sorted(list(subs_rest.intersection(subs_task)))

        if len(intersect_subs) < num_subjects:
            num_subjects = len(intersect_subs)
        subjects = intersect_subs[:num_subjects]

        print(f"[*] Found {len(intersect_subs)} common subjects, using {num_subjects}")

        # ==========================================
        # DATA LOADING
        # ==========================================
        rest_data_all = []
        rest2_data_all = []
        task_data_all = []
        valid_subjects = []

        # Select LR runs for strict phase-encoding matching
        rest_lr_1 = rest_run_ids[0]                                            # REST1_LR
        rest_lr_2 = rest_run_ids[2] if len(rest_run_ids) > 2 else rest_run_ids[0]  # REST2_LR
        task_lr = task_run_ids[1] if len(task_run_ids) > 1 else task_run_ids[0]     # Task_LR

        print(f"[*] Loading timeseries (REST1_LR=bold{rest_lr_1}, REST2_LR=bold{rest_lr_2}, TASK_LR=bold{task_lr})...")

        for sub_id in tqdm(subjects, desc="Loading data"):
            ts_rest = load_concat_timeseries(sub_id, [rest_lr_1], RAW_REST_DIR, dynamic_length)
            ts_rest2 = load_concat_timeseries(sub_id, [rest_lr_2], RAW_REST_DIR, dynamic_length)
            ts_task = load_concat_timeseries(sub_id, [task_lr], RAW_TASK_DIR, dynamic_length)
            
            if ts_rest is not None and ts_rest2 is not None and ts_task is not None:
                if ts_rest.shape[1] == dynamic_length and ts_rest2.shape[1] == dynamic_length and ts_task.shape[1] == dynamic_length:
                    rest_data_all.append(ts_rest)
                    rest2_data_all.append(ts_rest2)
                    task_data_all.append(ts_task)
                    valid_subjects.append(sub_id)

        num_subjects = len(valid_subjects)
        print(f"[*] Valid subjects with complete data: {num_subjects}")
        
        if num_subjects < 5:
            print("[!] Too few subjects (N < 5) for analysis. Aborting.")
            return
        
        # Convert to PyTorch Tensors
        rest_tensor = torch.tensor(np.array(rest_data_all), dtype=torch.float32)
        rest_tensor2 = torch.tensor(np.array(rest2_data_all), dtype=torch.float32)
        task_tensor = torch.tensor(np.array(task_data_all), dtype=torch.float32)
    
    print(f"[*] Tensor shapes - Rest: {rest_tensor.shape}, Task: {task_tensor.shape}")

    # ==========================================
    # 5-FOLD CROSS-VALIDATION (AWARENESS FIX)
    # ==========================================
    n_folds = 5
    indices = np.random.RandomState(42).permutation(num_subjects)
    fold_size = num_subjects // n_folds
    
    final_rest_embs = [None] * num_subjects
    final_task_embs = [None] * num_subjects
    fold_losses = []
    
    print(f"\n[*] Starting {n_folds}-Fold CV Training ({epochs} epochs/fold)...")
    print(f"    Protocol: Train on REST1+REST2 of TRAIN subjects -> Evaluate on HELD-OUT subjects")
    
    for fold in tqdm(range(n_folds), desc="CV Folds"):
        start_idx = fold * fold_size
        end_idx = num_subjects if fold == n_folds - 1 else (fold + 1) * fold_size
        test_idx = indices[start_idx:end_idx]
        train_idx = np.setdiff1d(indices, test_idx)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        
        train_rest1 = rest_tensor[train_idx]
        train_rest2 = rest_tensor2[train_idx]
        test_rest = rest_tensor[test_idx]
        test_task = task_tensor[test_idx]

        # --- Fresh Model Per Fold ---
        hyper_dict = getHyper_bolT()
        hyper_dict.dim = 360  # Glasser360 parcellation
        hyper_dict.loss = "TripletMarginLoss"
        hyper_dict.method = "Metric"
        hyper_dict.pooling = "cls"
        
        batch_size = min(16, n_train * 2)
        details = Option({
            "device": device,
            "nOfTrains": n_train,
            "batchSize": batch_size,
            "nOfEpochs": epochs,
            "nOfClasses": 0
        })

        bolt_model = BolT(hyper_dict, details).to(device)
        optimizer = torch.optim.AdamW(bolt_model.parameters(), lr=1e-4, weight_decay=1e-3)
        criterion = losses.NTXentLoss(temperature=0.1)
        
        # Combine REST1 + REST2 of TRAIN subjects as positive pairs
        train_tensor_combined = torch.cat([train_rest1, train_rest2], dim=0)
        train_labels = torch.cat([torch.arange(n_train), torch.arange(n_train)], dim=0)
        dataset = torch.utils.data.TensorDataset(train_tensor_combined, train_labels)
        
        try:
            from pytorch_metric_learning.samplers import MPerClassSampler
            sampler = MPerClassSampler(train_labels, m=2, batch_size=details.batchSize, length_before_new_iter=len(dataset))
            dataloader = DataLoader(dataset, batch_size=details.batchSize, sampler=sampler)
        except ImportError:
            dataloader = DataLoader(dataset, batch_size=details.batchSize, shuffle=True)
        
        # --- Training ---
        bolt_model.train()
        for ep in tqdm(range(epochs), desc=f"  Fold {fold+1} Training", leave=False):
            ep_loss = 0
            batches = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                embeddings = bolt_model(batch_x)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                try:
                    loss = criterion(embeddings, batch_y)
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()
                    batches += 1
                except BaseException:
                    pass
                    
            if ep == epochs - 1:
                avg_loss = ep_loss / max(1, batches)
                fold_losses.append(avg_loss)
                print(f"  Fold [{fold+1}/{n_folds}] Train: {n_train} | Test: {n_test} | Final Loss: {avg_loss:.4f}")

        # --- Inference on HELD-OUT test subjects ---
        bolt_model.eval()
        with torch.no_grad():
            for i, actual_idx in tqdm(enumerate(test_idx), desc=f"  Fold {fold+1} Inference", total=n_test, leave=False):
                rx = test_rest[i:i+1].to(device)
                tx = test_task[i:i+1].to(device)
                
                re_cls = bolt_model(rx)
                te_cls = bolt_model(tx)
                
                final_rest_embs[actual_idx] = re_cls.cpu().numpy().squeeze()
                final_task_embs[actual_idx] = te_cls.cpu().numpy().squeeze()
        
        # Free GPU memory between folds
        del bolt_model, optimizer, criterion, dataloader, dataset
        torch.cuda.empty_cache()
                
    rest_embs = np.array(final_rest_embs)
    task_embs = np.array(final_task_embs)

    print(f"\n[*] Embedding shape: {rest_embs.shape}")
    print(f"[*] Embedding stats - mean: {rest_embs.mean():.4f}, std: {rest_embs.std():.4f}")
    
    # ==========================================
    # RAW FC BASELINE (Finn et al.)
    # ==========================================
    print("\n[*] Computing Raw FC Baseline (Finn et al.)...")
    raw_rest = []
    raw_task = []
    triu_idx = np.triu_indices(360, k=1)
    
    for i in tqdm(range(num_subjects), desc="Vectorizing Raw FC"):
        raw_rest.append(np.corrcoef(rest_tensor[i].numpy())[triu_idx])
        raw_task.append(np.corrcoef(task_tensor[i].numpy())[triu_idx])
    
    raw_rest = np.array(raw_rest)
    raw_task = np.array(raw_task)
    
    raw_matrix = np.corrcoef(raw_task, raw_rest)[:num_subjects, num_subjects:]
    raw_metrics = compute_all_metrics(raw_matrix)

    # ==========================================
    # METRICBOLT RESULTS
    # ==========================================
    print("[*] Computing MetricBolT Identification Metrics...")
    id_matrix = np.corrcoef(task_embs, rest_embs)[:num_subjects, num_subjects:]
    metrics = compute_all_metrics(id_matrix)
    
    # ==========================================
    # RESULTS OUTPUT
    # ==========================================
    print("\n" + "=" * 60)
    print(f"RESULTS - {task_name.upper()} ({num_subjects} Subjects, {n_folds}-Fold CV)")
    print("=" * 60)
    print(f"{'Method':<30} {'Top-1':<10} {'Top-5':<10} {'MRR':<10} {'Diff-ID':<10}")
    print("-" * 60)
    print(f"{'Raw FC (Finn et al.)':<30} {raw_metrics['top_1_accuracy']:.4f}    {raw_metrics['top_5_accuracy']:.4f}    {raw_metrics['mrr']:.4f}    {raw_metrics['differential_id']:.4f}")
    print(f"{'MetricBolT (NT-Xent, 5-CV)':<30} {metrics['top_1_accuracy']:.4f}    {metrics['top_5_accuracy']:.4f}    {metrics['mrr']:.4f}    {metrics['differential_id']:.4f}")
    
    print(f"\nMetricBolT Comprehensive Metrics:")
    print("-" * 50)
    print(f"  Top-1 Accuracy:              {metrics['top_1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy:              {metrics['top_3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy:              {metrics['top_5_accuracy']:.4f}")
    print(f"  Top-10 Accuracy:             {metrics['top_10_accuracy']:.4f}")
    print(f"  Mean Rank:                   {metrics['mean_rank']:.2f}")
    print(f"  Mean Reciprocal Rank:        {metrics['mrr']:.4f}")
    print(f"  Differential Identifiability:{metrics['differential_id']:.4f}")
    print("=" * 60)
    
    # ==========================================
    # SAVE ALL OUTPUTS
    # ==========================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import datetime
    import shutil
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}_{task_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Save Embeddings
    np.save(os.path.join(run_dir, "rest_embeddings.npy"), rest_embs)
    np.save(os.path.join(run_dir, "task_embeddings.npy"), task_embs)
    np.save(os.path.join(run_dir, "id_matrix.npy"), id_matrix)
    np.save(os.path.join(run_dir, "raw_fc_matrix.npy"), raw_matrix)
    print(f"[OK] Embeddings & matrices saved to {run_dir}")
    
    # 2. Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(raw_matrix, cmap='RdBu_r', center=0, vmin=-0.5, vmax=1, ax=axes[0])
    axes[0].set_title(f'Raw FC Correlation ({num_subjects}x{num_subjects})')
    axes[0].set_xlabel('Subjects (Rest)')
    axes[0].set_ylabel('Subjects (Task)')
    
    sns.heatmap(id_matrix, cmap='RdBu_r', center=0, vmin=-0.5, vmax=1, ax=axes[1])
    axes[1].set_title(f'MetricBolT Correlation ({num_subjects}x{num_subjects})')
    axes[1].set_xlabel('Subjects (Rest)')
    axes[1].set_ylabel('Subjects (Task)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "correlation_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Heatmaps saved")
    
    # 3. Similarity Distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    n = id_matrix.shape[0]
    self_corr = np.diag(id_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = id_matrix[mask]
    ax.hist(other_corr, bins=50, alpha=0.7, label=f'Other (n={len(other_corr)})', density=True, color='steelblue')
    ax.hist(self_corr, bins=20, alpha=0.7, label=f'Self (n={len(self_corr)})', density=True, color='coral')
    ax.axvline(np.mean(self_corr), color='red', linestyle='--', linewidth=2, label=f'Self Mean: {np.mean(self_corr):.3f}')
    ax.axvline(np.mean(other_corr), color='blue', linestyle='--', linewidth=2, label=f'Other Mean: {np.mean(other_corr):.3f}')
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('MetricBolT: Self vs Other Correlation Distributions', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "similarity_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Similarity distributions saved")
    
    # 4. Comprehensive Results Report
    report_path = os.path.join(run_dir, "METRICBOLT_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("METRICBOLT BASELINE - BRAIN FINGERPRINTING COMPARISON REPORT\n")
        f.write("Awareness-Compliant: 5-Fold Subject-Level Cross-Validation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. EXPERIMENTAL CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Task: {task_name.upper()}\n")
        f.write(f"Subjects: {num_subjects}\n")
        f.write(f"Protocol: {n_folds}-Fold Cross-Validation (Awareness-Compliant)\n")
        f.write(f"Epochs per fold: {epochs}\n")
        f.write(f"Dynamic length: {dynamic_length}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: BolT Transformer + NT-Xent Loss\n")
        f.write(f"Optimizer: AdamW (lr=1e-4, wd=1e-3)\n\n")
        
        f.write("2. COMPARATIVE RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Method':<30} {'Top-1':<10} {'Top-5':<10} {'MRR':<10} {'Diff-ID':<10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Raw FC (Finn et al.)':<30} {raw_metrics['top_1_accuracy']:.4f}    {raw_metrics['top_5_accuracy']:.4f}    {raw_metrics['mrr']:.4f}    {raw_metrics['differential_id']:.4f}\n")
        f.write(f"{'MetricBolT (NT-Xent, 5-CV)':<30} {metrics['top_1_accuracy']:.4f}    {metrics['top_5_accuracy']:.4f}    {metrics['mrr']:.4f}    {metrics['differential_id']:.4f}\n\n")
        
        f.write("3. METRICBOLT COMPREHENSIVE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Top-1 Accuracy:              {metrics['top_1_accuracy']:.4f}\n")
        f.write(f"Top-3 Accuracy:              {metrics['top_3_accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy:              {metrics['top_5_accuracy']:.4f}\n")
        f.write(f"Top-10 Accuracy:             {metrics['top_10_accuracy']:.4f}\n")
        f.write(f"Mean Rank:                   {metrics['mean_rank']:.2f}\n")
        f.write(f"Mean Reciprocal Rank:        {metrics['mrr']:.4f}\n")
        f.write(f"Differential Identifiability:{metrics['differential_id']:.4f}\n\n")
        
        f.write("4. PER-FOLD TRAINING DETAILS\n")
        f.write("-" * 50 + "\n")
        for i, loss_val in enumerate(fold_losses):
            f.write(f"Fold {i+1}: Final NT-Xent Loss = {loss_val:.4f}\n")
        f.write(f"Mean Loss: {np.mean(fold_losses):.4f}\n\n")
        
        f.write("5. RAW FC COMPREHENSIVE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Top-1 Accuracy:              {raw_metrics['top_1_accuracy']:.4f}\n")
        f.write(f"Top-3 Accuracy:              {raw_metrics['top_3_accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy:              {raw_metrics['top_5_accuracy']:.4f}\n")
        f.write(f"Top-10 Accuracy:             {raw_metrics['top_10_accuracy']:.4f}\n")
        f.write(f"Mean Rank:                   {raw_metrics['mean_rank']:.2f}\n")
        f.write(f"Mean Reciprocal Rank:        {raw_metrics['mrr']:.4f}\n")
        f.write(f"Differential Identifiability:{raw_metrics['differential_id']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"[OK] Report saved to {report_path}")
    
    # 5. JSON Results for programmatic access
    json_results = {
        'task': task_name,
        'num_subjects': num_subjects,
        'n_folds': n_folds,
        'epochs': epochs,
        'dynamic_length': dynamic_length,
        'metricbolt_metrics': metrics,
        'raw_fc_metrics': raw_metrics,
        'fold_losses': fold_losses,
        'timestamp': timestamp
    }
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"[OK] JSON results saved")
    
    # 6. Zip everything for easy Kaggle download
    zip_path = os.path.join(WORKING_DIR, f"metricbolt_results_{task_name}_{timestamp}")
    shutil.make_archive(zip_path, 'zip', run_dir)
    print(f"[OK] All results zipped to {zip_path}.zip")
    
    print(f"\n{'='*60}")
    print(f"ALL OUTPUTS SAVED TO: {run_dir}")
    print(f"ZIP DOWNLOAD: {zip_path}.zip")
    print(f"{'='*60}")
    return metrics, raw_metrics

# %% [markdown]
# ## 5. Execute Pipeline

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MetricBolT Baseline - Awareness-Compliant")
    parser.add_argument("--task", type=str, default="all", help="Task name (motor, wm, emotion, etc., or 'all')")
    parser.add_argument("--num_subjects", type=int, default=339, help="Number of subjects")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per fold")
    parser.add_argument("--dynamic_length", type=int, default=284, help="Timeseries length (fallback)")
    
    args, unknown = parser.parse_known_args()
    
    task_lengths = {
        "motor": 284,
        "wm": 405, 
        "emotion": 176,
        "gambling": 253,
        "language": 316,
        "relational": 232,
        "social": 274
    }

    if args.task.lower() == "all":
        tasks_to_run = list(task_lengths.keys())
    else:
        tasks_to_run = [args.task]

    for task in tasks_to_run:
        dl = task_lengths.get(task.lower(), args.dynamic_length)
        print(f"\n{'*' * 60}")
        print(f"STARTING EXPERIMENT: Task = {task.upper()}, Dynamic Length = {dl}")
        print(f"{'*' * 60}\n")
        
        try:
            run_comparison(
                task_name=task, 
                num_subjects=args.num_subjects, 
                dynamic_length=dl, 
                epochs=args.epochs
            )
        except Exception as e:
            print(f"[ERROR] Failed on task {task}: {e}")
            import traceback
            traceback.print_exc()
