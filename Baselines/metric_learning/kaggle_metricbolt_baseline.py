import os
import sys
import glob
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ==========================================
# 1. SETUP & CLONE METRICBOLT PIPELINE
# ==========================================
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
METRICBOLT_REPO_URL = "https://github.com/CAIMI-WEIGroup/MetricBolT.git"
REPO_DIR = os.path.join(WORKING_DIR, "MetricBolT")

if not os.path.exists(REPO_DIR):
    print(f"[*] Cloning MetricBolT from {METRICBOLT_REPO_URL} into {REPO_DIR}...")
    subprocess.run(["git", "clone", METRICBOLT_REPO_URL, REPO_DIR], check=True)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

try:
    import pytorch_metric_learning
except ImportError:
    print("[*] Installing pytorch-metric-learning...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-metric-learning"], check=True)

try:
    from utils import Option
    from Models.BolT.hyperparams import getHyper_bolT
    from Models.BolT.bolT import BolT
    from pytorch_metric_learning import losses, distances
except ImportError as e:
    print(f"[!] Import Error from MetricBolT repo: {e}")
    sys.exit(1)

# ==========================================
# 2. PATH DISCOVERY LOGIC (Matches main code)
# ==========================================
RAW_REST_DIR = None
RAW_TASK_DIR = None

def setup_environment():
    global RAW_REST_DIR, RAW_TASK_DIR
    for d in [INPUT_DIR, WORKING_DIR, "..", "../DATA", "."]:
        for root, dirs, _ in os.walk(d):
            if "subjects" in dirs:
                if "rest" in root.lower() and RAW_REST_DIR is None:
                    RAW_REST_DIR = root
                elif ("task" in root.lower() or "motor" in root.lower()) and RAW_TASK_DIR is None:
                    RAW_TASK_DIR = root

setup_environment()

BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]

def get_image_ids(name):
    run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
    if not run_ids: 
        raise ValueError(f"Found no data for '{name}'")
    return run_ids

def load_concat_timeseries(subject_id, run_idx_list, base_dir, dynamic_length=None):
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
        
    # Concatenate all available runs to form a stable long sequence (e.g. LR + RL)
    ts_concat = np.concatenate(ts_list, axis=1)
    
    # Crop to requested length instead of padding, ensuring no artificial connectivity jumps
    if dynamic_length and ts_concat.shape[1] > dynamic_length:
        ts_concat = ts_concat[:, :dynamic_length]
        
    return ts_concat

# ==========================================
# 3. EVALUATION METRICS (Identical to True Benchmark)
# ==========================================
def calculate_accuracy(corr_matrix):
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if np.argmax(corr_matrix[i, :]) == i)
    return correct / n

def calculate_top_k_accuracy(corr_matrix, k=5):
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if i in np.argsort(corr_matrix[i, :])[-k:])
    return correct / n

def calculate_mean_rank(corr_matrix):
    n = corr_matrix.shape[0]
    ranks = []
    for i in range(n):
        sorted_idx = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)
    return np.mean(ranks)

def calculate_mrr(corr_matrix):
    n = corr_matrix.shape[0]
    mrr = 0
    for i in range(n):
        sorted_idx = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        mrr += 1.0 / rank
    return mrr / n

def differential_identifiability(corr_matrix):
    n = corr_matrix.shape[0]
    self_corr = np.diag(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[mask]
    return np.mean(self_corr) - np.mean(other_corr)

def compute_all_metrics(corr_matrix):
    return {
        'top_1_accuracy': calculate_accuracy(corr_matrix),
        'top_3_accuracy': calculate_top_k_accuracy(corr_matrix, k=3),
        'top_5_accuracy': calculate_top_k_accuracy(corr_matrix, k=5),
        'top_10_accuracy': calculate_top_k_accuracy(corr_matrix, k=10),
        'mean_rank': calculate_mean_rank(corr_matrix),
        'mrr': calculate_mrr(corr_matrix),
        'differential_id': differential_identifiability(corr_matrix)
    }

# ==========================================
# 4. TRAINING & EVALUATION SCRIPT
# ==========================================
def run_comparison(task_name="motor", num_subjects=100, dynamic_length=284, epochs=20, device="cuda" if torch.cuda.is_available() else "cpu"):
    print("-" * 60)
    print("METRICBOLT BASELINE COMPARISON EVALUATION")
    print("-" * 60)

    if RAW_REST_DIR is None or RAW_TASK_DIR is None:
        print("[!] Raw REST or TASK directories not found. Please ensure 'subjects' directories are present.")
        sys.exit(1)

    rest_run_ids = get_image_ids("rest")
    task_run_ids = get_image_ids(task_name)

    subs_rest = set([d for d in os.listdir(os.path.join(RAW_REST_DIR, "subjects")) if d.isdigit()])
    subs_task = set([d for d in os.listdir(os.path.join(RAW_TASK_DIR, "subjects")) if d.isdigit()])
    intersect_subs = sorted(list(subs_rest.intersection(subs_task)))

    if len(intersect_subs) < num_subjects:
        num_subjects = len(intersect_subs)
    subjects = intersect_subs[:num_subjects]

    print(f"[*] Extracting Timeseries for {num_subjects} Subjects (Rest & {task_name.capitalize()})...")
    
    rest_data_all = []
    rest2_data_all = []
    task_data_all = []
    valid_subjects = []

    print(f"[*] Extracting Timeseries for {num_subjects} Subjects (Rest & {task_name.capitalize()})...")
    
    rest_data_all = []
    rest2_data_all = []
    task_data_all = []
    valid_subjects = []

    # Find the LR run explicitly for strict phase-encoding matching
    # REST1_LR is index 0 in rest_run_ids. REST2_LR is index 2.
    # MOTOR_LR is index 1 in task_run_ids. MOTOR_RL is index 0.
    
    rest_lr_1 = rest_run_ids[0]
    rest_lr_2 = rest_run_ids[2] if len(rest_run_ids) > 2 else rest_run_ids[0]
    
    # MOTOR run ids: [5 (RL), 6 (LR)] -> We want 6 (index 1) if available, otherwise 0
    task_lr = task_run_ids[1] if len(task_run_ids) > 1 else task_run_ids[0]

    for sub_id in tqdm(subjects, desc="Loading data"):
        # We form exactly a 284-length sequence purely from LR scans to avoid LR/RL phase shift artifacts
        ts_rest = load_concat_timeseries(sub_id, [rest_lr_1], RAW_REST_DIR, dynamic_length)
        ts_rest2 = load_concat_timeseries(sub_id, [rest_lr_2], RAW_REST_DIR, dynamic_length)
        ts_task = load_concat_timeseries(sub_id, [task_lr], RAW_TASK_DIR, dynamic_length)
        
        if ts_rest is not None and ts_rest2 is not None and ts_task is not None:
            # Enforce strict length matching dropping subjects missing complete sets
            if ts_rest.shape[1] == dynamic_length and ts_rest2.shape[1] == dynamic_length and ts_task.shape[1] == dynamic_length:
                rest_data_all.append(ts_rest)
                rest2_data_all.append(ts_rest2)
                task_data_all.append(ts_task)
                valid_subjects.append(sub_id)

    num_subjects = len(valid_subjects)
    
    # To PyTorch Tensors
    rest_tensor = torch.tensor(np.array(rest_data_all), dtype=torch.float32)
    rest_tensor2 = torch.tensor(np.array(rest2_data_all), dtype=torch.float32)
    task_tensor = torch.tensor(np.array(task_data_all), dtype=torch.float32)
    
    # --- MODEL SETUP ---
    hyper_dict = getHyper_bolT()
    hyper_dict.dim = 360 # Glasser
    hyper_dict.loss = "TripletMarginLoss"
    hyper_dict.method = "Metric"
    hyper_dict.pooling = "cls" # output cls token embedding
    
    details = Option({
        "device": device,
        "nOfTrains": num_subjects,
        "batchSize": 16,  # Increased batch size for stable contrastive learning
        "nOfEpochs": epochs,
        "nOfClasses": 0
    })

    print(f"[*] Instantiating MetricBolT (Transformer) Model...")
    bolt_model = BolT(hyper_dict, details).to(device)
    
    # Lower, robust learning rate for Transformer to avoid collapse
    optimizer = torch.optim.AdamW(bolt_model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    # Use NTXentLoss (SimCLR loss) instead of TripletMarginLoss which is highly prone to mode collapse
    criterion = losses.NTXentLoss(temperature=0.1)
    
    # Combine the two rest runs for training so Contrastive Loss can find positive pairs
    train_tensor = torch.cat([rest_tensor, rest_tensor2], dim=0)
    train_labels = torch.cat([torch.arange(num_subjects), torch.arange(num_subjects)], dim=0)
    dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)
    
    try:
        from pytorch_metric_learning.samplers import MPerClassSampler
        sampler = MPerClassSampler(train_labels, m=2, batch_size=details.batchSize, length_before_new_iter=len(dataset))
        dataloader = DataLoader(dataset, batch_size=details.batchSize, sampler=sampler)
    except ImportError:
        dataloader = DataLoader(dataset, batch_size=details.batchSize, shuffle=True)
    
    # --- TRAINING STAGE ---
    print(f"[*] Commencing NT-Xent Contrastive Training ({epochs} Epochs)...")
    import warnings
    warnings.filterwarnings("ignore")

    bolt_model.train()
    for ep in range(epochs):
        ep_loss = 0
        batches = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            embeddings = bolt_model(batch_x)
            
            # L2 Normalize embeddings to prevent magnitude explosion / collapse
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            try:
                loss = criterion(embeddings, batch_y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                batches += 1
            except BaseException as e:
                pass 
        
        avg_loss = ep_loss / max(1, batches)
        print(f"Epoch [{ep+1}/{epochs}] - NT-Xent Loss: {avg_loss:.4f}")

    # --- INFERENCE STAGE ---
    print("[*] Extracting Task and Rest MetricBolT Embeddings...")
    bolt_model.eval()
    
    rest_embs = []
    task_embs = []
    
    with torch.no_grad():
        for i in range(num_subjects):
            rx = rest_tensor[i:i+1].to(device)
            tx = task_tensor[i:i+1].to(device)
            
            re_cls = bolt_model(rx)
            te_cls = bolt_model(tx)
            
            rest_embs.append(re_cls.cpu().numpy().squeeze())
            task_embs.append(te_cls.cpu().numpy().squeeze())
            
    rest_embs = np.array(rest_embs)
    task_embs = np.array(task_embs)

    print(f"[DEBUG] bolt_model output shape: {rest_embs.shape}")
    print(f"[DEBUG] bolt_model rest_embs mean/std: {rest_embs.mean():.4f} / {rest_embs.std():.4f}")
    
    # Compute Raw FC Baseline using the Exact Finn et al Data Vectorization
    raw_rest = []
    raw_task = []
    triu_idx = np.triu_indices(360, k=1)
    
    for i in range(num_subjects):
        raw_rest.append(np.corrcoef(rest_tensor[i].numpy())[triu_idx])
        raw_task.append(np.corrcoef(task_tensor[i].numpy())[triu_idx])
    
    raw_rest = np.array(raw_rest)
    raw_task = np.array(raw_task)
    
    # Pearson Correlation of the 64,620 dimensional edges
    raw_matrix = np.corrcoef(raw_task, raw_rest)[:num_subjects, num_subjects:]
    raw_acc = calculate_accuracy(raw_matrix)
    print(f"[DEBUG] True Raw Direct FC Top-1 Acc (Finn et al): {raw_acc:.4f}")

    # Compute NxN Connectivity Simulation matrix based on Pearson correlation
    # Mean centering the metric embeddings ensures accurate cosine projection 
    print("[*] Calculating Evaluation Metrics for Comparison...")
    
    # Pearson Correlation for final ID Matrix to maintain metric invariance
    id_matrix = np.corrcoef(task_embs, rest_embs)[:num_subjects, num_subjects:]

    # Output using exact `kaggle_brain_fingerprinting` struct
    metrics = compute_all_metrics(id_matrix)
    
    print("\n" + "=" * 50)
    print("METRICBOLT BASELINE RESULTS")
    print("=" * 50)
    print(f"{'Method':<25} {'Acc':<10} {'Top-5':<10} {'MRR':<10}")
    print("-" * 50)
    print(f"{'MetricBolT (NT-Xent)':<25} {metrics['top_1_accuracy']:.4f}    {metrics['top_5_accuracy']:.4f}    {metrics['mrr']:.4f}")
    
    print("\nComprehensive Metrics:")
    print("-" * 50)
    print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {metrics['top_10_accuracy']:.4f}")
    print(f"Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"Mean Reciprocal Rank: {metrics['mrr']:.4f}")
    print(f"Differential Identifiability: {metrics['differential_id']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetricBolT Direct Comparison Script")
    parser.add_argument("--task", type=str, default="motor", help="Task name to benchmark")
    parser.add_argument("--num_subjects", type=int, default=100, help="Number of subjects")
    parser.add_argument("--epochs", type=int, default=20, help="BolT training epochs")
    parser.add_argument("--dynamic_length", type=int, default=284, help="Timeseries length crop")
    
    # Use parse_known_args to ignore arbitrary Jupyter kernel arguments like -f
    args, unknown = parser.parse_known_args()
    
    run_comparison(task_name=args.task, num_subjects=args.num_subjects, dynamic_length=args.dynamic_length, epochs=args.epochs)
