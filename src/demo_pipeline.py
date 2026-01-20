import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import orthogonal_mp
from numpy.linalg import svd
from datetime import datetime
from tqdm import tqdm

# Setup timestamped output directory
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("results", "runs", f"{RUN_ID}_demo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
N_SUBJECTS = 20
N_PARCELS = 100  # Scaled down for faster Kaggle/Demo execution (Standard is 360)
BATCH_SIZE = 8
EPOCHS = 10
K_SVD_ITER = 5
L_SPARSITY = 2
K_ATOMS = 5

# ==========================================
# 2. MODELS (CONVOLUTIONAL AUTOENCODER)
# ==========================================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        # Handle interpolation for arbitrary sizes if needed, but here we assume divisible by 4
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ==========================================
# 3. UTILITIES & SPARSE DICTIONARY LEARNING
# ==========================================
def reconstruct_symmetric_matrix(lower_triangular_values, n_parcels):
    reconstructed_matrix = np.zeros((n_parcels, n_parcels))
    index = 0
    for i in range(1, n_parcels):
        for j in range(i):
            reconstructed_matrix[i, j] = lower_triangular_values[index]
            index += 1
    reconstructed_matrix += reconstructed_matrix.T
    np.fill_diagonal(reconstructed_matrix, 1)
    return reconstructed_matrix

def calculate_accuracy(correlation_matrix):
    num_correct = 0
    num_items = correlation_matrix.shape[0]
    for i in range(num_items):
        if i < correlation_matrix.shape[1] and correlation_matrix[i, i] == np.max(correlation_matrix[i, :]):
            num_correct += 1
    return num_correct / num_items

# K-SVD Components
def omp_sparse_coding(Y, D, L):
    X = np.zeros((D.shape[1], Y.shape[1]))
    for i in range(Y.shape[1]):
        X[:, i] = orthogonal_mp(D, Y[:, i], n_nonzero_coefs=L)
    return X

def update_dictionary(Y, D, X):
    for k in range(D.shape[1]):
        non_zero_indices = np.nonzero(X[k, :])[0]
        if len(non_zero_indices) == 0: continue
        residual = Y[:, non_zero_indices] - np.dot(D, X[:, non_zero_indices])
        residual += np.outer(D[:, k], X[k, non_zero_indices])
        U, S, Vt = svd(residual, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, non_zero_indices] = S[0] * Vt[0, :]
    return D, X

def k_svd(Y, K, L, n_iter=10):
    m, n = Y.shape
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0)
    for iteration in range(n_iter):
        X = omp_sparse_coding(Y, D, L)
        D, X = update_dictionary(Y, D, X)
    return D, X

# ==========================================
# 4. SYNTHETIC DATA GENERATION
# ==========================================
def generate_synthetic_fc(n_subjects, n_parcels):
    # Common population pattern
    base_pattern = np.random.rand(n_parcels, n_parcels) * 0.2
    base_pattern = (base_pattern + base_pattern.T) / 2
    
    # Subject specific fingerprints
    fingerprints = [np.random.rand(n_parcels, n_parcels) * 0.5 for _ in range(n_subjects)]
    fingerprints = [(f + f.T) / 2 for f in fingerprints]
    
    fc_rest = []
    fc_task = []
    
    for i in range(n_subjects):
        # Rest: Base + Fingerprint + Noise
        noise_rest = np.random.randn(n_parcels, n_parcels) * 0.05
        noise_rest = (noise_rest + noise_rest.T) / 2
        r = base_pattern + fingerprints[i] + noise_rest
        np.fill_diagonal(r, 1)
        fc_rest.append(np.clip(r, -1, 1))
        
        # Task: Base + Fingerprint + TaskShift + Noise
        task_shift = np.random.rand(n_parcels, n_parcels) * 0.3 # Strong common task component
        task_shift = (task_shift + task_shift.T) / 2
        noise_task = np.random.randn(n_parcels, n_parcels) * 0.05
        noise_task = (noise_task + noise_task.T) / 2
        t = base_pattern + fingerprints[i] + task_shift + noise_task
        np.fill_diagonal(t, 1)
        fc_task.append(np.clip(t, -1, 1))
        
    return np.array(fc_rest), np.array(fc_task)

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================
def main():
    print("Initializing Brain Fingerprinting End-to-End Pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate Data
    fc_rest, fc_task = generate_synthetic_fc(N_SUBJECTS, N_PARCELS)
    print(f"Generated synthetic data for {N_SUBJECTS} subjects.")

    # Prepare for PyTorch
    rest_tensor = torch.tensor(fc_rest[:, np.newaxis, :, :], dtype=torch.float32)
    task_tensor = torch.tensor(fc_task[:, np.newaxis, :, :], dtype=torch.float32)
    
    dataset = TensorDataset(rest_tensor, rest_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5.1 Train ConvAutoencoder
    print("\n--- Phase 1: Training ConvAutoencoder ---")
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # 5.2 Refinement Workflow
    print("\n--- Phase 2: Refinement (ConvAE + SDL) ---")
    model.eval()
    with torch.no_grad():
        task_reconstr = model(task_tensor.to(device)).cpu()
    
    # Calculate Residuals (Denoising)
    task_residual = task_tensor.squeeze(1) - task_reconstr.squeeze(1)
    
    # Apply SDL (Sparse Dictionary Learning)
    # Extract lower triangular part
    n_tri = int(N_PARCELS * (N_PARCELS - 1) / 2)
    Y = np.zeros((n_tri, N_SUBJECTS))
    tril_indices = np.tril_indices(N_PARCELS, k=-1)
    for i in range(N_SUBJECTS):
        Y[:, i] = task_residual[i].numpy()[tril_indices]

    print("Running K-SVD...")
    D, X = k_svd(Y, K_ATOMS, L_SPARSITY, n_iter=K_SVD_ITER)
    sdl_reconstruction = np.dot(D, X).transpose()
    
    DX = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
    for i in range(N_SUBJECTS):
        DX[i] = reconstruct_symmetric_matrix(sdl_reconstruction[i], N_PARCELS)
        
    # Final Refined Signal
    task_refined = task_residual.numpy() - DX

    # 5.3 Evaluation
    print("\n--- Phase 3: Evaluation & Correlation Analysis ---")
    
    # Baseline: Task vs Rest
    corr_raw = np.corrcoef(fc_task.reshape(N_SUBJECTS, -1), fc_rest.reshape(N_SUBJECTS, -1))[:N_SUBJECTS, N_SUBJECTS:]
    # After ConvAE
    corr_convae = np.corrcoef(task_residual.reshape(N_SUBJECTS, -1).numpy(), fc_rest.reshape(N_SUBJECTS, -1))[:N_SUBJECTS, N_SUBJECTS:]
    # After Full Pipeline
    corr_refined = np.corrcoef(task_refined.reshape(N_SUBJECTS, -1), fc_rest.reshape(N_SUBJECTS, -1))[:N_SUBJECTS, N_SUBJECTS:]

    acc_raw = calculate_accuracy(corr_raw)
    acc_convae = calculate_accuracy(corr_convae)
    acc_refined = calculate_accuracy(corr_refined)

    print(f"Identification Accuracy (Raw): {acc_raw*100:.2f}%")
    print(f"Identification Accuracy (After ConvAE): {acc_convae*100:.2f}%")
    print(f"Identification Accuracy (After ConvAE+SDL): {acc_refined*100:.2f}%")

    # 5.4 Visualization
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(corr_raw, cmap='coolwarm')
    plt.title("Correlation: Task vs Rest (Raw)")
    
    plt.subplot(1, 3, 3)
    sns.heatmap(corr_refined, cmap='coolwarm')
    plt.title("Correlation: Task vs Rest (Refined)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "demo_results.png"))
    print(f"\nVisualizations saved to '{os.path.join(OUTPUT_DIR, 'demo_results.png')}'")
    print("Seamless execution completed.")

if __name__ == "__main__":
    main()
