"""
Cross-Validation Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 5: Lack of cross-validation
- Reviewer 1, Point 6: Overfitting and data leakage concerns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.conv_ae import ConvAutoencoder
from models.sparse_dictionary_learning import k_svd, omp_sparse_coding
from utils.matrix_ops import calculate_accuracy, reconstruct_symmetric_matrix


class CrossValidation:
    """Implement k-fold cross-validation for the fingerprinting pipeline."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.results = []
        
    def _train_autoencoder(self, train_data: np.ndarray, val_data: np.ndarray, 
                          epochs: int = 20, batch_size: int = 16) -> ConvAutoencoder:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvAutoencoder().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_tensor = torch.from_numpy(train_data).float().unsqueeze(1).to(device)
        dataset = TensorDataset(train_tensor, train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
        return model

    def run_cross_validation(self, fc_task: np.ndarray, fc_rest: np.ndarray, 
                           output_dir: str, K: int = 15, L: int = 12) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        accuracies = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(fc_task)):
            print(f"Running Fold {fold+1}/{self.n_splits}...")
            
            # Split
            train_task = fc_task[train_idx]
            test_task = fc_task[test_idx]
            test_rest = fc_rest[test_idx]
            
            # 1. Train ConvAE
            model = self._train_autoencoder(train_task, None, epochs=10)
            model.eval()
            
            # 2. Compute Residuals
            with torch.no_grad():
                train_tensor = torch.from_numpy(train_task).float().unsqueeze(1).to(device)
                test_tensor = torch.from_numpy(test_task).float().unsqueeze(1).to(device)
                
                train_recon = model(train_tensor).cpu().numpy().squeeze()
                test_recon = model(test_tensor).cpu().numpy().squeeze()
            
            res_train = train_task - train_recon
            res_test = test_task - test_recon
            
            # 3. K-SVD on Train
            # Flatten lower triangle
            n_parcels = train_task.shape[1]
            tril_idx = np.tril_indices(n_parcels, k=-1)
            
            Y_train = np.array([res[tril_idx] for res in res_train]).T
            Y_test = np.array([res[tril_idx] for res in res_test]).T
            
            # Learn Dictionary
            D_train, _ = k_svd(Y_train, K, L, n_iter=5, verbose=False, random_state=42)
            
            # 4. Refine Test
            X_test = omp_sparse_coding(Y_test, D_train, L)
            shared_noise = (D_train @ X_test).T
            
            refined_test = []
            for i in range(len(test_task)):
                 # Reconstruct noise matrix
                 noise_mat = reconstruct_symmetric_matrix(shared_noise[i], n_parcels)
                 # Subtract noise from residual
                 refined_test.append(res_test[i] - noise_mat)
            
            refined_test = np.array(refined_test)
            
            # 5. Calculate Accuracy against Rest
            corr = np.corrcoef(refined_test.reshape(len(refined_test), -1), 
                               test_rest.reshape(len(test_rest), -1))[:len(test_task), len(test_task):]
            
            acc = calculate_accuracy(corr)
            accuracies.append(acc)
            print(f"Fold {fold+1} Accuracy: {acc:.4f}")
            
        self.results = accuracies
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # Generate summary
        self.generate_cv_report(os.path.join(output_dir, 'cv_report.txt'), mean_acc, std_acc)
        self.plot_fold_results(os.path.join(output_dir, 'cv_folds.png'))
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': accuracies
        }
        
    def generate_cv_report(self, output_path: str, mean_acc: float, std_acc: float) -> None:
        with open(output_path, 'w') as f:
            f.write("CROSS-VALIDATION REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Number of splits: {self.n_splits}\n")
            f.write(f"Fold Accuracies: {self.results}\n")
            f.write(f"Mean Accuracy: {mean_acc:.4f}\n")
            f.write(f"Std Deviation: {std_acc:.4f}\n")
            f.write(f"95% CI: [{mean_acc - 1.96*std_acc/np.sqrt(self.n_splits):.4f}, "
                    f"{mean_acc + 1.96*std_acc/np.sqrt(self.n_splits):.4f}]\n")
            
    def plot_fold_results(self, output_path: str) -> None:
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, self.n_splits + 1), self.results)
        plt.axhline(y=np.mean(self.results), color='r', linestyle='--', label=f'Mean: {np.mean(self.results):.4f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('K-Fold Cross-Validation Results')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.savefig(output_path)
        plt.close()
