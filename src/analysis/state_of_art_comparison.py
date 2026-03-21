"""
State-of-the-art Comparison Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 3: Comparison with established methods
- Reviewer 1, Point 3: Include methods like Finn et al. (2015) and others
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.matrix_ops import calculate_accuracy


class FinnFingerprinting:
    """Classic fingerprinting by Finn et al. (2015)."""
    
    def run(self, fc_task: np.ndarray, fc_rest: np.ndarray) -> float:
        n = fc_task.shape[0]
        task_flat = fc_task.reshape(n, -1)
        rest_flat = fc_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        return calculate_accuracy(corr_matrix)


class EdgeSelectionFingerprinting:
    """Fingerprinting using high-discriminative edges (e.g., Shen et al., 2017)."""
    
    def run(self, fc_task: np.ndarray, fc_rest: np.ndarray, top_k: float = 0.05) -> float:
        n, p, _ = fc_task.shape
        # Compute edge-wise discriminability on a toy subset or use simple variance
        # Real version would need cross-validation to select edges
        task_flat = fc_task.reshape(n, -1)
        rest_flat = fc_rest.reshape(n, -1)
        
        # Simple discriminability: variance across subjects
        variances = np.var(rest_flat, axis=0)
        top_edges = np.argsort(variances)[-int(top_k * variances.size):]
        
        task_sub = task_flat[:, top_edges]
        rest_sub = rest_flat[:, top_edges]
        
        corr_matrix = np.corrcoef(task_sub, rest_sub)[:n, n:]
        return calculate_accuracy(corr_matrix)


class PCAFingerprinting:
    """Fingerprinting using PCA-reconstructed matrices."""
    
    def run(self, fc_task: np.ndarray, fc_rest: np.ndarray, n_components: int = 50) -> float:
        n = fc_task.shape[0]
        task_flat = fc_task.reshape(n, -1)
        rest_flat = fc_rest.reshape(n, -1)
        
        pca = PCA(n_components=min(n_components, n))
        task_pca = pca.fit_transform(task_flat)
        rest_pca = pca.transform(rest_flat)
        
        corr_matrix = np.corrcoef(task_pca, rest_pca)[:n, n:]
        return calculate_accuracy(corr_matrix)


class SOTAComparison:
    """Orchestrate comparison with SOTA methods."""
    
    def __init__(self, fc_task: np.ndarray, fc_rest: np.ndarray):
        self.fc_task = fc_task
        self.fc_rest = fc_rest
        self.results = {}
        
    def run_all_comparisons(self, proposed_accuracy: float) -> Dict:
        # Finn et al.
        finn = FinnFingerprinting()
        self.results['Finn et al. (2015)'] = finn.run(self.fc_task, self.fc_rest)
        
        # Edge Selection
        edge = EdgeSelectionFingerprinting()
        self.results['Edge Selection'] = edge.run(self.fc_task, self.fc_rest)
        
        # PCA
        pca_fp = PCAFingerprinting()
        self.results['PCA-based'] = pca_fp.run(self.fc_task, self.fc_rest)
        
        # Proposed
        self.results['Proposed Method'] = proposed_accuracy
        
        return self.results
    
    def plot_comparison(self, output_path: str) -> None:
        methods = list(self.results.keys())
        accuracies = list(self.results.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['#95a5a6', '#95a5a6', '#95a5a6', '#e74c3c'] # Highlight proposed
        plt.bar(methods, accuracies, color=colors)
        plt.ylabel('Identification Accuracy')
        plt.title('Comparison with SOTA Methods')
        plt.ylim(0, 1.05)
        
        # Add values on top
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def generate_comparison_report(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write("SOTA COMPARISON REPORT\n")
            f.write("=" * 40 + "\n")
            for method, acc in self.results.items():
                f.write(f"{method:25}: {acc:.4f}\n")


def run_sota_comparison_pipeline(fc_task, fc_rest, proposed_acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    comp = SOTAComparison(fc_task, fc_rest)
    results = comp.run_all_comparisons(proposed_acc)
    comp.plot_comparison(os.path.join(output_dir, 'sota_comparison.png'))
    comp.generate_comparison_report(os.path.join(output_dir, 'sota_report.txt'))
    return results
