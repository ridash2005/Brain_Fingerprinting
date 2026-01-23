"""
Evaluation Metrics Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 5: Request for more comprehensive metrics beyond simple accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.matrix_ops import calculate_accuracy


def calculate_top_k_accuracy(corr_matrix: np.ndarray, k: int = 5) -> float:
    """Calculate top-k identification accuracy."""
    n = corr_matrix.shape[0]
    correct = 0
    for i in range(n):
        # argsort returns lowest to highest, so take last k
        top_k_indices = np.argsort(corr_matrix[i, :])[-k:]
        if i in top_k_indices:
            correct += 1
    return correct / n


def calculate_mean_rank(corr_matrix: np.ndarray) -> float:
    """Calculate mean rank of the correct identity (1 is best)."""
    n = corr_matrix.shape[0]
    ranks = []
    for i in range(n):
        # Sort descending
        sorted_indices = np.argsort(corr_matrix[i, :])[::-1]
        # Find where i is
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
    return np.mean(ranks)


def calculate_mean_reciprocal_rank(corr_matrix: np.ndarray) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    n = corr_matrix.shape[0]
    mrr_sum = 0.0
    for i in range(n):
        sorted_indices = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_indices == i)[0][0] + 1
        mrr_sum += 1.0 / rank
    return mrr_sum / n


def calculate_differential_identifiability(corr_matrix: np.ndarray) -> float:
    """Calculate Differential Identifiability (I_diff)."""
    n = corr_matrix.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    self_corr = corr_matrix[diag_mask]
    other_corr = corr_matrix[~diag_mask]
    return np.mean(self_corr) - np.mean(other_corr)



class ComprehensiveMetrics:
    """Compute and visualize various fingerprinting metrics."""
    
    def __init__(self, correlation_matrix: np.ndarray):
        self.corr_matrix = correlation_matrix
        self.n = correlation_matrix.shape[0]
        self.results = {}
        
    def compute_accuracy_metrics(self) -> Dict:
        """Compute standard identification accuracy."""
        accuracy = calculate_accuracy(self.corr_matrix)
        
        # Identification Rank
        ranks = []
        reciprocal_ranks = []
        for i in range(self.n):
            row = self.corr_matrix[i, :]
            target_val = row[i]
            # strict inequality for rank
            rank = np.sum(row > target_val) + 1
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
        
        self.results['accuracy'] = accuracy
        self.results['top_1_accuracy'] = accuracy
        self.results['top_3_accuracy'] = calculate_top_k_accuracy(self.corr_matrix, k=3)
        self.results['top_5_accuracy'] = calculate_top_k_accuracy(self.corr_matrix, k=5)
        self.results['top_10_accuracy'] = calculate_top_k_accuracy(self.corr_matrix, k=10)
        self.results['mean_rank'] = np.mean(ranks)
        self.results['median_rank'] = np.median(ranks)
        self.results['mean_reciprocal_rank'] = np.mean(reciprocal_ranks)
        return self.results

    def compute_discriminability_metrics(self) -> Dict:
        """Compute metrics related to distribution separation."""
        diag_mask = np.eye(self.n, dtype=bool)
        self_corr = self.corr_matrix[diag_mask]
        other_corr = self.corr_matrix[~diag_mask]
        
        # I_diff metric (Amico & Goñi, 2018)
        idiff = np.mean(self_corr) - np.mean(other_corr)
        self.results['i_diff'] = idiff
        self.results['differential_identifiability'] = idiff
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(self_corr) + np.var(other_corr)) / 2)
        self.results['cohens_d'] = (np.mean(self_corr) - np.mean(other_corr)) / pooled_std
        
        return self.results

    def compute_all_metrics(self) -> Dict:
        """Compute all available metrics."""
        self.compute_accuracy_metrics()
        self.compute_discriminability_metrics()
        return self.results

    def plot_metrics(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Similarity Distributions
        plt.figure(figsize=(10, 6))
        diag_mask = np.eye(self.n, dtype=bool)
        sns.kdeplot(self.corr_matrix[diag_mask], label='Self-Similarity', fill=True)
        sns.kdeplot(self.corr_matrix[~diag_mask], label='Other-Similarity', fill=True)
        plt.title('Distribution of Correlations')
        plt.xlabel('Pearson Correlation')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'))
        plt.close()

    def generate_report(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write("COMPREHENSIVE EVALUATION METRICS\n")
            f.write("=" * 40 + "\n")
            for metric, value in self.results.items():
                f.write(f"{metric:20}: {value:.4f}\n")


def run_evaluation_pipeline(correlation_matrix, output_dir):
    metrics = ComprehensiveMetrics(correlation_matrix)
    metrics.compute_accuracy_metrics()
    metrics.compute_discriminability_metrics()
    metrics.plot_metrics(output_dir)
    metrics.generate_report(os.path.join(output_dir, 'metrics_report.txt'))
    return metrics.results
