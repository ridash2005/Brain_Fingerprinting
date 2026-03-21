"""
Ablation Studies Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 4: Impact of each component (ConvAE, SDL) on identification accuracy
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.matrix_ops import calculate_accuracy


class AblationStudy:
    """Run ablation studies to evaluate individual component contributions."""
    
    def __init__(self, fc_task: np.ndarray, fc_rest: np.ndarray):
        self.fc_task = fc_task
        self.fc_rest = fc_rest
        self.n_subjects = fc_task.shape[0]
        self.results = {}
        
    def raw_fc_baseline(self) -> float:
        """Baseline 1: Raw Functional Connectivity (Finn et al., 2015)."""
        n = self.n_subjects
        task_flat = self.fc_task.reshape(n, -1)
        rest_flat = self.fc_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['Raw FC'] = acc
        return acc

    def convae_only(self, cae_res_task: np.ndarray, cae_res_rest: np.ndarray) -> float:
        """Ablation 2: Convolutional Autoencoder Denoising only."""
        # Identification based on residuals from ConvAE
        n = self.n_subjects
        task_flat = cae_res_task.reshape(n, -1)
        rest_flat = cae_res_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['ConvAE Only'] = acc
        return acc
        
    def sdl_only(self, sdl_refined_task: np.ndarray, sdl_refined_rest: np.ndarray) -> float:
        """Ablation 3: Sparse Dictionary Learning only."""
        # Identification based on SDL-refined fingerprints (applied to raw FC)
        n = self.n_subjects
        task_flat = sdl_refined_task.reshape(n, -1)
        rest_flat = sdl_refined_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['SDL Only'] = acc
        return acc
        
    def full_pipeline(self, final_task: np.ndarray, final_rest: np.ndarray) -> float:
        """Ablation 4: Full Pipeline (ConvAE + SDL)."""
        # Identification based on final fingerprints
        n = self.n_subjects
        task_flat = final_task.reshape(n, -1)
        rest_flat = final_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['Full Pipeline'] = acc
        return acc
        
    def plot_results(self, output_path: str) -> None:
        names = list(self.results.keys())
        values = list(self.results.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71'] # Grey, Blue, Orange, Green
        bars = plt.bar(names, values, color=colors[:len(names)])
        plt.ylabel('Identification Accuracy')
        plt.title('Ablation Study: Broad Component Impact')
        plt.ylim(0, 1.05)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    def generate_report(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write("ABLATION STUDY REPORT (Broad Categories)\n")
            f.write("=" * 45 + "\n")
            # Sort manually for consistent output
            order = ['Raw FC', 'ConvAE Only', 'SDL Only', 'Full Pipeline']
            for name in order:
                if name in self.results:
                    f.write(f"{name:20}: {self.results[name]:.4f}\n")

def run_all_ablations(fc_task, fc_rest, cae_res_task, cae_res_rest, sdl_task, sdl_rest, full_task, full_rest, output_dir):
    """Run all ablation variants using pre-calculated features."""
    os.makedirs(output_dir, exist_ok=True)
    study = AblationStudy(fc_task, fc_rest)
    study.raw_fc_baseline()
    study.convae_only(cae_res_task, cae_res_rest)
    study.sdl_only(sdl_task, sdl_rest)
    study.full_pipeline(full_task, full_rest)
    study.plot_results(os.path.join(output_dir, 'ablation_results.png'))
    study.generate_report(os.path.join(output_dir, 'ablation_report.txt'))
    return study.results
