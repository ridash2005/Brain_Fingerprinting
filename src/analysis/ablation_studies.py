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
        """Baseline performance using raw FC matrices."""
        n = self.n_subjects
        task_flat = self.fc_task.reshape(n, -1)
        rest_flat = self.fc_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['Raw FC'] = acc
        return acc
    
    def convae_only(self, cae_features_task: np.ndarray, cae_features_rest: np.ndarray) -> float:
        """Performance using only ConvAE features."""
        n = self.n_subjects
        task_flat = cae_features_task.reshape(n, -1)
        rest_flat = cae_features_rest.reshape(n, -1)
        corr_matrix = np.corrcoef(task_flat, rest_flat)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['ConvAE Only'] = acc
        return acc
        
    def sdl_only(self, sdl_features_task: np.ndarray, sdl_features_rest: np.ndarray) -> float:
        """Performance using only SDL features (applied to raw FC)."""
        n = self.n_subjects
        corr_matrix = np.corrcoef(sdl_features_task, sdl_features_rest)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['SDL Only'] = acc
        return acc
        
    def full_pipeline(self, final_features_task: np.ndarray, final_features_rest: np.ndarray) -> float:
        """Performance of the full ConvAE + SDL pipeline."""
        n = self.n_subjects
        corr_matrix = np.corrcoef(final_features_task, final_features_rest)[:n, n:]
        acc = calculate_accuracy(corr_matrix)
        self.results['Full Pipeline'] = acc
        return acc
        
    def plot_results(self, output_path: str) -> None:
        names = list(self.results.keys())
        values = list(self.results.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'])
        plt.ylabel('Fingerprinting Accuracy')
        plt.title('Ablation Study: Component Contributions')
        plt.ylim(0, 1.05)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def generate_report(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write("ABLATION STUDY REPORT\n")
            f.write("=" * 40 + "\n")
            for name, acc in self.results.items():
                f.write(f"{name:20}: {acc:.4f}\n")


def run_all_ablations(fc_task, fc_rest, cae_task, cae_rest, sdl_raw_task, sdl_raw_rest, full_task, full_rest, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    study = AblationStudy(fc_task, fc_rest)
    study.raw_fc_baseline()
    study.convae_only(cae_task, cae_rest)
    study.sdl_only(sdl_raw_task, sdl_raw_rest)
    study.full_pipeline(full_task, full_rest)
    study.plot_results(os.path.join(output_dir, 'ablation_results.png'))
    study.generate_report(os.path.join(output_dir, 'ablation_report.txt'))
    return study.results
