"""
Robustness Analysis Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 10: Need robustness measures - how performance degrades with noise/sample size
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.conv_ae import ConvAutoencoder
from models.sparse_dictionary_learning import k_svd
from utils.matrix_ops import reconstruct_symmetric_matrix, calculate_accuracy


class RobustnessAnalysis:
    """Comprehensive robustness analysis for fingerprinting pipeline."""
    
    def __init__(self, fc_task: np.ndarray, fc_rest: np.ndarray,
                 model_path: Optional[str] = None):
        self.fc_task = fc_task
        self.fc_rest = fc_rest
        self.n_subjects = fc_task.shape[0]
        self.n_parcels = fc_task.shape[1]
        self.model_path = model_path
        self.results = {}
        
    def _compute_accuracy(self, fc_task: np.ndarray, fc_rest: np.ndarray) -> float:
        fc_task_flat = fc_task.reshape(fc_task.shape[0], -1)
        fc_rest_flat = fc_rest.reshape(fc_rest.shape[0], -1)
        n = fc_task_flat.shape[0]
        corr_matrix = np.corrcoef(fc_task_flat, fc_rest_flat)[:n, n:]
        return calculate_accuracy(corr_matrix)
    
    def noise_robustness(self, noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
                         n_repeats: int = 5) -> Dict:
        results = {level: [] for level in noise_levels}
        for noise_level in noise_levels:
            for _ in range(n_repeats):
                noise = np.random.randn(*self.fc_task.shape) * noise_level
                fc_noisy = self.fc_task + noise
                accuracy = self._compute_accuracy(fc_noisy, self.fc_rest)
                results[noise_level].append(accuracy)
        
        self.results['noise'] = {
            'noise_levels': noise_levels,
            'accuracies': {k: np.mean(v) for k, v in results.items()},
            'stds': {k: np.std(v) for k, v in results.items()}
        }
        return self.results['noise']
    
    def sample_size_robustness(self, sample_fractions: List[float] = [0.2, 0.5, 0.8, 1.0],
                               n_repeats: int = 5) -> Dict:
        results = {frac: [] for frac in sample_fractions}
        for fraction in sample_fractions:
            n_samples = max(5, int(self.n_subjects * fraction))
            for _ in range(n_repeats):
                idx = np.random.choice(self.n_subjects, n_samples, replace=False)
                fc_task_sub = self.fc_task[idx]
                fc_rest_sub = self.fc_rest[idx]
                accuracy = self._compute_accuracy(fc_task_sub, fc_rest_sub)
                results[fraction].append(accuracy)
        
        self.results['sample_size'] = {
            'fractions': sample_fractions,
            'sample_sizes': [max(5, int(self.n_subjects * f)) for f in sample_fractions],
            'accuracies': {k: np.mean(v) for k, v in results.items()},
            'stds': {k: np.std(v) for k, v in results.items()}
        }
        return self.results['sample_size']
    
    def missing_data_robustness(self, missing_fractions: List[float] = [0.0, 0.1, 0.2],
                                n_repeats: int = 5) -> Dict:
        results = {frac: [] for frac in missing_fractions}
        for fraction in missing_fractions:
            for _ in range(n_repeats):
                fc_missing = self.fc_task.copy()
                mask = np.random.random(self.fc_task.shape) < fraction
                fc_missing[mask] = 0
                accuracy = self._compute_accuracy(fc_missing, self.fc_rest)
                results[fraction].append(accuracy)
        
        self.results['missing'] = {
            'fractions': missing_fractions,
            'accuracies': {k: np.mean(v) for k, v in results.items()},
            'stds': {k: np.std(v) for k, v in results.items()}
        }
        return self.results['missing']
    
    def plot_robustness_results(self, output_path: str) -> None:
        plot_keys = [k for k in ['noise', 'sample_size', 'missing'] if k in self.results]
        if not plot_keys: return
        fig, axes = plt.subplots(1, len(plot_keys), figsize=(5*len(plot_keys), 4))
        if len(plot_keys) == 1: axes = [axes]
        
        for i, key in enumerate(plot_keys):
            data = self.results[key]
            if key == 'sample_size':
                x = data['sample_sizes']
            else:
                x = list(data['accuracies'].keys())
            y = list(data['accuracies'].values())
            yerr = list(data['stds'].values())
            axes[i].errorbar(x, y, yerr=yerr, marker='o')
            axes[i].set_title(f'{key.replace("_", " ").title()} Robustness')
            axes[i].set_ylabel('Accuracy')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_report(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write("ROBUSTNESS ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n")
            for key in self.results:
                f.write(f"\n{key.upper()} ANALYSIS:\n")
                for k, v in self.results[key]['accuracies'].items():
                    f.write(f"  {k}: {v:.4f} +/- {self.results[key]['stds'][k]:.4f}\n")
            
    def run_all_analyses(self, output_dir: str) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        self.noise_robustness()
        self.sample_size_robustness()
        self.missing_data_robustness()
        self.plot_robustness_results(os.path.join(output_dir, 'robustness_plots.png'))
        self.generate_report(os.path.join(output_dir, 'robustness_report.txt'))
        return self.results


def run_robustness_pipeline(fc_task_path, fc_rest_path, output_dir):
    fc_task = np.load(fc_task_path)
    fc_rest = np.load(fc_rest_path)
    analysis = RobustnessAnalysis(fc_task, fc_rest)
    return analysis.run_all_analyses(output_dir)
