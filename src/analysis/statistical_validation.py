"""
Statistical Validation Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 5: Don't provide p-values or confidence intervals
- Reviewer 1, Point 5: Suggest methods like permutation tests or McNemar's test
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.matrix_ops import calculate_accuracy


def permutation_test(
    accuracy_model: float,
    accuracy_baseline: float,
    correlation_matrix: np.ndarray,
    n_permutations: int = 200,
    random_state: Optional[int] = 42
) -> Dict[str, float]:
    """
    Perform permutation test to assess statistical significance.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_subjects = correlation_matrix.shape[0]
    observed_diff = accuracy_model - accuracy_baseline
    
    null_diffs = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(n_subjects)
        perm_matrix = correlation_matrix[perm_idx, :][:, perm_idx]
        perm_accuracy = calculate_accuracy(perm_matrix)
        null_diffs.append(perm_accuracy - accuracy_baseline)
    
    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'null_distribution': null_diffs
    }


def bootstrap_confidence_interval(
    fc_task: np.ndarray,
    fc_rest: np.ndarray,
    fingerprint_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42
) -> Dict[str, float]:
    """
    Compute confidence intervals for accuracy using bootstrapping.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_subjects = fc_task.shape[0]
    bootstrap_accuracies = []
    
    for _ in range(n_bootstrap):
        resample_idx = np.random.choice(n_subjects, n_subjects, replace=True)
        task_b = fc_task[resample_idx]
        rest_b = fc_rest[resample_idx]
        acc = fingerprint_func(task_b, rest_b)
        bootstrap_accuracies.append(acc)
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_accuracies, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_accuracies, (1 - alpha/2) * 100)
    
    return {
        'mean': np.mean(bootstrap_accuracies),
        'std': np.std(bootstrap_accuracies),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }



def paired_t_test(metrics1: np.ndarray, metrics2: np.ndarray) -> Dict[str, float]:
    """Perform paired t-test on results."""
    t_stat, p_val = stats.ttest_rel(metrics1, metrics2)
    mean_diff = np.mean(metrics1) - np.mean(metrics2)
    std_diff = np.std(metrics1 - metrics2, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0.0
    
    return {
        't_statistic': t_stat,
        'p_value': p_val,
        'effect_size': cohens_d
    }


def comprehensive_statistical_report(
    results: Dict[str, Dict],
    output_path: str
) -> None:
    """
    Generate a comprehensive statistical report.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE STATISTICAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if 'permutation' in results:
            perm = results['permutation']
            f.write("1. PERMUTATION TEST\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Observed difference: {perm['observed_diff']:.4f}\n")
            f.write(f"   P-value: {perm['p_value']:.6f}\n")
            f.write(f"   Significance (alpha=0.05): {'Yes' if perm['p_value'] < 0.05 else 'No'}\n\n")
        
        if 'bootstrap' in results:
            boot = results['bootstrap']
            f.write("2. BOOTSTRAP CONFIDENCE INTERVALS\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Mean accuracy: {boot['mean']:.4f}\n")
            f.write(f"   Standard deviation: {boot['std']:.4f}\n")
            f.write(f"   95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]\n\n")
