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


def permutation_test(acc_model, corr_matrix, n_permutations=1000):
    """
    Permutation test for statistical significance against RANDOM CHANCE.
    Null Hypothesis: The subject identity in Task does not match Rest.
    """
    n = corr_matrix.shape[0]
    
    null_accs = []
    for _ in tqdm(range(n_permutations), desc="Permutation test (Chance)"):
        perm = np.random.permutation(n)
        # Shuffle ONLY rows to break subject-identity correspondence
        perm_matrix = corr_matrix[perm, :]
        null_acc = calculate_accuracy(perm_matrix)
        null_accs.append(null_acc)
    
    # One-tailed test: Fraction of null accuracies >= observed accuracy
    p_value = (np.sum(np.array(null_accs) >= acc_model) + 1) / (n_permutations + 1)
    return p_value, null_accs


def paired_permutation_test(preds_model, preds_baseline, n_permutations=1000):
    """
    Paired permutation test (Approximate Randomization Test) to compare two models.
    Null Hypothesis: The two models have the same performance.
    """
    # Difference in mean accuracy
    obs_diff = np.mean(preds_model) - np.mean(preds_baseline)
    
    n = len(preds_model)
    null_diffs = []
    
    # Difference vector (1: Model better, -1: Baseline better, 0: Tie)
    diffs = preds_model.astype(float) - preds_baseline.astype(float)
    
    for _ in range(n_permutations):
        # Randomly sign-flip the differences
        signs = np.random.choice([-1, 1], size=n)
        null_diff = np.mean(diffs * signs)
        null_diffs.append(null_diff)
        
    # Two-tailed p-value
    p_value = (np.sum(np.abs(np.array(null_diffs)) >= np.abs(obs_diff)) + 1) / (n_permutations + 1)
    return p_value, null_diffs


def mcnemar_test(preds_model, preds_baseline):
    """
    Perform McNemar's test to compare two models.
    Uses Exact Binomial Test if discordant pairs < 25, otherwise Chi-squared.
    """
    # Contingency table
    # b: model correct, baseline incorrect
    # c: model incorrect, baseline correct
    b = np.sum(preds_model & ~preds_baseline)
    c = np.sum(~preds_model & preds_baseline)
    
    if b + c == 0:
        return 0.0, 1.0
        
    if b + c < 25:
        # Exact Binomial Test
        p_value = 2 * stats.binom.cdf(min(b, c), b + c, 0.5)
        statistic = 0.0 
    else:
        # Chi-squared with continuity correction
        statistic = (max(0, np.abs(b - c) - 1))**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
    return statistic, min(1.0, p_value)


def paired_t_test(metrics1, metrics2):
    """Perform paired t-test on results (e.g. across folds)."""
    t_stat, p_val = stats.ttest_rel(metrics1, metrics2)
    mean_diff = np.mean(metrics1) - np.mean(metrics2)
    # Cohen's d for effect size
    std_diff = np.std(metrics1 - metrics2, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0
    return t_stat, p_val, cohens_d


def bootstrap_ci(fc_task, fc_rest, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    n_subjects = fc_task.shape[0]
    accuracies = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = np.random.choice(n_subjects, n_subjects, replace=True)
        # Flatten for correlation
        task_sample = fc_task[idx].reshape(n_subjects, -1)
        rest_sample = fc_rest[idx].reshape(n_subjects, -1)
        corr = np.corrcoef(task_sample, rest_sample)[:n_subjects, n_subjects:]
        accuracies.append(calculate_accuracy(corr))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
    
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def comprehensive_statistical_report(results: Dict, output_path: str) -> None:
    """Generate a comprehensive statistical report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE STATISTICAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for key, val in results.items():
            f.write(f"{key.upper()}\n")
            f.write("-" * 40 + "\n")
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    f.write(f"   {k2:20}: {v2}\n")
            else:
                f.write(f"   Value: {val}\n")
            f.write("\n")

