#!/usr/bin/env python3
"""
CVAE+SDL Baseline vs. Custom Architecture Comparison

This template shows how to compare your custom architecture against the baseline.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Paths to results
BASELINE_RESULTS = "cvae_sdl_results/cvae_sdl_*/results.json"  # Latest baseline run
CUSTOM_RESULTS = "path/to/your/custom/results.json"  # Your architecture results

METRICS = ['top_1_accuracy', 'top_5_accuracy', 'mrr', 'differential_identifiability']

# ==========================================
# LOAD RESULTS
# ==========================================

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_metrics(results):
    """Extract average metrics from results."""
    avg_metrics = results.get('average_metrics', {})
    return avg_metrics

# ==========================================
# COMPARISON TABLE
# ==========================================

def print_comparison_table(baseline_metrics, custom_metrics):
    """Print side-by-side comparison."""
    
    print("\n" + "="*70)
    print(" CVAE+SDL BASELINE vs. CUSTOM ARCHITECTURE")
    print("="*70)
    print()
    print(f"{'Metric':<35} {'Baseline':<15} {'Custom':<15} {'Difference':<15}")
    print("-"*70)
    
    for metric in METRICS:
        baseline_val = baseline_metrics.get(metric, 0.0)
        custom_val = custom_metrics.get(metric, 0.0)
        diff = custom_val - baseline_val
        
        # Determine if improvement or decrease
        if metric == 'differential_identifiability':
            better = "↑" if diff > 0 else "↓"
        else:
            better = "↑" if diff > 0 else "↓"
        
        print(f"{metric:<35} {baseline_val:.4f}        {custom_val:.4f}        {better} {diff:+.4f}")
    
    print()

# ==========================================
# VISUALIZATION
# ==========================================

def plot_comparison(baseline_results, custom_results, output_path="comparison.png"):
    """Create comparison visualization."""
    
    baseline_metrics = extract_metrics(baseline_results)
    custom_metrics = extract_metrics(custom_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [k for k in METRICS if k in baseline_metrics]
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
        baseline_val = baseline_metrics.get(metric, 0.0)
        custom_val = custom_metrics.get(metric, 0.0)
        
        x = ['Baseline\n(CVAE+SDL)', 'Custom\nArchitecture']
        y = [baseline_val, custom_val]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(y) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # Improvement indicator
        diff = custom_val - baseline_val
        status = "✓ Better" if diff > 0 else "✕ Worse"
        ax.set_title(f"{metric}\n{status} ({diff:+.4f})", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()

# ==========================================
# DETAILED PAIR ANALYSIS
# ==========================================

def compare_state_pairs(baseline_results, custom_results):
    """Compare performance across state pairs."""
    
    baseline_pairs = baseline_results.get('all_pairs', {})
    custom_pairs = custom_results.get('all_pairs', {})
    
    print("\n" + "="*70)
    print(" STATE PAIR COMPARISON (Top-1 Accuracy)")
    print("="*70)
    print()
    print(f"{'State Pair':<25} {'Baseline':<12} {'Custom':<12} {'Difference':<12}")
    print("-"*70)
    
    improvements = []
    decreases = []
    
    for pair in sorted(baseline_pairs.keys()):
        baseline_acc = baseline_pairs.get(pair, {}).get('top_1_accuracy', 0.0)
        custom_acc = custom_pairs.get(pair, {}).get('top_1_accuracy', 0.0)
        diff = custom_acc - baseline_acc
        
        status = "↑" if diff > 0 else "↓"
        print(f"{pair:<25} {baseline_acc:.4f}       {custom_acc:.4f}       {status} {diff:+.4f}")
        
        if diff > 0:
            improvements.append((pair, diff))
        else:
            decreases.append((pair, abs(diff)))
    
    print()
    print(f"Improvements: {len(improvements)}/{len(baseline_pairs)}")
    if improvements:
        top_improvement = max(improvements, key=lambda x: x[1])
        print(f"  Best improvement: {top_improvement[0]} (+{top_improvement[1]:.4f})")
    
    print(f"\nDecreases: {len(decreases)}/{len(baseline_pairs)}")
    if decreases:
        top_decrease = max(decreases, key=lambda x: x[1])
        print(f"  Largest decrease: -{top_decrease[1]:.4f}")

# ==========================================
# STATISTICAL SUMMARY
# ==========================================

def summary_statistics(baseline_results, custom_results):
    """Print summary statistics."""
    
    baseline_metrics = extract_metrics(baseline_results)
    custom_metrics = extract_metrics(custom_results)
    
    print("\n" + "="*70)
    print(" SUMMARY STATISTICS")
    print("="*70)
    
    # Overall improvement
    baseline_acc = baseline_metrics.get('top_1_accuracy', 0.0)
    custom_acc = custom_metrics.get('top_1_accuracy', 0.0)
    improvement = ((custom_acc - baseline_acc) / baseline_acc) * 100
    
    print(f"\nTop-1 Accuracy Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"  ✓ Custom architecture OUTPERFORMS baseline")
    elif improvement < -5:
        print(f"  ✕ Custom architecture significantly underperforms baseline")
    else:
        print(f"  ~ Similar performance (within measurement noise)")
    
    # Report configuration
    print(f"\nConfiguration:")
    print(f"  Subjects: {baseline_results.get('config', {}).get('num_subjects', 'N/A')}")
    print(f"  Train/Test Split: {baseline_results.get('config', {}).get('n_train', 'N/A')} / {baseline_results.get('config', {}).get('n_test', 'N/A')}")
    print(f"  Baseline Epochs: {baseline_results.get('config', {}).get('epochs', 'N/A')}")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import sys
    from glob import glob
    
    # Auto-detect latest baseline results
    baseline_files = sorted(glob("cvae_sdl_results/cvae_sdl_*/results.json"))
    if not baseline_files:
        print("[ERROR] No baseline results found in cvae_sdl_results/")
        sys.exit(1)
    
    baseline_path = baseline_files[-1]  # Latest
    print(f"[1] Loading baseline results: {baseline_path}")
    baseline_results = load_results(baseline_path)
    
    # Get custom results path
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
    else:
        custom_path = input("\nEnter path to custom architecture results JSON: ").strip()
    
    if not Path(custom_path).exists():
        print(f"[ERROR] File not found: {custom_path}")
        sys.exit(1)
    
    print(f"[2] Loading custom architecture results: {custom_path}")
    custom_results = load_results(custom_path)
    
    # Extract metrics
    baseline_metrics = extract_metrics(baseline_results)
    custom_metrics = extract_metrics(custom_results)
    
    # Print comparison
    print_comparison_table(baseline_metrics, custom_metrics)
    
    # Detailed comparison
    compare_state_pairs(baseline_results, custom_results)
    
    # Summary
    summary_statistics(baseline_results, custom_results)
    
    # Visualization
    plot_comparison(baseline_results, custom_results)
    
    print("\n" + "="*70)
    print(" Comparison Complete!")
    print("="*70 + "\n")
