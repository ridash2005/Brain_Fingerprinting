# Analysis Modules for IEEE TCDS Reviewer Comments

## Overview

This directory contains comprehensive analysis modules created to address all experimental and implementation concerns raised by IEEE TCDS reviewers. Each module is designed to be run independently or as part of the complete analysis pipeline.

---

## Reviewer Comment to Analysis Mapping

### Reviewer 1 Comments

| Point | Comment Summary | Solution | File |
|-------|----------------|----------|------|
| 1 | Residual may contain noise, not meaningful signal | Interpretability analysis showing residual patterns | `interpretability.py` |
| 2 | Baseline poorly designed, missing SOTA comparison | Finn et al. (2015) implementation + other methods | `state_of_art_comparison.py` |
| 3 | No proper cross-validation, data leakage concerns | K-fold CV with proper train/test splits | `cross_validation.py` |
| 4 | Missing implementation details | Updated `models/conv_ae.py` and `models/sparse_dictionary_learning.py` with full specs |
| 5 | No statistical validation (p-values, CIs) | Permutation tests, bootstrap CIs | `statistical_validation.py` |
| 6 | Why convolutions for correlation matrices? | Justification added to `models/conv_ae.py` docstrings |
| 7 | No analysis of what AE learns, brain region contributions | Filter visualization, latent space, network contribution | `interpretability.py` |
| 8 | Overclaiming applications | N/A (manuscript issue) |
| 9 | Missing ablation studies | Complete ablation: Raw FC, ConvAE only, SDL only, full pipeline | `ablation_studies.py` |
| 10 | Only accuracy metric, missing robustness | Top-k, MRR, ranking + noise/sample size robustness | `evaluation_metrics.py`, `robustness_analysis.py` |

### Reviewer 2 Comments

| Point | Comment Summary | Solution | File |
|-------|----------------|----------|------|
| 1(1) | Role of SDL unclear | Enhanced documentation in `models/sparse_dictionary_learning.py` |
| 1(2) | Need ablation study, K/L optimization analysis | Ablation study + hyperparameter grid search | `ablation_studies.py`, `models/sparse_dictionary_learning.py` |
| 2 | No dataset description | Comprehensive HCP description | `dataset_description.py` |
| 3 | No SOTA comparison | Multiple comparison methods | `state_of_art_comparison.py` |
| 4 | Table format issues | N/A (manuscript issue) |

---

## Module Descriptions

### 1. `statistical_validation.py`
**Purpose:** Provide rigorous statistical validation for fingerprinting improvements.

**Key Functions:**
- `permutation_test()`: Assess significance of accuracy improvement via permutation
- `bootstrap_confidence_interval()`: Compute 95% CIs for accuracy estimates
- `mcnemar_test()`: Compare two fingerprinting methods statistically
- `paired_t_test()`: Compare across CV folds
- `multiple_comparison_correction()`: Bonferroni, Holm, FDR corrections

### 2. `ablation_studies.py`
**Purpose:** Systematic evaluation of each component's contribution.

**Configurations Tested:**
1. Raw FC (Finn et al. baseline)
2. Group average subtraction
3. ConvAE encoder features only
4. ConvAE residuals only (no SDL)
5. SDL only (on raw FC)
6. ConvAE + SDL (full pipeline)

### 3. `cross_validation.py`
**Purpose:** Proper train/test splitting to prevent data leakage.

**Features:**
- K-fold cross-validation with ConvAE trained per fold
- Nested CV for unbiased hyperparameter selection
- Leave-one-out CV for small sample validation
- Clear documentation of data leakage prevention protocol

### 4. `state_of_art_comparison.py`
**Purpose:** Compare against established fingerprinting methods.

**Methods Implemented:**
- Finn et al. (2015): Original FC correlation fingerprinting
- Edge Selection: Top discriminative edges
- PCA-based: Principal component projection
- Network-wise: Per-network identification

### 5. `interpretability.py`
**Purpose:** Understand what models learn.

**Analyses:**
- ConvAE filter visualization
- Latent space structure and subject discriminability
- Reconstruction vs. residual pattern analysis
- Dictionary atom visualization
- Network contribution to each atom
- Atom usage patterns and specificity

### 6. `robustness_analysis.py`
**Purpose:** Characterize performance under various conditions.

**Tests:**
- Noise robustness: Accuracy vs. noise level curves
- Sample size: Performance with reduced subjects
- Missing data: Tolerance to missing edges

### 7. `evaluation_metrics.py`
**Purpose:** Comprehensive evaluation beyond top-1 accuracy.

**Metrics:**
- Top-k accuracy (k=1,3,5,10)
- Mean rank and Mean Reciprocal Rank (MRR)
- Differential identifiability
- Intra-class correlation (ICC)
- Self vs. other correlation distributions with effect sizes

### 8. `dataset_description.py`
**Purpose:** Generate comprehensive dataset documentation.

**Contents:**
- HCP acquisition parameters
- Preprocessing pipeline details
- Subject selection criteria
- FC computation methodology
- Dataset statistics and visualizations

### 9. `run_complete_analysis.py`
**Purpose:** Master script to run all analyses.

**Usage:**
```bash
python run_complete_analysis.py --task motor --output_dir results/analysis
```

---

## Output Structure

Running the complete analysis generates:

```
results/complete_analysis/
├── 1_statistical_validation/
│   └── statistical_report.txt
├── 2_ablation_studies/
│   ├── ablation_report.txt
│   └── ablation_results.png
├── 3_cross_validation/
│   ├── cv_report.txt
│   └── cv_results.png
├── 4_sota_comparison/
│   ├── sota_report.txt
│   └── sota_comparison.png
├── 5_interpretability/
│   ├── convae_filters.png
│   ├── latent_analysis.png
│   ├── reconstruction.png
│   ├── dictionary_atoms.png
│   ├── atom_usage.png
│   └── networks.png
├── 6_robustness/
│   ├── robustness_report.txt
│   └── robustness_plots.png
├── 7_evaluation_metrics/
│   ├── metrics_report.txt
│   └── metrics_plots.png
├── 8_dataset_description/
│   ├── dataset_description.txt
│   └── fc_distributions.png
└── ANALYSIS_SUMMARY.txt
```

---

## Requirements

Additional dependencies for analysis modules:
```
scipy>=1.7.0
scikit-learn>=1.0.0
tqdm>=4.60.0
seaborn>=0.11.0
```

---

## Quick Start

```python
# Run complete analysis
from src.analysis.run_complete_analysis import main
results = main()

# Or run individual analyses
from src.analysis.ablation_studies import run_ablation_pipeline
ablation_results = run_ablation_pipeline(
    fc_task_path='FC_DATA/fc_motor.npy',
    fc_rest_path='FC_DATA/fc_rest.npy',
    model_path='src/models/trained/conv_ae_rest_best_model.pth',
    output_dir='results/ablation'
)
```
