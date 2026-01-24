# Comparison: `notebooks/hcp_kaggle.ipynb` vs. `notebooks/kaggle_brain_fingerprinting.py`

This document outlines the differences between the foundational data loader notebook (`notebooks/hcp_kaggle.ipynb`) and the advanced analysis pipeline script (`notebooks/kaggle_brain_fingerprinting.py`).

## 1. High-Level Summary

| Feature | `notebooks/hcp_kaggle.ipynb` | `notebooks/kaggle_brain_fingerprinting.py` |
| :--- | :--- | :--- |
| **Purpose** | Data Loading & Exploratory Analysis (EDA) | Research-Grade Brain Fingerprinting Pipeline |
| **Methodology** | Standard Functional Connectivity (Pearson Corr) | **Hybrid Model**: ConvAutoencoder + Sparse Dictionary Learning (K-SVD) |
| **Frameworks** | `numpy`, `matplotlib`, `nilearn` | `torch` (PyTorch), `scikit-learn`, `numpy` |
| **Validation** | Basic Visualization (Heatmaps) | **Rigorous**: Cross-Validation, Permutation Tests, Bootstrap CIs |
| **Output** | Single Run Plots | Manuscript-Ready Tables, Ablation Studies, Statistical Reports |

---

## 2. Detailed Technical Comparison

### A. Data Handling
*   **`notebooks/hcp_kaggle.ipynb`**:
    *   Focuses on *downloading* and *extracting* the HCP dataset (using `tarfile`, `requests`).
    *   Implements basic loaders (`load_timeseries`, `load_evs`) to process file paths into NumPy arrays.
    *   Computes FC matrices using simple `np.corrcoef`.
*   **`notebooks/kaggle_brain_fingerprinting.py`**:
    *   Assumes data is present (or uses the same loaders) but adds **Preprocessing**.
    *   Implements **Train/Test Splitting** (Inductive sets).
    *   Standardizes inputs for Neural Network training.

### B. Algorithmic Logic
*   **`notebooks/hcp_kaggle.ipynb`**:
    *   **Baseline Method**: Compares subjects based on raw correlation of time-series.
    *   Goal: "Does Subject A's motor task look like their resting state?" using raw correlations.
*   **`notebooks/kaggle_brain_fingerprinting.py`**:
    *   **Proposed Method**:
        1.  **ConvAutoencoder**: Compresses FC matrices to remove transient noise (non-fingerprint info).
        2.  **Residual Calculation**: Computes $R = X - \hat{X}$ (what the Autoencoder *didn't* capture).
        3.  **K-SVD (Dictionary Learning)**: Learns a dictionary of noise/artifact patterns from the Train set residuals.
        4.  **Denoising**: Subtracts the learned sparse noise components to refine the fingerprint.
    *   **Result**: A "Refined FC" matrix heavily optimized for ID accuracy.

### C. Validation & Statistics
*   **`notebooks/hcp_kaggle.ipynb`**:
    *   Visual inspection (Does the diagonal look bright in the confusion matrix?).
    *   Basic accuracy calculation (optional).
*   **`notebooks/kaggle_brain_fingerprinting.py`**:
    *   **Hyperparameter Tuning**: Grid searches for optimal Atoms ($K$) and Sparsity ($L$).
    *   **Inductive Cross-Validation**: Ensures the dictionary learned on Train data generalizes to Test data (vital for valid claims).
    *   **Statistical Tests**:
        *   **Permutation Test**: $P$-value against random chance.
        *   **McNemar’s Test**: Compares specific hits/misses against the Baseline.
        *   **Bootstrap**: 95% Confidence Intervals for accuracy.

### D. Reproducibility
*   **`notebooks/hcp_kaggle.ipynb`**:
    *   Runs linearly; results vary with data subsets.
*   **`notebooks/kaggle_brain_fingerprinting.py`**:
    *   **Fixed Seeds**: Uses `random_state=42` in classifiers and K-SVD to guarantee identical results across runs.
    *   **Automated Reporting**: Generates a text file summarizing all metrics for copy-pasting into papers.

---

## 3. Workflow Recommendation

1.  **Use `notebooks/hcp_kaggle.ipynb`** if you need to:
    *   Understanding the directory structure of the HCP dataset.
    *   Visualize raw time-series or regions using `nilearn`.
    *   sanity-check that the data downloaded correctly.

2.  **Use `notebooks/kaggle_brain_fingerprinting.py`** if you need to:
    *   **Run the actual experiment.**
    *   Generate results for the manuscript/report.
    *   Compare the Deep Learning approach against standard baselines.
