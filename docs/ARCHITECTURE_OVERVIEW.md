# Brain Fingerprinting: Architecture Overview

This document provides a comprehensive overview of the **Brain Fingerprinting** project. This repository implements a pipeline for individual identification (fingerprinting) from fMRI Functional Connectivity (FC) matrices, utilizing Convolutional Autoencoders (ConvAE) and Sparse Dictionary Learning (SDL) to refine signals and improve accuracy.

---

## 📂 Project Structure Overview

The project is organized into core processing scripts, data management utilities, model definitions, analysis modules, and result logging.

### 1. Analysis & Statistical Validation (`src/analysis`)
Created to address reviewer comments and provide rigorous validation for manuscript preparation.

*   **`src/analysis/statistical_validation.py`**: Implements permutation tests, bootstrap confidence intervals, and multiple comparison corrections (Bonferroni, Holm, FDR).
*   **`src/analysis/ablation_studies.py`**: Systematic comparison of various pipeline configurations (Raw FC, ConvAE only, SDL only, Full Pipeline).
*   **`src/analysis/state_of_art_comparison.py`**: Comprehensive benchmarks against Finn et al. (2015), edge selection, and PCA-based methods.
*   **`src/analysis/interpretability.py`**: Tools for visualizing what the models learn (ConvAE filters, latent space distributions, and dictionary atoms).
*   **`src/analysis/robustness_analysis.py`**: Evaluates how performance degrades with added noise, reduced sample sizes, or missing data.
*   **`src/analysis/evaluation_metrics.py`**: Beyond top-1 accuracy: implements Top-k, Mean Reciprocal Rank (MRR), and similarity distributions.
*   **`src/analysis/dataset_description.py`**: Generates comprehensive statistics and documentation for the HCP dataset used.
*   **`src/analysis/run_complete_analysis.py`**: Master script that runs all the above analyses and generates a final experimental report.

### 2. Core Processing & Refinement Pipelines (`src/processing`)
These files represent the core components for denoising and refining FC matrices.

*   **`src/processing/refine_whole_brain.py`**: Primary refinement script for full 360x360 matrices.
*   **`src/processing/generate_whole_brain_fc.py`**: Computes full-brain Functional Connectivity (FC) matrices from timeseries.
*   **`src/processing/optimize_hyperparameters.py`**: Hyperparameter grid search for dictionary size ($K$) and sparsity ($L$).

### 3. Model Architectures (`src/models`)

*   **`src/models/conv_ae.py`**: Defines the Convolutional Autoencoder. Updated with BatchNorm, weight initialization, and architecture justification for manuscript.
*   **`src/models/sparse_dictionary_learning.py`**: Implements the **K-SVD algorithm**. Updated with theoretical justification and hyperparameter analysis.
*   **`src/models/baseline_correlation.py`**: Standard **Finn et al. (2015)** identification baseline.

### 4. Training & Data Acquisition

*   **`src/train_model.py`**: Training engine for Autoencoders with validation-based early stopping.
*   **`src/data/download_hcp_data.py`**: Automated downloader for the HCP dataset components.

### 5. Utilities & Config (`src/utils`)

*   **`src/utils/config_parser.py`**: Parser for `config/basic_parameters.txt`.
*   **`src/utils/matrix_ops.py`**: Mathematical utilities like `reconstruct_symmetric_matrix` and `calculate_accuracy`.

### 6. Visualizations & Demo

*   **`src/visualization/plot_optimization_heatmap.py`**: Generates heatmaps from optimization logs.
*   **`src/demo_pipeline.py`**: A standalone script using synthetic data to demonstrate the entire pipeline in seconds.

### 7. Notebooks (`notebooks`)

*   **`notebooks/kaggle_brain_fingerprinting.py`**: A self-contained script optimized for **Kaggle**. Generates a full manuscript-ready report (including all analysis modules) in a single run.

---

## ⏱ Typical Workflow Summary

1.  **Installation**: `pip install -r requirements.txt`
2.  **Configuration**: Set paths in `config/basic_parameters.txt`.
3.  **Data Fetch**: Run `src/data/download_hcp_data.py`.
4.  **FC Generation**: Run `src/processing/generate_whole_brain_fc.py -task rest`.
5.  **Model Training**: Run `src/train_model.py -model conv_ae -data rest`.
6.  **Comprehensive Analysis**: Run `src/analysis/run_complete_analysis.py --task motor --output_dir results/final_report`.
7.  **Results**: Check `results/final_report/ANALYSIS_SUMMARY.txt` for the final figures and statistics for the manuscript.
