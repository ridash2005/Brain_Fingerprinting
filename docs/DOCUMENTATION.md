# Brain Fingerprinting Project Documentation

This document provides a comprehensive overview of the **Brain Fingerprinting** project. This repository implements a pipeline for individual identification (fingerprinting) from fMRI Functional Connectivity (FC) matrices, utilizing Convolutional Autoencoders (ConvAE) and Sparse Dictionary Learning (SDL) to refine signals and improve accuracy.

---

## 📂 Project Structure Overview

The project is organized into core processing scripts, data management utilities, model definitions, and result logging. Below is a detailed breakdown of every file in the repository.

### 1. Core Processing & Refinement Pipelines (`src/processing`)
These files represent the main scientific contribution of the project: denoising and refining FC matrices to enhance individual "fingerprints."

*   **`src/processing/refine_whole_brain.py`**:
    *   **Purpose**: The primary refinement script for full 360x360 matrices.
    *   **Logic**: Loads a trained `ConvAutoencoder`, calculates residuals (Original - Reconstructed), applies `k_svd` to these residuals, and computes a "refined" FC matrix. It evaluates identification accuracy before and after processing.
*   **`src/processing/refine_whole_brain_avg.py`**:
    *   **Purpose**: A baseline variant of the refinement pipeline.
    *   **Logic**: Instead of using an Autoencoder for the first denoising step, it subtracts the **group-average** FC matrix from each individual's FC to isolate individual-specific residuals before applying SDL.
*   **`src/processing/refine_network.py` / `src/processing/refine_network_avg.py`**:
    *   **Purpose**: Specialized versions of the refinement scripts that operate on specific functional networks (e.g., Frontoparietal, DMN) filtered from the full parcellation.
*   **`src/processing/optimize_hyperparameters.py` / `src/processing/optimize_hyperparameters_avg.py`**:
    *   **Purpose**: Hyperparameter grid search for SDL components.
    *   **Logic**: Performs a grid search over dictionary size ($K$) and sparsity ($L$) to find optimal parameters that maximize identification accuracy.
*   **`src/processing/generate_whole_brain_fc.py`**:
    *   **Purpose**: Computes full-brain Functional Connectivity (FC) matrices.
    *   **Logic**: Loads BOLD timeseries, computes Pearson correlation matrices for all subjects, and saves them as `.npy` files.
*   **`src/processing/generate_network_fc.py`**:
    *   **Purpose**: Computes network-specific FC matrices by filtering regions prior to correlation.

---

## 2. Model Architectures (`src/models`)

*   **`src/models/conv_ae.py`**:
    *   **Purpose**: Defines a 2D Convolutional Autoencoder for 360x360 FC matrices. Treat matrices as images to learn common population-level features.
*   **`src/models/conv_ae_fp.py`**:
    *   **Purpose**: A variant for smaller, network-specific matrices (e.g., 50x50), featuring bilinear interpolation for dimension matching.
*   **`src/models/sparse_dictionary_learning.py`**:
    *   **Purpose**: Implements the **K-SVD algorithm**.
    *   **Functions**: `omp_sparse_coding`, `update_dictionary`, and `k_svd`.
*   **`src/models/baseline_correlation.py`**:
    *   **Purpose**: Implements the standard **Finn et al. (2015)** identification method using raw correlations between task and rest states.

---

## 3. Training & Data Acquisition

*   **`src/train_model.py`**:
    *   **Purpose**: Training engine for Autoencoders. Handles data splitting, optimization, and saving the best performing weights based on validation loss.
*   **`src/data/download_hcp_data.py`**:
    *   **Purpose**: Automated downloader for the HCP dataset components (Rest, Task, Atlas) from remote mirrors into the local project structure.

---

## 4. Utilities & Config (`src/utils`)

*   **`src/utils/config_parser.py`**:
    *   **Purpose**: Robust parser for `config/basic_parameters.txt`.
*   **`src/utils/matrix_ops.py`**:
    *   **Purpose**: Mathematical utilities like `reconstruct_symmetric_matrix` and `calculate_accuracy`.
*   **`src/utils/hcp_io.py`**:
    *   **Purpose**: Low-level I/O for HCP timeseries and EV (Event) files.

---

## 5. Visualizations & Demo

*   **`src/visualization/plot_optimization_heatmap.py`**:
    *   **Purpose**: Generates heatmaps from optimization log files to identify the best (K, L) pairs.
*   **`src/demo_pipeline.py`**:
    *   **Purpose**: A standalone script using synthetic data to demonstrate the entire pipeline in seconds. Ideal for testing if the environment is set up correctly.

---

## 6. Notebooks (`notebooks`)

*   **`notebooks/kaggle_brain_fingerprinting.py`**:
    *   **Purpose**: A self-contained, high-performance script optimized for **Kaggle** and Jupyter environments.
    *   **Features**: Includes synthetic data simulation, data downloaders, and all core model/SDL logic in a single file for easy portability.

---

## ⏱ Typical Workflow Summary

1.  **Installation**: `pip install -r requirements.txt`
2.  **Configuration**: Set paths in `config/basic_parameters.txt`.
3.  **Data Fetch**: Run `src/data/download_hcp_data.py`.
4.  **FC Generation**: Run `src/processing/generate_whole_brain_fc.py -task rest`.
5.  **Model Training**: Run `src/train_model.py -model conv_ae -data rest`.
6.  **Refinement**: Run `src/processing/refine_whole_brain.py -task motor` to see the performance improvement.
