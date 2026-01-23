# Kaggle Brain Fingerprinting Code Explanation

This document provides a detailed, cell-by-cell explanation of the `kaggle_brain_fingerprinting.py` script. This script implements a complete pipeline for "Brain Fingerprinting" — the process of identifying individuals based on their functional connectivity (FC) patterns derived from fMRI data.

## Overview

The goal of this analysis is to improve the identification of subjects by denoising their connectivity matrices using a **Convolutional Autoencoder (ConvAE)** and **Sparse Dictionary Learning (SDL)**. The script compares this proposed method against standard baselines.

---

## 1. Setup & Requirements

### What it does
This section sets up the Python environment, imports necessary libraries, and defines data paths. It uses `torch` for deep learning, `numpy` for matrix operations, and `sklearn` for some mathematical tools.

### Key Components
-   **Imports**: Libraries like `torch` (for neural networks), `scikit-learn` (for matching pursuit), and `matplotlib`/`seaborn` (for plotting).
-   **Directory Setup**: It automatically detects if it's running on Kaggle or a local machine and sets up `INPUT_DIR` (where data lives) and `OUTPUT_DIR` (where results go).
-   **`setup_environment()`**: This function searches for the dataset. If it finds a zip file, it extracts it. It locates the directories containing "resting state" and "task" fMRI data.
### `generate_fc_for_task()`
This is a crucial helper function. It loads fMRI time-series data (BOLDSignals) for each subject and computes the **Functional Connectivity (FC)** matrix.

**Equation**: The FC matrix is typically the Pearson correlation of time series between brain regions $i$ and $j$:

$$
FC_{ij} = \frac{\text{cov}(ts_i, ts_j)}{\sigma_{ts_i} \sigma_{ts_j}}
$$

where $ts$ is the time series.

---

## 2. Core Model Architectures

### What it does
Defines the deep learning model used to extract stable features from the connectivity matrices.

### The Convolutional Autoencoder (`ConvAutoencoder`)
This is a neural network designed to compress the input data into a lower-dimensional "latent" representation and then reconstruct it. The idea is that the "noise" (individual variability that isn't useful for ID) will be lost, while the stable "fingerprint" features are kept.

-   **Input**: A $360 \times 360$ FC matrix (treated as an image).
-   **Encoder**: A series of Convolutional layers (`Conv2d`) that reduce the size of the image while increasing the number of "channels" (features).
    -   It uses `ReLU` activation functions (linear correction) and `MaxPool2d` to downsample.
-   **Decoder**: The reverse of the encoder. It uses Transposed Convolutions (`ConvTranspose2d`) to upscale the latent features back to the original $360 \times 360$ size.

### Loss Function
The model is trained to minimize the Mean Squared Error (MSE) between the input $X$ and the reconstruction $\hat{X}$:

$$
\mathcal{L} = \frac{1}{N} \sum (X - \hat{X})^2
$$

---

## 3. Sparse Dictionary Learning (K-SVD)

### What it does
After the Autoencoder cleans the data, the script uses **Sparse Dictionary Learning** to further refine the "residuals" (the parts the autoencoder missed) to see if they contain useful specific information, or to simply model the noise structure to remove it.

### Concepts & Equations

### 1. Sparse Coding (via OMP - Orthogonal Matching Pursuit)
We want to represent a signal $y$ as a combination of a few "atoms" from a dictionary $D$.

$$
y \approx D x
$$

subject to $\|x\|_0 \le L$ (where $x$ has at most $L$ non-zero elements).

2.  **K-SVD Algorithm**:
    This calculates the dictionary $D$ that best represents the data. It iterates between:
    -   **Sparse Coding**: Finding the best $x$ for the current $D$.
    -   **Dictionary Update**: Updating $D$ to minimize the error.
    
    The script implements `k_svd` manually using Singular Value Decomposition (SVD) to update atoms one by one.

3.  **`perform_grid_search`**:
    This function tries different values for $K$ (number of atoms) and $L$ (sparsity level) to find which combination gives the best identification accuracy.

---

## 4. Evaluation Metrics

### What it does
Defines how we measure success.

-   **`calculate_accuracy` (Top-1)**: For a given subject's task FC, does it correlate most strongly with *their own* resting FC?

$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{argmax}_j(\text{Corr}(Task_i, Rest_j)) == i)
$$
-   **`calculate_top_k_accuracy`**: Is the correct subject in the top $k$ matches?
-   **`calculate_mrr` (Mean Reciprocal Rank)**: If the correct match is the 1st guess, score is 1. If 2nd, score is 1/2. If 10th, 1/10. This averages those scores.
-   **`differential_identifiability`**: The gap between the self-correlation and the average correlation with others. Higher is better.

$$
I_{diff} = \bar{r}_{self} - \bar{r}_{others}
$$

---

## 5. Statistical Validation

### What it does
Ensures the results are real and not just due to random chance.

-   **`bootstrap_ci`**: Resamples the subjects with replacement 1000 times to calculate a Confidence Interval (e.g., "Accuracy is 95% ± 2%").
-   **`permutation_test`**: Randomly shuffles the labels of subjects to see if the model's accuracy is significantly better than random guessing.
    
    $p$-value calculation:

$$
p = \frac{\text{\# times random shuffle } \ge \text{ observed acc}}{\text{total permutations}}
$$
-   **`mcnemar_test`**: A statistical test specifically for comparing two classifiers (e.g., Baseline vs. Our Model) on the same data.

---

## 6. Ablation Studies

### What it does
Systematically tests which parts of the "Proposed Method" actually help.

It runs:
1.  **Raw FC**: Baseline (just correlation).
2.  **Group Avg Subtraction**: Removing the common average brain pattern.
3.  **SDL only**: Just Dictionary Learning, no Autoencoder.
4.  **ConvAE Latent**: Using the compressed features from the Autoencoder.
5.  **ConvAE Residuals**: Using the "error" part of the Autoencoder.
6.  **ConvAE + SDL (Proposed)**: Removing the shared structure (via Autoencoder) and then refining the remainder with SDL.

---

## 7. State-of-the-Art (SOTA) Comparisons

### What it does
Compares the new method against published methods:
-   **Finn et al. (2015)**: The original connectome fingerprinting conceptualization.
-   **Edge Selection**: Using only the most variable edges (connections) in the brain, filtering out stable ones.

---

## 8. Robustness Analysis

### What it does
Tests if the model breaks easily.
-   **Noise Robustness**: Adds Gaussian noise to the matrices to see how accuracy drops.
-   **Sample Size Robustness**: Runs the analysis with only 20%, 40%, etc. of the subjects to see how much data is needed.

---

## 9. Cross-Validation

### What it does
Splits the subjects into "Folds" (e.g., 5 groups).
1.  Train the Autoencoder on 4 groups.
2.  Test on the 5th group.
3.  Repeat 5 times.

This ensures the model isn't "memorizing" the test subjects.

---

## 10. Visualization Functions

### What it does
Generates the plots for the paper.
-   `plot_ablation_results`: Bar chart of accuracies.
-   `plot_robustness`: Line charts showing accuracy vs. noise/sample size.
-   `plot_full_correlation_matrix`: A large heatmap showing correlations between all Task and Rest subjects (diagonal should be bright).
-   `plot_similarity_distributions`: Histograms showing "Self" correlations vs "Other" correlations.

---

## 11. Generate Manuscript Report & 12. Main Pipeline

### What it does
-   **`generate_manuscript_report`**: Writes a text file summarizing all the numbers suitable for copy-pasting into a paper.
-   **`run_complete_analysis`**: The "Conductor". It calls all the previous functions in order:
    1.  Load Data.
    2.  Train Autoencoder.
    3.  Run Ablations.
    4.  Run SOTA comparisons.
    5.  Run Statistics.
    6.  Run Cross-Validation.
    7.  Generate Plots and Report.

---

## 13. Execute Pipeline

### What it does
The entry point (`if __name__ == "__main__":`).
1.  Defines the tasks to run (e.g., "motor", "working memory").
2.  Checks which data is actually available on the disk.
3.  Runs a **Hyperparameter Optimization** step first (finding the best $K$ and $L$ for K-SVD).
4.  Runs the full pipeline for each available task.
5.  Zips all results for download.
