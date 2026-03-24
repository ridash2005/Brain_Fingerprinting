# Comprehensive Pipeline Explanation: `notebooks/kaggle_brain_fingerprinting.py`

This document provides a mathematically rigorous and logically exhaustive explanation of the Brain Fingerprinting pipeline implemented in `notebooks/kaggle_brain_fingerprinting.py`. This script represents the final, production-ready analysis designed for the IEEE Transactions on Cognitive and Developmental Systems (TCDS) manuscript.

---

## 1. Mathematical Foundation of Functional Connectivity (FC)

The pipeline begins with the transformation of raw fMRI BOLD (Blood-Oxygen-Level-Dependent) time series into Functional Connectivity matrices.

### 1.1 The Pearson Correlation Coefficient
For two brain regions (parcels) $i$ and $j$ with time series $ts_i$ and $ts_j$, the connectivity $FC_{ij}$ is defined as:

$$
FC_{ij} = \frac{\sum_{t=1}^T (ts_i(t) - \bar{ts}_i)(ts_j(t) - \bar{ts}_j)}{\sqrt{\sum_{t=1}^T (ts_i(t) - \bar{ts}_i)^2 \sum_{t=1}^T (ts_j(t) - \bar{ts}_j)^2}}
$$

Where:
- $T$ is the number of time points.
- $\bar{ts}$ is the mean of the time series.
- $FC \in \mathbb{R}^{N \times N}$ is a symmetric $360 \times 360$ matrix ($N=360$ Glasser parcels).

### 1.2 Data Preprocessing
The script implements:
- **Demeaning**: $ts_{normalized} = ts - \mu_{ts}$
- **Ordering**: Parcels are ordered according to the Glasser Atlas, preserving **Functional Topography** (network blocks like Visual, DMN, etc.).
- **Dimensions**: $\mathbf{X}_{subject} \in [-1, 1]^{360 \times 360}$

---

## 2. Phase 1: The "Double-Denoising" Strategy

The core innovation is a two-stage denoising process: **ConvAE** for global structure and **SDL** for structured noise.

### 2.1 Stage 1: Convolutional Autoencoder (ConvAE)
The ConvAE acts as a non-linear filter to remove the **Shared Anatomical Blueprint** (the "Mean Brain").

#### Architecture & Dimensional Flow:
1.  **Input ($X$)**: Shape $[1, 360, 360]$ (C=1, H=360, W=360)
2.  **Encoder ($E$)**:
    - $Layer 1$: $Conv2d(1, 16, 3 \times 3) \to MaxPool \to$ Shape $[16, 180, 180]$
    - $Layer 2$: $Conv2d(16, 32, 3 \times 3) \to MaxPool \to$ Shape $[32, 90, 90]$
    - $Layer 3$: $Conv2d(32, 64, 3 \times 3) \to MaxPool \to$ Shape $[64, 45, 45]$
    - **Latent Bottleneck ($z$)**: Shape $[64, 45, 45]$ (129,600 dimensions)
3.  **Decoder ($D_{ae}$)**:
    - Transposed Convolutions upscale features back to $[1, 360, 360]$.

#### Objective Function:
The model minimizes the Mean Squared Error (MSE):

$$
\min_{\theta_E, \theta_D} \frac{1}{M} \sum_{m=1}^M \|\mathbf{X}_m - D_{ae}(E(\mathbf{X}_m))\|_F^2
$$

#### Logical Justification (The Residual Hypothesis):
The ConvAE learns the reconstruction $\hat{\mathbf{X}} = D_{ae}(E(\mathbf{X}))$. We define the **Residual** $\mathbf{R} = \mathbf{X} - \hat{\mathbf{X}}$. 
- **Dimensions**: $\mathbf{R} \in \mathbb{R}^{360 \times 360}$.
- **Logic**: By removing the dominant shared signal, we increase the **Signal-to-Noise Ratio (SNR)** of the unique subject-specific deviations.

### 2.2 Stage 2: Sparse Dictionary Learning (SDL)
The residuals $\mathbf{R}$ still contain structured noise (scanner artifacts). SDL is applied to the **Lower Triangular (tril)** elements to remove these patterns.

#### Vectorization using Tril Indexing
Because the FC matrix is symmetric, the upper triangle is redundant. We extract the lower triangle (excluding diagonal) to create a feature vector $\mathbf{y}$:
- **Number of unique edges**: $N_{tri} = \frac{N(N-1)}{2} = \frac{360 \times 359}{2} = 64,620$
- **Extraction**: $\mathbf{y} = vec(\text{tril}(\mathbf{R}, -1))$
- **Dimensions**: $\mathbf{y} \in \mathbb{R}^{64,620}$

#### The K-SVD Objective:
We represent the vectorized subject residual $\mathbf{y}_m$ as a sparse combination of dictionary atoms:

$$
\mathbf{y}_m \approx \mathbf{D}\mathbf{x}_m \quad \text{subject to} \quad \|\mathbf{x}_m\|_0 \le L
$$

Where:
- **Dictionary ($\mathbf{D}$)**: $\mathbb{R}^{64,620 \times K}$ (contains $K$ common noise patterns).
- **Sparse Codes ($\mathbf{x}_m$)**: $\mathbb{R}^{K \times 1}$ (contains $L$ non-zero coefficients).

#### Final Pure Fingerprint ($\mathbf{F}$):
The final fingerprint is the high-fidelity residual after removing both common structure and common noise:

$$
\mathbf{F}_m = \mathbf{y}_m - \mathbf{D}\mathbf{x}_m
$$

- **Dimensions**: $\mathbf{F}_m \in \mathbb{R}^{64,620}$
- **Logic**: $\mathbf{D}\mathbf{x}_m$ captures *structured* residual artifacts. Subtracting them isolates the *idiosyncratic* biological signal.

---

## 3. Phase 2: Hyperparameter Optimization (Grid Search)

We empirically find optimal $(K, L)$ by scanning a grid and maximizing Top-1 Identification Accuracy.

### 3.1 Coarse Search
- **Grid**: $K \in \{4, 8, 12 \dots 32\}$, $L \in \{2 \dots K\}$.
- **Iterations**: Fast K-SVD (2 iterations).

### 3.2 Fine Search
- **Grid**: Narrow window around coarse best (e.g., $K^* \pm 3$).
- **Iterations**: Accurate K-SVD (5 iterations).

---

## 4. Phase 3: Evaluation Metrics

Dimensions of the Correlation Matrix: $N_{subjects} \times N_{subjects}$ (e.g., $339 \times 339$).

1.  **Top-1 Accuracy**:
    
    $$
    Acc = \frac{1}{M} \sum_{i=1}^M \mathbb{I}\left(\text{argmax}_j (\text{Corr}(\mathbf{F}_{task,i}, \mathbf{F}_{rest,j})) = i\right)
    $$

2.  **Mean Reciprocal Rank (MRR)**:
    
    $$
    MRR = \frac{1}{M} \sum_{i=1}^M \frac{1}{Rank_i}
    $$

3.  **Differential Identifiability ($I_{diff}$)**:
    
    $$
    I_{diff} = \frac{1}{M} \sum \text{Corr}_{self} - \frac{1}{M(M-1)} \sum \text{Corr}_{others}
    $$

---

## 5. Phase 4: Statistical Validation

-   **Bootstrap CIs**: Resampling subjects $B=1000$ times.
-   **Permutation Test**: Shuffling identities to build a chance null distribution.
    
    $$
    p = \frac{\sum_{b=1}^B \mathbb{I}(Acc_{null} \ge Acc_{obs}) + 1}{B + 1}
    $$

---

## 6. Phase 5: Awareness Compliance & Cross-Validation

To prevent **Data Leakage**, the pipeline processes subjects **Inductively**:
1.  **Fold $n$ Training**: Learn weights $\theta$ and Dictionary $\mathbf{D}$ from 80% of subjects.
2.  **Fold $n$ Testing**: Use the **SAME** $\theta$ and $\mathbf{D}$ to process the held-out 20%.
3.  **Benefit**: Ensures the dictionary doesn't "memorize" the identity of test subjects.

---

## Summary of Logic: Why it works
1.  **ConvAE** removes the "Mean Face" of the brain ($360 \times 360$).
2.  **Vectorization** collapses the symmetric matrix into $64,620$ unique connections.
3.  **SDL** removes "Structured Scanner Noise" using a learned dictionary $\mathbf{D} \in \mathbb{R}^{64,620 \times K}$.
4.  **Result**: A clean, high-dimensional vector $\mathbf{F} \in \mathbb{R}^{64,620}$ that contains the purest possible individual fingerprint.
