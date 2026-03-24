# Comprehensive Pipeline Explanation: `notebooks/kaggle_brain_fingerprinting.py`

This document provides a mathematically rigorous and logically exhaustive explanation of the Brain Fingerprinting pipeline implemented in `notebooks/kaggle_brain_fingerprinting.py`. This script represents the final, production-ready analysis designed for the IEEE Transactions on Cognitive and Developmental Systems (TCDS) manuscript.

---

## 1. Mathematical Foundation of Functional Connectivity (FC)

The pipeline transforms raw fMRI BOLD (Blood-Oxygen-Level-Dependent) time series into Functional Connectivity matrices.

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
- **Demeaning**: $ts_{\text{normalized}} = ts - \mu_{ts}$
- **Ordering**: Parcels are ordered according to the Glasser Atlas to preserve **Functional Topography**.
- **State 1 (Raw FC)**: $\mathbf{X}_{\text{subject}} \in [-1, 1]^{360 \times 360}$

---

## 2. Phase 1: The "Double-Denoising" Strategy

The core innovation is a two-stage denoising process: **ConvAE** for global structure and **SDL** for structured noise.

### 2.1 Stage 1: Convolutional Autoencoder (ConvAE)
The ConvAE acts as a non-linear filter to remove the **Shared Anatomical Blueprint** (the "Mean Brain").

#### Architecture & Dimensional Flow:
1.  **Input ($\mathbf{X}$)**: Shape $[1, 360, 360]$
2.  **Encoder ($\text{E}$)**:
    - Layer 1: $Conv2d(1, 16, 3 \times 3) \to MaxPool \to$ Shape $[16, 180, 180]$
    - Layer 2: $Conv2d(16, 32, 3 \times 3) \to MaxPool \to$ Shape $[32, 90, 90]$
    - Layer 3: $Conv2d(32, 64, 3 \times 3) \to MaxPool \to$ Shape $[64, 45, 45]$
    - **Latent Bottleneck ($\mathbf{z}$)**: Shape $[64, 45, 45]$ (129,600 dimensions)
3.  **Decoder ($\text{D}_{\text{ae}}$)**:
    - Upscales features back to **State 2 (Reconstruction)**: $\hat{\mathbf{X}} \in \mathbb{R}^{360 \times 360}$.

#### Objective Function:

$$
\min_{\theta_E, \theta_D} \frac{1}{M} \sum_{m=1}^M \|\mathbf{X}_m - \hat{\mathbf{X}}_m\|_F^2
$$

### 2.2 Stage 2: Data Vectorization (The "Tril" Extraction)
This is a critical intermediate state. Because the reconstructed matrix $\hat{\mathbf{X}}$ and the input $\mathbf{X}$ are symmetric, we calculate the **Residual Matrix** $\mathbf{R} = \mathbf{X} - \hat{\mathbf{X}}$ and then vectorize only the unique elements.

#### Why the Lower Triangular Matrix?
- **Redundancy**: The upper triangle is a mirror of the lower.
- **Diagonal**: The diagonal of a correlation matrix is always 1, providing zero subject-specific information.
- **Transformation**: We extract the strictly lower triangular elements to form a subject-specific feature vector.

#### Vectorization Formula:
- **Number of unique elements**: $N_{\text{tri}} = \frac{N(N-1)}{2} = \frac{360 \times 359}{2} = 64,620$
- **Extraction**: $\mathbf{y} = \text{vec}(\text{tril}(\mathbf{R}, -1))$
- **State 3 (Vectorized Residual)**: $\mathbf{y} \in \mathbb{R}^{64,620}$

### 2.3 Stage 3: Sparse Dictionary Learning (SDL)
The vectorized residuals $\mathbf{y}$ are cleaned of "Structured Noise" using K-SVD.

#### K-SVD Objective:

$$
\mathbf{y}_m \approx \mathbf{D}\mathbf{x}_m \quad \text{subject to} \quad \|\mathbf{x}_m\|_0 \le L
$$

Where:
- **Dictionary ($\mathbf{D}$)**: $\mathbf{D} \in \mathbb{R}^{64,620 \times K}$ (contains $K$ common noise atoms).
- **Sparse Codes ($\mathbf{x}_m$)**: $\mathbf{x}_m \in \mathbb{R}^{K \times 1}$ (contains $L$ non-zero coefficients).

#### State 4 (The Pure Fingerprint):
The final fingerprint $\mathbf{F}$ is the residual of the residual:

$$
\mathbf{F}_m = \mathbf{y}_m - \mathbf{D}\mathbf{x}_m
$$

- **Dimensions**: $\mathbf{F}_m \in \mathbb{R}^{64,620}$
- **Logic**: Isolates the *idiosyncratic* biological signal from systemic noise.

### 2.4 Final Identification Step (The "Match")
The pure fingerprints $\{ \mathbf{F} \}$ are the ultimate output of the denoising pipeline. We use them to perform subject identification by matching a "Query" (Task condition) against a "Database" (Rest condition).

#### The Matching Algorithm:
For each subject $i$ in the Task group, we calculate the Pearson Correlation between their denoised fingerprint $\mathbf{F}_{\text{task},i}$ and every subject $j$ in the denoised Rest database:

$$
\text{Matching Score}_{i,j} = \text{Corr}(\mathbf{F}_{\text{task},i}, \mathbf{F}_{\text{rest},j})
$$

#### Prediction Rule:
The predicted identity for subject $i$ is the subject $j$ in the database that yields the maximum correlation:

$$
\text{Predicted ID}_i = \text{argmax}_{j} (\text{Matching Score}_{i,j})
$$

#### Why this works:
Because $\mathbf{F}$ has been stripped of the "Mean Brain" (via ConvAE) and "Scanner Noise" (via SDL), the remaining signal is dominated by the subject's unique functional architecture. This results in:
1.  **High Self-Correlation**: $\text{Corr}(\mathbf{F}_{\text{task},i}, \mathbf{F}_{\text{rest},i}) \approx 1$
2.  **Low Other-Correlation**: $\text{Corr}(\mathbf{F}_{\text{task},i}, \mathbf{F}_{\text{rest},j}) \approx 0$ (for $j \neq i$)

This separation ensures that the diagonal of the $M \times M$ identification matrix is maximally bright, yielding high **Top-1 Accuracy**.

---

## 3. Phase 2: Hyperparameter Optimization (Grid Search)

We scan $(K, L)$ to maximize Top-1 Accuracy.
- **Coarse**: $K \in \{4, 8, \dots, 32\}$, $L \in \{2, 4, \dots, K\}$.
- **Fine**: Steps of 1 around coarse winner.

---

## 4. Phase 3: Evaluation Metrics

Performance is measured on the $M \times M$ identification matrix.

1.  **Top-1 Accuracy**:
    
$$
\text{Accuracy} = \frac{1}{M} \sum_{i=1}^{M} \text{I } [ \text{argmax}_{j} (\text{Corr}(\mathbf{F}_{\text{task},i}, \mathbf{F}_{\text{rest},j})) = i ]
$$

2.  **Mean Reciprocal Rank (MRR)**:
    
$$
\text{MRR} = \frac{1}{M} \sum_{i=1}^{M} \frac{1}{\text{Rank}_{i}}
$$

3.  **Differential Identifiability ($I_{\text{diff}}$)**:
    
$$
I_{\text{diff}} = \frac{1}{M} \sum (\text{Corr}_{\text{self}}) - \frac{1}{M(M-1)} \sum (\text{Corr}_{\text{others}})
$$

---

## 5. Phase 4: Statistical Validation

-   **Bootstrap CIs**: Resampling subjects $B=1000$ times.
-   **Permutation Test**:

$$
p = \frac{\sum_{b=1}^{B} [ \text{Acc}_{\text{null}} \ge \text{Acc}_{\text{obs}} ] + 1}{B + 1}
$$

---

## Summary of Dimensional Flow

| State | Transformation | Data Represented | Dimensions |
| :--- | :--- | :--- | :--- |
| **State 1** | Raw Input | Functional Connectivity | $360 \times 360$ |
| **State 2** | ConvAE Latent | Shared Structure Bundle | $64 \times 45 \times 45$ |
| **State 3** | **Tril Extraction** | Vectorized Residuals | $64,620 \times 1$ |
| **State 4** | SDL / K-SVD | Sparse Artifact Codes | $K \times 1$ (e.g. $15 \times 1$) |
| **Final** | **Denoised ID** | The Pure Fingerprint | $64,620 \times 1$ |

**Conclusion**: The use of the **Lower Triangular Matrix** is essential for high-dimensional stability, reducing the feature space by ~50% before SDL while maintaining 100% of the unique connectivity information.
