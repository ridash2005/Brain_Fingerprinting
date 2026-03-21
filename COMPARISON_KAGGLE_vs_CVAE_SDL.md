# Side-by-Side Comparison: kaggle_brain_fingerprinting.py vs cvae_sdl.py

## Overview

| Aspect | kaggle_brain_fingerprinting.py | cvae_sdl.py |
|--------|--------------------------------|------------|
| **Purpose** | Comprehensive IEEE TCDS manuscript analysis (multi-method) | Production CVAE+SDL baseline for architecture comparison |
| **File Size** | 1,911 lines | 611 lines |
| **Scope** | Ablation studies, comparisons, interpretability, robustness | Single focused method (CVAE+SDL only) |
| **Architecture** | ConvAutoencoder (2D convolutions) | ConditionalVAE (fully-connected) |
| **Framework** | Notebook-style with markdown cells | Standalone CLI application |
| **Target Audience** | Manuscript reviewers, comprehensive pipeline | Researchers comparing custom architectures |

---

## Architecture Comparison

### Neural Network Model

#### kaggle_brain_fingerprinting.py: ConvAutoencoder
```python
class ConvAutoencoder(nn.Module):
    - Input: 360×360 FC matrices (2D)
    - Encoder: Conv2d(1,16) → Conv2d(16,32) → Conv2d(32,64) + MaxPool
    - Latent: 64×45×45 = 129,600 dimensions
    - Decoder: ConvTranspose2d (mirror of encoder)
    - Activation: ReLU (encoder), Tanh (decoder)
    - Loss: MSE only (no KL divergence)
```

**Characteristics:**
- ✓ 2D convolutions capture spatial structure
- ✓ Works with matrix format (no vectorization needed)
- ✗ Not a VAE (no stochastic latent space)
- ✗ Not conditional on fMRI state

#### cvae_sdl.py: ConditionalVAE
```python
class ConditionalVAE(nn.Module):
    - Input: 64,620-dimensional vectors (lower-triangular FC)
    - Encoder: Linear(64629, 3000) → Linear(3000, 2000) → ... → Linear(500, 100)
    - Latent: 100 dimensions (mu, logvar)
    - Decoder: Linear(109, 500) → ... → Linear(3000, 64620)
    - Activation: Tanh throughout
    - Conditioning: 9-dimensional one-hot for fMRI state
    - Loss: MSE reconstruction + KL divergence (proper CVAE)
```

**Characteristics:**
- ✓ Fully-connected architecture (matches Lu et al. paper)
- ✓ Proper VAE with reparameterization trick
- ✓ Conditional on 9 HCP states (rest1, rest2, gambling, motor, wm, emotion, language, relational, social)
- ✓ 100-dimensional latent space (paper specification)
- ✓ Tanh activation throughout (paper specification)

---

## Training & Optimization

### Optimizer Configuration

| Aspect | kaggle_brain_fingerprinting.py | cvae_sdl.py |
|--------|--------------------------------|------------|
| **Optimizer** | Adam | SGD (Stochastic Gradient Descent) |
| **Learning Rate** | 0.001 | 1e-4 |
| **LR Schedule** | None (fixed) | ExponentialLR (gamma=0.9991) |
| **Gradient Clipping** | Not mentioned | Yes (max_norm=5.0) |
| **Batch Size** | 16 | 64 |
| **Epochs (default)** | 20 | 50 |

**Paper Requirement**: Lu et al. specifies SGD with decay=0.9991
- ✓ cvae_sdl.py matches paper exactly
- ✗ kaggle version uses Adam (incorrect for paper reproduction)

### Loss Function

**kaggle_brain_fingerprinting.py:**
```python
# ConvAutoencoder
loss = MSE(output, input)  # Reconstruction only
```

**cvae_sdl.py:**
```python
# ConditionalVAE (proper CVAE loss)
recon_loss = MSE(recon_x, x)
kld = -0.5 * mean(1 + logvar - mu² - exp(logvar))
loss = recon_loss + beta * kld
```

**Analysis:**
- Kaggle: Deterministic autoencoder loss
- CVAE_SDL: Proper probabilistic VAE loss with KL regularization
- ✓ cvae_sdl.py implements correct CVAE mathematically

---

## Data Handling

### Input Format

| Aspect | kaggle_brain_fingerprinting.py | cvae_sdl.py |
|--------|--------------------------------|------------|
| **Format** | 360×360 matrices | 64,620-dimensional vectors |
| **Data Layout** | 2D (convolution-friendly) | 1D vectorized (lower-triangular) |
| **Normalization** | [-1, 1] clipped | [-1, 1] clipped |
| **Vectorization** | Not needed | Via lower-triangular indices |

**Code comparison:**
```python
# kaggle_brain_fingerprinting.py
fc = np.corrcoef(timeseries)  # 360×360 matrix
fc_input = fc.reshape(1, 1, 360, 360)  # Conv2d expects (B, C, H, W)

# cvae_sdl.py
fc_lower = fc[np.tril_indices(360, k=-1)]  # Extract lower-triangular: 64,620 elements
fc_input = fc_lower.reshape(1, 64620)  # Fully-connected expects (B, Features)
```

### State Conditioning

**kaggle_brain_fingerprinting.py:**
- No explicit conditioning
- Trained separately on each task
- Binary classification-style approach

**cvae_sdl.py:**
```python
ALL_STATES = ["rest1", "rest2", "gambling", "motor", "wm", 
              "emotion", "language", "relational", "social"]
STATE_IDX = {state: idx for idx, state in enumerate(ALL_STATES)}

# For each sample:
one_hot_condition = np.zeros(9)
one_hot_condition[STATE_IDX[state]] = 1

# During training:
x_cond = concatenate([fc_vector, one_hot_condition])  # 64,620 + 9 = 64,629
```

**Analysis:**
- ✓ cvae_sdl.py properly conditions decoder on 9-state one-hot
- ✗ kaggle version doesn't use state information in architecture

---

## Dictionary Learning (K-SVD + OMP)

### Implementation

**Both implement identical K-SVD algorithms:**

```python
# Identical in both files
def omp_sparse_coding(Y, D, L):
    return orthogonal_mp(D, Y, n_nonzero_coefs=L)

def update_dictionary(Y, D, X):
    for k in range(D.shape[1]):
        idx = np.nonzero(X[k, :])
        U, S, Vt = np.linalg.svd(residual)
        D[:, k] = U[:, 0]
        ...

def k_svd(Y, K, L, n_iter=10, random_state=None):
    # Standard K-SVD with initialization and iterative refinement
```

| Aspect | kaggle_brain_fingerprinting.py | cvae_sdl.py |
|--------|--------------------------------|------------|
| **K-SVD Iterations** | 10 | 10 |
| **OMP Implementation** | sklearn's orthogonal_mp | sklearn's orthogonal_mp |
| **Grid Search** | Yes (K, L optimization) | No (fixed defaults) |
| **Default K** | Grid-search optimized | 50 |
| **Default L** | Grid-search optimized | 20 |

---

## Evaluation Protocol

### Train-Test Split Strategy

| Aspect | kaggle_brain_fingerprinting.py | cvae_sdl.py |
|--------|--------------------------------|------------|
| **Validation Method** | K-fold cross-validation | 50-50 train-test split |
| **K (folds)** | 5 | N/A (single split) |
| **Random Seed** | Varies | Fixed seed=42 |
| **Subject Allocation** | Familial-aware grouping | Random permutation |
| **Reproducibility** | Medium (variable across folds) | High (deterministic seed) |

**Code:**
```python
# kaggle_brain_fingerprinting.py
kf = KFold(n_splits=5, shuffle=True, random_state=None)
for train_idx, test_idx in kf.split(subjects):
    # Each fold trains new model

# cvae_sdl.py
indices = np.random.RandomState(42).permutation(n_subjects)
train_idx = indices[:n_subjects//2]
test_idx = indices[n_subjects//2:]
# Single fixed split
```

### Metrics Evaluation

**Both implement similar metrics:**

```python
# Both files implement:
- top_1_accuracy: % of correct self-matches
- top_5_accuracy: % where self-match in top-5
- top_10_accuracy: % where self-match in top-10
- mean_rank: Average rank of self-match
- mrr (Mean Reciprocal Rank): 1/rank averaged
- differential_identifiability: self_corr_mean - other_corr_mean
```

**Difference:**
- **kaggle**: Reports per-fold averages, no per-pair breakdown shown
- **cvae_sdl**: Reports all 72 state-pair combinations explicitly

---

## Comparison Framework

### Grid Search for Hyperparameters

**kaggle_brain_fingerprinting.py:**
```python
def perform_grid_search(Y, rest_flat, K_range=(2,16), L_max=None, n_iter=3):
    """
    Two-stage grid search:
    1. Coarse search: Wide K range, large steps
    2. Fine search: Narrow range around best K
    
    Returns: Heatmap visualization, best_K, best_L
    """
```

**Coverage:**
- K values: 2, 4, 6, 8, 10, 12, 14, 16
- L values: 2 to K (stepping by 2)
- Total combinations: ~50+ tested

**cvae_sdl.py:**
- No grid search implemented
- Fixed K=50, L=20 (from paper defaults)
- CLI accepts `--K` and `--L` arguments for user tuning

**Analysis:**
- ✓ Kaggle: Exhaustive hyperparameter search (publication-quality)
- ✗ Kaggle: Computationally expensive (grid heatmaps, early stopping)
- ✓ CVAE_SDL: Fast baseline with sensible defaults
- ✗ CVAE_SDL: No automated hyperparameter optimization

---

## Ablation Studies

### kaggle_brain_fingerprinting.py (Available)

```python
def run_complete_analysis():
    # 1. Raw FC (no processing)
    # 2. ConvAE only (no K-SVD)
    # 3. SDL only (K-SVD on raw, no ConvAE)
    # 4. ConvAE + SDL (combined pipeline)
```

**Ablation outputs:**
- Accuracy drop without ConvAE: `ΔAcc`
- Accuracy drop without K-SVD: `ΔAcc`
- Synergistic contribution measurement

### cvae_sdl.py (Not Included)

- Focuses on single CVAE+SDL method only
- Designed for baseline comparison, not method ablation
- Could be extended but outside scope

---

## CLI & Usability

### Command-Line Interface

**kaggle_brain_fingerprinting.py:**
```python
# Notebook-style (no direct CLI)
# Markdown cells with parameters at top:
# - N_SUBJECTS = 997
# - N_FOLDS = 5
# - PERFORM_GRID_SEARCH_GLOBAL = True
# - TUNED_PARAMS = {"motor": (12, 8), "rest": (10, 6), ...}
# - USE_SYNTHETIC = False
```

**Usage:** Edit notebook cells, run all

**cvae_sdl.py:**
```bash
python cvae_sdl.py \
  --subjects 339 \
  --epochs 50 \
  --lr 1e-4 \
  --K 50 \
  --L 20
```

**CLI options:**
- `--subjects`: Number of subjects (default 339)
- `--epochs`: Training epochs (default 50)
- `--lr`: Learning rate (default 1e-4)
- `--K`: Dictionary atoms (default 50)
- `--L`: Sparsity level (default 20)

---

## Output & Results

### Results Directory Structure

**kaggle_brain_fingerprinting.py:**
```
manuscript_results/
├── grid_search_{task}_{timestamp}.png    (Heatmap)
├── {task}_analysis/
│   ├── accuracy_summary.txt
│   ├── ablation_comparison.png
│   ├── model_checkpoints/
│   └── train_logs/
├── hcp_fingerprinting_results_{timestamp}.zip
```

**cvae_sdl.py:**
```
cvae_sdl_results/
└── cvae_sdl_{timestamp}/
    ├── results.json              (All metrics)
    ├── training_losses.npy       (Loss curve)
    ├── cvae_model.pt            (Model weights)
    └── dictionary_D.npy         (K-SVD dictionary)
```

**Difference:**
- Kaggle: Multiple formats (visualizations, text, checkpoints)
- CVAE_SDL: Machine-readable JSON + NumPy arrays (comparison-friendly)

---

## Code Quality & Production Readiness

| Aspect | kaggle_brain_fingerprinting.py | cvae_sdl.py |
|--------|--------------------------------|------------|
| **Error Handling** | Moderate (try-except blocks) | Excellent (data validation, graceful failures) |
| **Comments** | Extensive (markdown cells) | Concise but clear section headers |
| **Reproducibility** | Medium (variable seeds, fold-dependent) | High (fixed seed=42 throughout) |
| **GPU Support** | Yes (auto-detect) | Yes (auto-detect) |
| **Synthetic Data** | Yes (fallback mode) | No (requires real data) |
| **Documentation** | Markdown cells in notebook | README_BASELINE.md + QUICKSTART.md |

---

## Feature Comparison Matrix

| Feature | kaggle_bf | cvae_sdl |
|---------|:---------:|:--------:|
| **CVAE Architecture (Paper-correct)** | ✗ | ✓ |
| **Fully-connected layers** | ✗ | ✓ |
| **Conditional on 9 states** | ✗ | ✓ |
| **SGD optimizer (paper spec)** | ✗ | ✓ |
| **Tanh activation (paper spec)** | ✗ | ✓ |
| **K-SVD + OMP** | ✓ | ✓ |
| **Grid search for K,L** | ✓ | ✗ |
| **K-fold cross-validation** | ✓ | ✗ |
| **Ablation studies** | ✓ | ✗ |
| **Interpretability analysis** | ✓ | ✗ |
| **CLI interface** | ✗ | ✓ |
| **JSON results** | ✗ | ✓ |
| **Comparison tooling** | ✗ | ✓ |
| **Fixed reproducible seed** | ✗ | ✓ |

---

## Key Differences Summary

### Architecture
| Aspect | kaggle_bf | cvae_sdl |
|--------|-----------|----------|
| Model | ConvAutoencoder (2D) | ConditionalVAE (FC) |
| Paper compliance | ✗ (Wrong architecture) | ✓ (Exact match) |

### Training
| Aspect | kaggle_bf | cvae_sdl |
|--------|-----------|----------|
| Optimizer | Adam | SGD + decay |
| Paper compliance | ✗ | ✓ |

### Validation
| Aspect | kaggle_bf | cvae_sdl |
|--------|-----------|----------|
| Split strategy | 5-fold CV (K-fold) | 50-50 fixed |
| Reproducibility | Medium | High |

### Analysis Scope
| Aspect | kaggle_bf | cvae_sdl |
|--------|-----------|----------|
| Purpose | Comprehensive manuscript | Single focused baseline |
| Methods | Multiple (ablation, comparisons) | One method (CVAE+SDL) |
| Grid search | Yes | No |

---

## When to Use Each

### kaggle_brain_fingerprinting.py
✓ **Best for:**
- Comprehensive IEEE TCDS manuscript analysis
- Ablation studies (impact of each component)
- Grid search for optimal K, L
- Multiple method comparisons
- Exploratory analysis with synthetic data fallback

✗ **Not recommended for:**
- Reproducible baseline for comparing architectures
- Production deployment
- Fixed protocol comparison

### cvae_sdl.py
✓ **Best for:**
- Production-ready baseline matching Lu et al. paper
- Fair baseline for comparing custom architectures
- Fast, deterministic evaluation
- Integration into comparison pipelines
- Publication-quality baseline results

✗ **Not recommended for:**
- Comprehensive manuscript-level analysis
- Ablation studies
- Hyperparameter optimization
- Exploring multiple methods

---

## Correcting kaggle_brain_fingerprinting.py to Match Paper

To make kaggle_brain_fingerprinting.py match Lu et al. paper specifications, these changes would be needed:

1. **Replace ConvAutoencoder with ConditionalVAE**
   ```python
   # Change from ConvAutoencoder to ConditionalVAE
   # Use fully-connected d-3000-2000-1000-500-100 architecture
   ```

2. **Change optimizer to SGD with decay**
   ```python
   optimizer = optim.SGD(model.parameters(), lr=1e-4)
   scheduler = optim.lr_scheduler.ExponentialLR(gamma=0.9991)
   ```

3. **Add proper CVAE loss with KL divergence**
   ```python
   loss = recon_loss + beta * kld  # Instead of MSE only
   ```

4. **Vectorize FC matrices to 64,620 dimensions**
   ```python
   fc_vector = fc[np.tril_indices(360, k=-1)]  # Lower-triangular
   ```

5. **Add state conditioning (9-state one-hot)**
   ```python
   condition = one_hot(state, n_states=9)
   conditioned_input = concatenate([fc_vector, condition])
   ```

6. **Use 50-50 train-test split instead of K-fold**
   ```python
   train_idx = indices[:n_subjects//2]
   test_idx = indices[n_subjects//2:]
   ```

---

## Conclusion

**cvae_sdl.py** is the correct production baseline because:
1. ✓ Implements exact Lu et al. architecture (fully-connected)
2. ✓ Uses correct optimizer (SGD with decay)
3. ✓ Includes proper state conditioning (9-state one-hot)
4. ✓ Has fixed reproducible seed (seed=42)
5. ✓ Produces deterministic results
6. ✓ Ready for fair architectural comparisons

**kaggle_brain_fingerprinting.py** is better for:
1. ✓ Comprehensive manuscript analysis
2. ✓ Hyperparameter exploration
3. ✓ Ablation studies
4. ✓ Understanding each component's contribution
5. ✓ Multiple method comparisons

**Recommendation**: Use cvae_sdl.py as your production baseline for comparing custom architectures. Use kaggle_brain_fingerprinting.py for exploratory analysis and ablation studies in separate studies.
