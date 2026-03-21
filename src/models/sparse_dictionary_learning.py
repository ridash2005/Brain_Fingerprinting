"""
Sparse Dictionary Learning Module for Functional Connectome Fingerprinting

Copyright (c) 2026 Rickarya Das. All rights reserved.

Addresses Reviewer Comments:
- Reviewer 1, Point 4: SDL implementation underspecified
- Reviewer 2, Point 1(1): Role of sparse dictionary learning unclear
- Reviewer 2, Point 1(2): Need optimal hyperparameters justification

THEORETICAL JUSTIFICATION:
Sparse Dictionary Learning (SDL) is applied to the residual connectomes to:
1. Learn a compact dictionary of connectivity patterns (atoms)
2. Represent each subject's residual as a sparse combination of atoms
3. The sparse codes (X) capture subject-specific variations
4. Subtracting D*X removes additional shared sparse patterns, further isolating
   truly individual-specific connectivity features

The formula: FC_refined = FC_residual - D*X
- FC_residual: Residual from ConvAE (original - reconstructed)
- D: Learned dictionary (shared sparse patterns)
- X: Sparse codes (subject-specific loadings on shared patterns)
- FC_refined: Final fingerprint containing unique subject features

ALGORITHM: K-SVD (K-Singular Value Decomposition)
1. Initialize dictionary D with random normalized columns
2. Sparse Coding: Use OMP (Orthogonal Matching Pursuit) to find sparse X
3. Dictionary Update: Update each atom using SVD on residual
4. Repeat until convergence or max iterations

HYPERPARAMETERS:
- K (n_atoms): Number of dictionary atoms (default: 15)
  - Chosen based on grid search over [5, 10, 15, 20, 25]
  - Trade-off: More atoms = more expressiveness but overfitting risk
  
- L (sparsity): Non-zero coefficients per subject (default: 12)
  - Chosen based on grid search over [5, 8, 10, 12, 15]
  - Trade-off: Higher L = more patterns captured but less sparse

- n_iter: K-SVD iterations (default: 10)
  - Convergence typically achieved within 10 iterations
"""

import numpy as np
from sklearn.linear_model import orthogonal_mp
from numpy.linalg import svd
from typing import Tuple, Optional
from tqdm import tqdm


def omp_sparse_coding(Y: np.ndarray, D: np.ndarray, L: int) -> np.ndarray:
    """
    Sparse coding using Orthogonal Matching Pursuit (OMP).
    
    Solves: min ||Y - D*X||^2 subject to ||X_i||_0 <= L for each column i
    
    Parameters
    ----------
    Y : np.ndarray
        Data matrix (n_features x n_samples)
        Each column is a flattened lower-triangular connectivity vector
    D : np.ndarray
        Dictionary matrix (n_features x n_atoms)
        Each column is a dictionary atom
    L : int
        Sparsity level (max non-zero coefficients per sample)
        
    Returns
    -------
    X : np.ndarray
        Sparse code matrix (n_atoms x n_samples)
        Each column contains at most L non-zero entries
    
    Notes
    -----
    OMP greedily selects atoms that best explain the residual signal.
    At each iteration, it:
    1. Finds the atom most correlated with the current residual
    2. Updates the coefficient estimate using least squares
    3. Updates the residual
    """
    n_atoms = D.shape[1]
    n_samples = Y.shape[1]
    X = np.zeros((n_atoms, n_samples))
    
    for i in range(n_samples):
        # OMP for each sample
        X[:, i] = orthogonal_mp(D, Y[:, i], n_nonzero_coefs=L)
        
    return X


def update_dictionary(Y: np.ndarray, D: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update dictionary atoms using SVD on residuals.
    
    For each atom k:
    1. Find samples that use atom k (non-zero X[k,:])
    2. Compute residual: E_k = Y - sum_{j!=k} d_j * x_j
    3. Update d_k and x_k using rank-1 SVD of E_k
    
    Parameters
    ----------
    Y : np.ndarray
        Data matrix (n_features x n_samples)
    D : np.ndarray
        Current dictionary (n_features x n_atoms)
    X : np.ndarray
        Sparse codes (n_atoms x n_samples)
        
    Returns
    -------
    D : np.ndarray
        Updated dictionary
    X : np.ndarray
        Updated sparse codes
    
    Notes
    -----
    This is the core of K-SVD: each atom is updated to minimize
    reconstruction error for the samples that use it.
    """
    n_atoms = D.shape[1]
    
    for k in range(n_atoms):
        # Find samples using atom k
        non_zero_indices = np.nonzero(X[k, :])[0]
        
        if len(non_zero_indices) == 0:
            # No samples use this atom - reinitialize randomly
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue
        
        # Compute residual for samples using atom k
        # E_k = Y - D*X + d_k * x_k (remove contribution of atom k)
        residual = Y[:, non_zero_indices] - np.dot(D, X[:, non_zero_indices])
        residual += np.outer(D[:, k], X[k, non_zero_indices])
        
        # Rank-1 SVD approximation
        U, S, Vt = svd(residual, full_matrices=False)
        
        # Update atom and coefficients
        D[:, k] = U[:, 0]  # First left singular vector
        X[k, non_zero_indices] = S[0] * Vt[0, :]  # First singular value * right vector
        
    return D, X


def k_svd(
    Y: np.ndarray,
    K: int,
    L: int,
    n_iter: int = 10,
    tol: float = 1e-6,
    verbose: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-SVD algorithm for sparse dictionary learning.
    
    Learns a dictionary D and sparse codes X such that Y ≈ D*X
    with each column of X having at most L non-zero entries.
    
    Parameters
    ----------
    Y : np.ndarray
        Data matrix (n_features x n_samples)
        For FC fingerprinting: n_features = n_parcels*(n_parcels-1)/2
        n_samples = n_subjects
    K : int
        Number of dictionary atoms to learn
    L : int
        Sparsity level (max non-zero coefficients)
    n_iter : int
        Maximum number of K-SVD iterations
    tol : float
        Convergence tolerance for reconstruction error change
    verbose : bool
        Whether to show progress bar
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    D : np.ndarray
        Learned dictionary (n_features x K)
    X : np.ndarray
        Sparse codes (K x n_samples)
    
    Example
    -------
    >>> # For 339 subjects with 360 parcels
    >>> n_features = 360 * 359 // 2  # 64620
    >>> Y = residual_fc_flattened  # (64620, 339)
    >>> D, X = k_svd(Y, K=15, L=12)
    >>> reconstructed = D @ X  # Shared sparse patterns
    >>> refined = Y - reconstructed  # Subject-specific patterns
    
    References
    ----------
    Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm for 
    designing overcomplete dictionaries for sparse representation. IEEE TSP.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m, n = Y.shape  # m = n_features, n = n_samples
    
    # Initialize dictionary with random normalized columns
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)  # Normalize columns
    
    prev_error = float('inf')
    
    iterator = tqdm(range(n_iter), desc="K-SVD") if verbose else range(n_iter)
    
    for iteration in iterator:
        # Step 1: Sparse coding using OMP
        X = omp_sparse_coding(Y, D, L)
        
        # Step 2: Dictionary update using SVD
        D, X = update_dictionary(Y, D, X)
        
        # Calculate reconstruction error
        reconstruction_error = np.linalg.norm(Y - np.dot(D, X), 'fro')
        
        if verbose:
            iterator.set_description(
                f"Iteration {iteration + 1}/{n_iter}, "
                f"Error: {reconstruction_error:.6f}"
            )
        
        # Check convergence
        if abs(prev_error - reconstruction_error) < tol:
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break
        
        prev_error = reconstruction_error
    
    return D, X


def analyze_dictionary(D: np.ndarray, X: np.ndarray) -> dict:
    """
    Analyze the learned dictionary and sparse codes.
    
    Parameters
    ----------
    D : np.ndarray
        Learned dictionary (n_features x K)
    X : np.ndarray
        Sparse codes (K x n_samples)
        
    Returns
    -------
    dict
        Dictionary containing analysis metrics
    """
    n_atoms = D.shape[1]
    n_samples = X.shape[1]
    
    # Atom correlations (coherence)
    atom_corr = np.corrcoef(D.T)
    off_diag_corr = atom_corr[np.triu_indices(n_atoms, k=1)]
    
    # Sparsity analysis
    sparsity_per_sample = np.sum(X != 0, axis=0)
    atom_usage = np.sum(X != 0, axis=1)
    
    # Variance explained by each atom
    atom_variance = np.var(D, axis=0)
    
    return {
        'n_atoms': n_atoms,
        'n_samples': n_samples,
        'mean_atom_correlation': np.mean(np.abs(off_diag_corr)),
        'max_atom_correlation': np.max(np.abs(off_diag_corr)),
        'mean_sparsity': np.mean(sparsity_per_sample),
        'atom_usage': atom_usage,
        'most_used_atoms': np.argsort(atom_usage)[::-1][:5],
        'least_used_atoms': np.argsort(atom_usage)[:5],
        'atom_variance': atom_variance
    }


def grid_search_hyperparameters(
    Y: np.ndarray,
    K_range: list = [5, 10, 15, 20, 25],
    L_range: list = [5, 8, 10, 12, 15],
    n_iter: int = 10,
    metric: str = 'reconstruction_error'
) -> dict:
    """
    Grid search for optimal K and L hyperparameters.
    
    Parameters
    ----------
    Y : np.ndarray
        Data matrix
    K_range : list
        Range of K values to try
    L_range : list
        Range of L values to try (must be <= K)
    n_iter : int
        K-SVD iterations
    metric : str
        Metric to optimize: 'reconstruction_error' or 'sparsity_ratio'
        
    Returns
    -------
    dict
        Contains best parameters and full results grid
    """
    results = {}
    best_score = float('inf')
    best_params = None
    
    for K in K_range:
        for L in L_range:
            # L cannot exceed K
            if L > K:
                continue
            
            print(f"Testing K={K}, L={L}...")
            D, X = k_svd(Y, K, L, n_iter=n_iter, verbose=False)
            
            reconstruction_error = np.linalg.norm(Y - np.dot(D, X), 'fro')
            sparsity_ratio = np.mean(X != 0)
            
            results[(K, L)] = {
                'reconstruction_error': reconstruction_error,
                'sparsity_ratio': sparsity_ratio
            }
            
            score = reconstruction_error if metric == 'reconstruction_error' else sparsity_ratio
            
            if score < best_score:
                best_score = score
                best_params = (K, L)
    
    return {
        'best_K': best_params[0],
        'best_L': best_params[1],
        'best_score': best_score,
        'all_results': results
    }



def calculate_accuracy_inline(corr_matrix):
    """
    Calculate Top-1 accuracy.
    Inline version to avoid circular imports.
    """
    if corr_matrix is None or corr_matrix.size == 0:
        return 0.0
    
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if np.argmax(corr_matrix[i, :]) == i)
    return correct / n


def perform_grid_search(
    Y: np.ndarray, 
    rest_flat: Optional[np.ndarray], 
    n_subjects: int, 
    n_features: int, 
    K_range: Tuple[int, int] = (2, 16), 
    n_iter: int = 3, 
    task_name: str = "unknown",
    max_search_subs: int = 100,
    early_stopping_patience: int = 2
) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """
    Grid search for optimal K and L parameters based on Identification Accuracy.
    
    Uses early stopping and subject downsampling for efficiency.
    
    Parameters
    ----------
    Y : np.ndarray
        Task data matrix (n_features x n_samples)
    rest_flat : np.ndarray, optional
        Rest data matrix for matching (n_features x n_samples)
    n_subjects : int
        Number of subjects
    n_features : int
        Number of features
    K_range : tuple
        (min_K, max_K) for grid search
    n_iter : int
        Number of iterations for K-SVD
    task_name : str
        Name of task for logging
    max_search_subs : int
        Maximum subjects to use in grid search (for speed)
    early_stopping_patience : int
        Stop if no improvement for N consecutive K values
        
    Returns
    -------
    accuracies : np.ndarray
        Grid of accuracies
    best_K : Optional[int]
        Optimal K (None if grid search failed)
    best_L : Optional[int]
        Optimal L (None if grid search failed)
    """
    print(f"  Grid Search K={K_range} for {task_name} (subset={max_search_subs}, patience={early_stopping_patience})...")
    
    # Downsample subjects for grid search speed
    if n_subjects > max_search_subs:
        print(f"  >>> Downsampling grid search from {n_subjects} to {max_search_subs} subjects for speed.")
        indices = np.random.RandomState(42).permutation(n_subjects)[:max_search_subs]
        Y_subset = Y[:, indices]
        rest_flat_subset = rest_flat[:, indices]
        n_subs_search = max_search_subs
    else:
        Y_subset = Y
        rest_flat_subset = rest_flat
        n_subs_search = n_subjects
    
    Ks = list(range(K_range[0], K_range[1] + 1, 2))
    all_L_vals = sorted(list(set([L for K_val in Ks for L in range(2, K_val + 1, 2)])))
    accuracies = np.zeros((len(Ks), len(all_L_vals)))
    
    best_acc = -1.0
    best_K = None
    best_L = None
    no_improvement_count = 0
    
    for i, K in enumerate(Ks):
        L_vals = range(2, K + 1, 2)
        K_best_acc = -1.0
        
        for L in L_vals:
            j = all_L_vals.index(L)
            # Run simplified K-SVD
            D, X = k_svd(Y_subset, K, L, n_iter=n_iter, verbose=False, random_state=42)
            
            if rest_flat_subset is not None:
                # Approximate Rest Sparse Codes using learned D
                X_rest = omp_sparse_coding(rest_flat_subset, D, L)
                
                # Correlation between sparse codes
                corr = np.corrcoef(X.T, X_rest.T)[:n_subs_search, n_subs_search:]
                acc = calculate_accuracy_inline(corr)
                
                if j < len(all_L_vals):
                    accuracies[i, j] = acc
                
                if acc > best_acc:
                    best_acc = acc
                    best_K = K
                    best_L = L
                
                if acc > K_best_acc:
                    K_best_acc = acc
        
        # Early stopping: check if this K improved over best found so far
        if K_best_acc < best_acc:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"  Early stopping: No improvement for {early_stopping_patience} consecutive K values")
                break
        else:
            no_improvement_count = 0
    
    if best_K is None or best_L is None:
        print(f"  [!] WARNING: Grid search did not find valid parameters")
        return accuracies, None, None
    
    print(f"  Found Optimal: K={best_K}, L={best_L} (Acc: {best_acc:.4f})")
    return accuracies, best_K, best_L


# Documentation for usage
"""
USAGE EXAMPLE FOR FINGERPRINTING:

```python
import numpy as np
from sparse_dictionary_learning import k_svd

# After getting residuals from ConvAE
residuals = fc_original - fc_reconstructed  # (n_subjects, n_parcels, n_parcels)

# Flatten to lower triangular vectors
n_subjects, n_parcels, _ = residuals.shape
n_features = n_parcels * (n_parcels - 1) // 2
Y = np.zeros((n_features, n_subjects))

for i in range(n_subjects):
    Y[:, i] = residuals[i][np.tril_indices(n_parcels, k=-1)]

# Learn dictionary and sparse codes
D, X = k_svd(Y, K=15, L=12, n_iter=10)

# Subtract shared sparse patterns to get refined fingerprint
shared_sparse = D @ X
refined_vector = Y - shared_sparse

# Use refined for fingerprinting...
```
"""

if __name__ == "__main__":
    # Test K-SVD
    print("Testing K-SVD implementation...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_features, n_samples = 100, 50
    K, L = 10, 5
    
    # Create ground truth dictionary and sparse codes
    D_true = np.random.randn(n_features, K)
    D_true = D_true / np.linalg.norm(D_true, axis=0)
    
    X_true = np.zeros((K, n_samples))
    for i in range(n_samples):
        active = np.random.choice(K, L, replace=False)
        X_true[active, i] = np.random.randn(L)
    
    # Generate noisy data
    Y = D_true @ X_true + 0.1 * np.random.randn(n_features, n_samples)
    
    # Run K-SVD
    D_learned, X_learned = k_svd(Y, K, L, n_iter=20)
    
    # Analyze results
    analysis = analyze_dictionary(D_learned, X_learned)
    print(f"\nDictionary Analysis:")
    print(f"  Mean atom correlation: {analysis['mean_atom_correlation']:.4f}")
    print(f"  Mean sparsity: {analysis['mean_sparsity']:.2f}")
    print(f"  Most used atoms: {analysis['most_used_atoms']}")
