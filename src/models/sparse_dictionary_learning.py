import numpy as np
from sklearn.linear_model import orthogonal_mp
from numpy.linalg import svd
from tqdm import tqdm 

# K-SVD Algorithm

def omp_sparse_coding(Y, D, L):
    """ Solves for X using Orthogonal Matching Pursuit (OMP) algorithm """
    n = Y.shape[1]
    X = np.zeros((D.shape[1], n))
    
    for i in range(n):
        # Solve for each subject (Y[:,i]) and get the sparse representation
        X[:, i] = orthogonal_mp(D, Y[:, i], n_nonzero_coefs=L)
        
    return X

def update_dictionary(Y, D, X):
    """ Updates the dictionary D using SVD on the residual """
    for k in range(D.shape[1]):
        # Find the indices where X has non-zero entries for the current atom k
        non_zero_indices = np.nonzero(X[k, :])[0]
        
        if len(non_zero_indices) == 0:
            continue
        
        # Compute the residual without using the k-th atom
        residual = Y[:, non_zero_indices] - np.dot(D, X[:, non_zero_indices])
        residual += np.outer(D[:, k], X[k, non_zero_indices])
        
        # Apply SVD to the residual to find new dk and xk
        U, S, Vt = svd(residual, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, non_zero_indices] = S[0] * Vt[0, :]
        
    return D, X

def k_svd(Y, K, L, n_iter=10):
    """ K-SVD algorithm for sparse dictionary learning """
    m, n = Y.shape
    
    # Initialize the dictionary with random values
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0)  # Normalize columns of D
    
    with tqdm(total=n_iter) as pbar:
        for iteration in range(n_iter):
            # Step 1: Sparse coding (OMP)
            X = omp_sparse_coding(Y, D, L)

            # Step 2: Dictionary update (SVD)
            D, X = update_dictionary(Y, D, X)

            # Calculate the reconstruction error
            reconstruction_error = np.linalg.norm(Y - np.dot(D, X), 'fro')

            # Update the progress bar with reconstruction error
            pbar.set_description(f"Iteration {iteration + 1}/{n_iter}, Reconstruction Error: {reconstruction_error:.6f}")
            pbar.update(1)
    
    return D, X

# D: Learned dictionary (m x K)
# X: Sparse representation matrix (K x n)
