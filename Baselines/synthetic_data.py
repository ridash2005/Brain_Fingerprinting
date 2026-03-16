import numpy as np
import os

def generate_synthetic_fc(num_subjects, n_parcels=360):
    """Generate synthetic functional connectivity matrices."""
    # Create a base connectivity pattern
    base_pattern = np.random.rand(n_parcels, n_parcels)
    base_pattern = (base_pattern + base_pattern.T) / 2
    
    fc_data = []
    for i in range(num_subjects):
        # Subject-specific variation
        subject_variation = np.random.normal(0, 0.1, (n_parcels, n_parcels))
        subject_variation = (subject_variation + subject_variation.T) / 2
        
        # Add a "fingerprint" - a unique identifying component for each subject
        fingerprint = np.zeros((n_parcels, n_parcels))
        fingerprint[i % n_parcels, :] = 1.0
        fingerprint[:, i % n_parcels] = 1.0
        
        fc = base_pattern + subject_variation + 0.5 * fingerprint
        # Ensure it's a valid correlation-like matrix (bounded -1 to 1)
        fc = np.clip(fc, -1.0, 1.0)
        np.fill_diagonal(fc, 1.0)
        fc_data.append(fc)
        
    return np.array(fc_data)

def generate_synthetic_timeseries(num_subjects, n_parcels=360, length=284):
    """Generate synthetic BOLD timeseries data."""
    ts_data = []
    for i in range(num_subjects):
        # Subject-specific latent signal
        latent = np.random.randn(1, length)
        # Mix latent signal into parcels to create correlations
        weights = np.random.randn(n_parcels, 1)
        ts = weights @ latent + np.random.normal(0, 0.5, (n_parcels, length))
        
        # Add subject-specific "fingerprint" signal
        fingerprint_signal = np.sin(np.linspace(0, 2 * np.pi * (i + 1), length)).reshape(1, -1)
        ts += 0.2 * fingerprint_signal
        
        ts_data.append(ts)
    return np.array(ts_data)

def setup_synthetic_data(fc_data_dir, tasks=["motor"], num_subjects=50):
    """Setup synthetic data files for testing."""
    os.makedirs(fc_data_dir, exist_ok=True)
    
    # Generate rest FC
    fc_rest = generate_synthetic_fc(num_subjects)
    np.save(os.path.join(fc_data_dir, "fc_rest.npy"), fc_rest)
    print(f"[SYNTHETIC] Saved fc_rest.npy to {fc_data_dir}")
    
    # Generate task FCs
    for task in tasks:
        fc_task = generate_synthetic_fc(num_subjects)
        # Make task FC somewhat correlated with rest FC for the same subject
        # but with some task-specific change
        fc_task = 0.7 * fc_rest + 0.3 * fc_task
        np.save(os.path.join(fc_data_dir, f"fc_{task}.npy"), fc_task)
        print(f"[SYNTHETIC] Saved fc_{task}.npy to {fc_data_dir}")
