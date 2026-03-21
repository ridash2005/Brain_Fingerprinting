import os
import numpy as np

def generate_mock_hcp(base_dir, n_subjects=5, n_parcels=360, n_tp=100):
    """
    Generates a mock HCP dataset structure for integration testing.
    """
    tasks = ['hcp_rest', 'hcp_task']
    bold_runs = 18 # As defined in hcp_io.py
    
    for task in tasks:
        task_path = os.path.join(base_dir, task)
        os.makedirs(task_path, exist_ok=True)
        
        # Generate subjects
        subjects_path = os.path.join(task_path, "subjects")
        for sub_id in range(n_subjects):
            sub_ts_path = os.path.join(subjects_path, str(sub_id), "timeseries")
            os.makedirs(sub_ts_path, exist_ok=True)
            
            for run_id in range(1, bold_runs + 1):
                # Shape is (n_parcels, n_tp)
                ts = np.random.randn(n_parcels, n_tp).astype(np.float32)
                np.save(os.path.join(sub_ts_path, f"bold{run_id}_Atlas_MSMAll_Glasser360Cortical.npy"), ts)
        
        # Generate regions.npy
        # Shape (n_parcels, 3) -> [name, network, myelin]
        regions = []
        networks = ['Visual', 'Auditory', 'Default', 'Frontoparietal', 'Somatomotor']
        for i in range(n_parcels):
            regions.append([f"Region_{i}", networks[i % len(networks)], np.random.rand()])
        
        np.save(os.path.join(task_path, "regions.npy"), np.array(regions, dtype=object))

    print(f"Mock HCP dataset generated at: {base_dir}")

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "./MOCK_DATA"
    n_subs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    generate_mock_hcp(base_dir, n_subjects=n_subs)
