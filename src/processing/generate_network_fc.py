import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from src.utils.hcp_io import *
from src.utils.config_parser import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load timeseries for a specific task.')
parser.add_argument('-task', type=str, required=True, help='Name of the task (e.g., "rest", "motor")')
parser.add_argument('-network', type=str, required=True, help='Name of the network for functional connectivity')
args = parser.parse_args()

# Load basic parameters
basic_parameters = parse_basic_params()
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
subjects = range(N_SUBJECTS)

# Load the regions data
regions_path = os.path.join(HCP_DIR, 'hcp_rest', 'regions.npy')
if not os.path.exists(regions_path):
    # Try alternate path if first one fails
    regions_path = os.path.join(HCP_DIR, 'hcp_task', 'regions.npy')

regions = np.load(regions_path, allow_pickle=True)

# Access the data for each region
region_names = regions[:, 0].tolist()  # Extract the first column (names)
network_types = regions[:, 1].tolist()  # Extract the second column (network types)
myelin_values = regions[:, 2].astype(float)  # Convert the third column (myelin) to float

# Create a dictionary to hold the region information
region_info = {
    'name': region_names,
    'network': network_types,
    'myelin': myelin_values
}

# Check if the specified network is valid
network_names = np.unique(region_info["network"])
if args.network not in network_names:
    raise ValueError(f"Invalid network name. Available networks: {network_names}")

# Prepare to load fMRI data
fmri_data = []
for subject in subjects:
    # Set directory based on the task name
    task_dir = os.path.join(HCP_DIR, "hcp_rest" if args.task.lower() == "rest" else "hcp_task")
    ts_concat = load_timeseries(subject, name=args.task, dir=task_dir)
    fmri_data.append(ts_concat)

print(f"Loaded fMRI data for '{args.task}'")

# Identify indices for the specified network
network_indices = [i for i, net in enumerate(region_info['network']) if net == args.network]

# Initialize FC matrix
n_regions = len(network_indices)
fc = np.zeros((N_SUBJECTS, n_regions, n_regions))

for sub, ts in enumerate(fmri_data):
    # Select regions for the specified network
    ts_network = ts[network_indices, :]
    # Compute the correlation matrix
    fc[sub] = np.corrcoef(ts_network)

print(f"Generated Functional Connectomes for the '{args.network}' network of shape = {fc.shape}")

# Save the functional connectivity array
os.makedirs('./FC_DATA', exist_ok=True)
fc_filename = os.path.join('./FC_DATA', f'fc_{args.task}_{args.network}.npy')
np.save(fc_filename, fc)
print(f"Saved functional connectivity matrix to '{fc_filename}'")
