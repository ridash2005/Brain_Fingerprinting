import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.conv_ae import ConvAutoencoder
from src.models.conv_ae_fp import ConvAutoencoderFP
from src.models.sparse_dictionary_learning import *
from src.utils.config_parser import *
from src.utils.matrix_ops import * 

# Argument parsing for task and network selection
parser = argparse.ArgumentParser(description='Functional Connectome Task and Network Selection')
parser.add_argument('-task', type=str, required=True, choices=['motor', 'wm', 'emotion'], help="Task to analyze: motor, wm, or emotion")
parser.add_argument('-network', type=str, required=True, help="Brain network/region to analyze (e.g., Frontoparietal, DMN, etc.)")
args = parser.parse_args()

# Basic parameters
basic_parameters = parse_basic_params()
RUN_ID = get_run_timestamp()
OUTPUT_DIR = ensure_dir(os.path.join("results", "runs", f"{RUN_ID}_refine_network_{args.task}_{args.network}"))

HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']

# Load the trained model for the specified network
model_path = f'./src/models/trained/conv_ae_fp_{args.network}/best_model.pth'
model = ConvAutoencoderFP()
state_dict = torch.load(model_path)
# Handle both full checkpoints and pure state dicts
if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'])
else:
    model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Select the appropriate functional connectome (FC) data based on the task
fc_task_path = f'FC_DATA/fc_{args.task}_{args.network}.npy'
fc_task = np.load(fc_task_path)

# Extract the network-specific FC data (assuming the network reduces the number of parcels)
regions = np.load(os.path.join(HCP_DIR, 'hcp_rest', 'regions.npy'), allow_pickle=True)

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
network_indices = [i for i, net in enumerate(region_info['network']) if net == args.network]

fc_task = fc_task[:, np.newaxis, :, :]  # Reshape to (N_SUBJECTS, 1, N_PARCELS, N_PARCELS)
fc_task_tensor = torch.tensor(fc_task, dtype=torch.float32)

# Perform reconstruction
with torch.no_grad():
    fc_task_reconstr = model(fc_task_tensor)

# Calculate the residuals
fc_task_residual = fc_task_tensor.squeeze(1) - fc_task_reconstr.squeeze(1)

# Apply SDL (Sparse Dictionary Learning)
Y = np.zeros((int(len(network_indices) * (len(network_indices) - 1) / 2), N_SUBJECTS))
for i in range(N_SUBJECTS):
    Y[:, i] = fc_task_residual[i][np.tril_indices(fc_task_residual[i].shape[0], k=-1)]

D, X = k_svd(Y, 15, 12)
sdl_retr = np.dot(D, X).transpose()
DX = np.zeros((N_SUBJECTS, len(network_indices), len(network_indices)))

for i in range(N_SUBJECTS):
    DX[i, :, :] = reconstruct_symmetric_matrix(sdl_retr[i, :], len(network_indices))

fc_task_refined = fc_task_residual - DX

# Plot the first sample and its reconstruction
first_sample = fc_task[0, 0, :, :]
reconstructed_sample = fc_task_reconstr[0, 0, :, :].numpy()
residual_sample = fc_task_residual[0, :, :].numpy()
refined_sample = fc_task_refined[0, :, :].numpy()

plt.figure(figsize=(16, 6))

plt.subplot(1, 4, 1)
plt.imshow(first_sample, cmap='viridis')
plt.title(f'Original {args.task.capitalize()} Sample')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(reconstructed_sample, cmap='viridis')
plt.title(f'Reconstructed {args.task.capitalize()} Sample')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(residual_sample, cmap='viridis')
plt.title(f'Residual {args.task.capitalize()} Sample')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(refined_sample, cmap='viridis')
plt.title(f'Refined {args.task.capitalize()} Sample')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{args.task}_{args.network}_fcs.png"))
plt.close()

# Load rest data for comparison
fc_rest = np.load(f'FC_DATA/fc_rest_{args.network}.npy')
fc_rest_tensor = torch.tensor(fc_rest, dtype=torch.float32)

# Correlation analysis between task and rest FC before and after processing
corr_task_rest_before_convae = np.corrcoef(fc_task_tensor.view(N_SUBJECTS, -1).numpy(), fc_rest_tensor.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]
corr_task_rest_after_convae = np.corrcoef(fc_task_residual.view(N_SUBJECTS, -1).numpy(), fc_rest_tensor.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]
corr_task_rest_after_sdl = np.corrcoef(fc_task_refined.view(N_SUBJECTS, -1).numpy(), fc_rest_tensor.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]

plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.heatmap(corr_task_rest_before_convae, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (Before ConvAE)")

plt.subplot(1, 3, 2)
sns.heatmap(corr_task_rest_after_convae, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After ConvAE)")

plt.subplot(1, 3, 3)
sns.heatmap(corr_task_rest_after_sdl, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After ConvAE+SDL)")

plt.savefig(os.path.join(OUTPUT_DIR, f"{args.task}_{args.network}_corr.png"))
plt.close()

# Display accuracy results and save them to a text file
accuracy_before = calculate_accuracy(corr_task_rest_before_convae)
accuracy_after_convae = calculate_accuracy(corr_task_rest_after_convae)
accuracy_after_sdl = calculate_accuracy(corr_task_rest_after_sdl)

# Print the results
print(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy before processing: {accuracy_before}')
print(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy after ConvAutoEncoder: {accuracy_after_convae}')
print(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy after ConvAutoEncoder+SDL: {accuracy_after_sdl}')

with open(os.path.join(OUTPUT_DIR, f'accuracy_results.txt'), 'w') as file:
    file.write(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy before processing: {accuracy_before}\n')
    file.write(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy after ConvAutoEncoder: {accuracy_after_convae}\n')
    file.write(f'{args.task.capitalize()} vs Rest ({args.network}) - Accuracy after ConvAutoEncoder+SDL: {accuracy_after_sdl}\n')

print(f'Accuracy results saved in "{OUTPUT_DIR}".')
