import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.config_parser import *
from src.utils.matrix_ops import * 

# Basic parameters
basic_parameters = parse_basic_params()
RUN_ID = get_run_timestamp()
OUTPUT_DIR = ensure_dir(os.path.join("results", "runs", f"{RUN_ID}_baseline"))

HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']

fc_rest = np.load('FC_DATA/fc_rest.npy')

# List of tasks and corresponding file names
tasks = {
    'motor': 'FC_DATA/fc_motor.npy',
    'wm': 'FC_DATA/fc_wm.npy',
    'emotion': 'FC_DATA/fc_emotion.npy'
}

# Open the result file once and append results
result_file = os.path.join(OUTPUT_DIR, 'baseline_accuracy_results.txt')
with open(result_file, 'w') as file:
    for task, path in tasks.items():
        # Load task FC data
        fc_task = np.load(path)

        # Flatten and correlate
        fc_task_flat = fc_task.reshape(N_SUBJECTS, -1)
        fc_rest_flat = fc_rest.reshape(N_SUBJECTS, -1)
        corr_task_rest = np.corrcoef(fc_task_flat, fc_rest_flat, rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]

        # Calculate and write accuracy
        accuracy = calculate_accuracy(corr_task_rest)
        file.write(f'{task.capitalize()} vs Rest - Accuracy: {accuracy:.4f}\n')

print(f"Baseline accuracy results saved to {result_file}")
