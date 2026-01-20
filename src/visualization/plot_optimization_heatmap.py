import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot optimization heatmap from logs.')
    parser.add_argument('-log', type=str, required=True, help='Path to the log file')
    parser.add_argument('-output', type=str, default='results/optimization_heatmap.png', help='Path to save the output plot')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: Log file '{args.log}' not found.")
        return

    accuracy_matrix = np.zeros((14, 14))  # K=2 to 15, L=2 to 15

    # Read the log file
    with open(args.log, 'r') as file:
        for line in file:
            # Split the line into parts
            parts = [part.strip(' ,=') for part in line.strip().split() if part.strip(' ,=')]
            if len(parts) < 5: continue
            
            try:
                # Extract K, L, and accuracy
                # Expected format: "K= 15, L = 2 85.0"
                k = int(parts[1])
                l = int(parts[3])
                accuracy = float(parts[4])
                
                # Store accuracy in the appropriate location in the matrix
                accuracy_matrix[k - 2, l - 2] = accuracy  # Subtracting 2 to index from 0
            except (ValueError, IndexError):
                continue

    if np.max(accuracy_matrix) == 0:
        print("Warning: No valid accuracy data found in the log file.")
        return

    # Set up the colormap and normalization
    cmap = sns.color_palette("plasma", as_cmap=True)
    non_zero_vals = accuracy_matrix[accuracy_matrix > 0]
    norm = Normalize(vmin=np.min(non_zero_vals), vmax=np.max(non_zero_vals))

    # Plot heatmap of accuracies
    plt.figure(figsize=(12, 10))
    sns.heatmap(accuracy_matrix, annot=True, fmt=".2f", cmap=cmap, norm=norm, 
                xticklabels=range(2, 16), yticklabels=range(2, 16))
    plt.xlabel('L (Sparsity Level)')
    plt.ylabel('K (Number of Atoms)')
    plt.title(f'Accuracy Heatmap - {os.path.basename(args.log)}')
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Heatmap saved to '{args.output}'")

if __name__ == "__main__":
    main()
