"""
Interpretability Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 7: No analysis of what the AE learns
- Reviewer 1, Point 7: Contribution of brain regions to identification
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.conv_ae import ConvAutoencoder
from utils.matrix_ops import reconstruct_symmetric_matrix


class ConvAEFilterAnalysis:
    """Analyze and visualize filters learned by the ConvAutoencoder."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = ConvAutoencoder()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def get_first_layer_filters(self) -> np.ndarray:
        # Access encoder first layer - check if it's a Sequential or indexed
        # Based on typical implementation in this project
        try:
            filters = self.model.encoder[0].weight.data.cpu().numpy()
        except:
            # Fallback for different encoder structures
            first_conv = None
            for module in self.model.encoder.modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break
            filters = first_conv.weight.data.cpu().numpy() if first_conv else np.zeros((1,1,1,1))
        return filters
        
    def plot_filters(self, output_path: str) -> None:
        filters = self.get_first_layer_filters()
        n_filters = filters.shape[0]
        
        rows = int(np.ceil(np.sqrt(n_filters)))
        cols = int(np.ceil(n_filters / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if n_filters == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_filters):
            f = filters[i, 0] if filters.ndim > 2 else filters[i]
            sns.heatmap(f, ax=axes[i], cmap='RdBu_r', center=0, cbar=False)
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
            
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class DictionaryAtomAnalysis:
    """Analyze SDL dictionary atoms and their spatial representation."""
    
    def __init__(self, dictionary: np.ndarray, n_parcels: int):
        self.D = dictionary
        self.n_parcels = n_parcels
        self.n_atoms = dictionary.shape[1]
        
    def map_atom_to_matrix(self, atom_idx: int) -> np.ndarray:
        atom = self.D[:, atom_idx]
        return reconstruct_symmetric_matrix(atom, self.n_parcels)
        
    def compute_parcel_contribution(self) -> np.ndarray:
        parcel_contrib = np.zeros(self.n_parcels)
        for i in range(self.n_atoms):
            matrix = np.abs(self.map_atom_to_matrix(i))
            parcel_contrib += np.sum(matrix, axis=0)
        return parcel_contrib / self.n_atoms


def run_interpretability_pipeline(model_path, fc_path, dictionary, sparse_codes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ConvAE Analysis
    if model_path and os.path.exists(model_path):
        cae_anal = ConvAEFilterAnalysis(model_path)
        cae_anal.plot_filters(os.path.join(output_dir, 'convae_filters.png'))
    
    # 2. Dictionary Analysis
    n_parcels = 360 
    dict_anal = DictionaryAtomAnalysis(dictionary, n_parcels)
    contrib = dict_anal.compute_parcel_contribution()
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(n_parcels), contrib)
    plt.xlabel('Parcel Index')
    plt.ylabel('Contribution Strength')
    plt.title('Brain Region Contribution (Proposed Method)')
    plt.savefig(os.path.join(output_dir, 'parcel_contributions.png'))
    plt.close()
    
    return {'parcel_contributions': contrib}
