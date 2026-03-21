"""
Dataset Description Module for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 1: Lack of dataset details (subjects, sessions, task descriptions)
"""

import os

def generate_dataset_documentation(output_path: str):
    """
    Generate a text file describing the dataset used for the manuscript.
    In a real scenario, this would extract stats from the data files.
    """
    with open(output_path, 'w') as f:
        f.write("DATASET DESCRIPTION: HUMAN CONNECTOME PROJECT (HCP)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. SUBJECTS\n")
        f.write("   - Total Subjects: 100 individuals from the HCP 1200 Subjects Release.\n")
        f.write("   - Demographic: Healthy adults (ages 22-35).\n")
        f.write("   - Selection: Random subset used for demonstrating the pipeline.\n\n")
        
        f.write("2. SCANNING PROTOCOL (RESTING STATE)\n")
        f.write("   - Session: REST1_LR\n")
        f.write("   - Repetition Time (TR): 0.72s\n")
        f.write("   - Duration: ~15 minutes (1200 frames)\n\n")
        
        f.write("3. TASK PROTOCOL (MOTOR TASK)\n")
        f.write("   - Task: Motor task (finger tapping, toe squeezing, etc.)\n")
        f.write("   - Session: TF_MOTOR_LR\n")
        f.write("   - Duration: ~3.5 minutes (284 frames)\n\n")
        
        f.write("4. PREPROCESSING & PARCELLATION\n")
        f.write("   - Pipeline: HCP Minimal Preprocessing Pipeline.\n")
        f.write("   - Parcellation: Glasser et al. (2016) Multi-modal Parcellation (MMP 1.0).\n")
        f.write("   - Number of Regions: 360 (180 per hemisphere).\n")
        f.write("   - Connectivity Metric: Pearson Correlation (Fisher Z-transformed).\n")

if __name__ == "__main__":
    generate_dataset_documentation("dataset_metadata.txt")
