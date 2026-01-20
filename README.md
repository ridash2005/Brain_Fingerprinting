# Brain Fingerprinting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository implements a robust pipeline for **Brain Fingerprinting** using Functional Connectivity (FC) matrices derived from HCP fMRI data. It features a novel refinement mechanism combining **Convolutional Autoencoders** and **Sparse Dictionary Learning (SDL)** to denoise task-based signals and improve subject identification accuracy.

## 🚀 Project Overview

The core objective is to extract "refined" functional connectomes that are more consistent across different mental states (rest vs tasks) than the raw signals. Our pipeline consists of:
1. **Denoising**: Using a trained Autoencoder to remove common task-related components.
2. **Refinement**: Applying Sparse Dictionary Learning (K-SVD) to the residuals to capture the individual's unique fingerprint.

---

## 📂 Repository Structure

```text
├── config/              # Configuration parameters (HCP paths, constants)
├── docs/                # Detailed documentation and images
├── logs/                # Execution logs and hyperparameter search results
├── notebooks/           # Research and exploration notebooks
├── results/             # Output directory for plots and accuracy metrics
├── src/                 # Core Source Code
│   ├── data/            # Data acquisition and preprocessing
│   ├── models/          # Neural Network architectures and SDL
│   ├── processing/      # Full-brain and network-specific pipelines
│   ├── utils/           # Helper functions for matrix ops and parsing
│   └── visualization/   # Plotting scripts
├── tests/               # Script verification and unit tests
└── requirements.txt     # Project dependencies
```

---

## 🛠️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Brain_Fingerprinting.git
   cd Brain_Fingerprinting
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Paths**:
   Update `config/basic_parameters.txt` with your local HCP data directory path (absolute path recommended).

---

## 📖 Usage Instructions

### 1. Data Acquisition
To download and extract the HCP dataset (339 subjects, Glasser parcellation):
```bash
python src/data/download_hcp_data.py
```
*Use `--dry-run` to check connectivity without downloading.*

### 2. Generate Functional Connectivity (FC)
Generate whole-brain FC matrices from timeseries:
```bash
python src/processing/generate_whole_brain_fc.py -task motor
```

### 3. Model Training
Train the Convolutional Autoencoder on resting-state data:
```bash
python src/train_model.py -model conv_ae -data rest
```

### 4. Refinement & Identification
Run the refinement pipeline to calculate identification accuracy:
```bash
# Using trained Autoencoder
python -W ignore src/processing/refine_whole_brain.py -task motor

# Using Group-Average Baseline
python src/processing/refine_whole_brain_avg.py -task motor
```

### 6. Kaggle & Notebooks
A self-contained, high-performance notebook script is available for Kaggle users:
1.  Open `notebooks/kaggle_brain_fingerprinting.py`.
2.  In VS Code, use **"Export to Jupyter Notebook"** or simply copy-paste cells into a Kaggle kernel.
3.  Set `use_synthetic=False` to run on real data after uploading your matrices.
Search for optimal K (atoms) and L (sparsity) for SDL:
```bash
python src/processing/optimize_hyperparameters.py -data rest -task motor
```
Then visualize the results:
```bash
python src/visualization/plot_optimization_heatmap.py -log logs/accuracy_log_rest_to_motor.txt
```

---

## 📊 Results & Logging

To ensure reproducibility and easy comparison between experiments, all outputs are organized by **Run ID** (timestamped):
- **Visualizations & Metrics**: Stored in `results/runs/YYYYMMDD_HHMMSS_{experiment_name}/`
- **Optimization Logs**: Stored in `logs/runs/YYYYMMDD_HHMMSS_{experiment_name}/`
- **Models**: Every training run saves a backup of the best model in its specific run directory, while maintaining the latest production model in `src/models/trained/`.

Our pipeline significantly improves subject identification accuracy over baseline methods (Finn et al., 2015).

## 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
