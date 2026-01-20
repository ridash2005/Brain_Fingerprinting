import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.conv_ae import ConvAutoencoder
from src.utils.matrix_ops import get_run_timestamp, ensure_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

# Argument parsing for model selection and dataset choice
parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('-model', type=str, required=True, help="Model to train")
parser.add_argument('-data', type=str, choices=['rest', 'wm', 'emotion', 'motor'], required=True, help="Dataset to train on")
args = parser.parse_args()

# Setup timestamped output directory
RUN_ID = get_run_timestamp()
OUTPUT_DIR = ensure_dir(os.path.join("results", "runs", f"{RUN_ID}_train_{args.data}"))

# Load the specified dataset
data_paths = {
    'rest': 'FC_DATA/fc_rest.npy',
    'wm': 'FC_DATA/fc_wm.npy',
    'emotion': 'FC_DATA/fc_emotion.npy',
    'motor': 'FC_DATA/fc_motor.npy'
}

data = np.load(data_paths[args.data])
data = data[:, np.newaxis, :, :]  # Reshape to (339, 1, 360, 360)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

# Create dataset and split into training and validation sets
dataset = TensorDataset(data_tensor, data_tensor)  # Use data as both input and target
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
if args.model == 'conv_ae': 
    model = ConvAutoencoder()

model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
train_losses = []
val_losses = []
best_val_loss = float('inf')
os.makedirs('./src/models/trained', exist_ok=True)
os.makedirs('./results', exist_ok=True)
best_model_path = f'./src/models/trained/{args.model}_{args.data}_best_model.pth'

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, _ = val_data
            val_inputs = val_inputs.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_inputs)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        # Also save a copy in the run directory for reproducibility
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_model.pth"))
        print(f'Saved best model with validation loss: {best_val_loss:.4f}')

print('Training complete')

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title(f'Training and Validation Losses ({args.data})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))  
plt.legend()
plt.grid()

# Save the figure
plt.savefig(os.path.join(OUTPUT_DIR, f"training_validation_loss.png"))
plt.close()
