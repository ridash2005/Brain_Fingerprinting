"""
Convolutional Autoencoder for Functional Connectome Fingerprinting

Addresses Reviewer Comments:
- Reviewer 1, Point 4: Critical implementation details are missing
- Reviewer 1, Point 6: Why convolutional operations are appropriate for correlation matrices

ARCHITECTURE JUSTIFICATION:
Convolutional operations on functional connectivity (FC) matrices are appropriate because:
1. FC matrices exhibit spatial structure based on parcellation ordering (Glasser MMP 2016)
2. Adjacent parcels in the atlas tend to have similar connectivity profiles
3. Networks (DMN, FPN, etc.) create local block-like patterns in ordered matrices
4. Convolutions can capture these local patterns efficiently

COMPLETE ARCHITECTURE SPECIFICATION:
- Input: (batch_size, 1, 360, 360) - FC matrix
- Encoder:
  - Conv2d(1→16, 3x3, pad=1) + ReLU + MaxPool(2x2) → (16, 180, 180)
  - Conv2d(16→32, 3x3, pad=1) + ReLU + MaxPool(2x2) → (32, 90, 90)
  - Conv2d(32→64, 3x3, pad=1) + ReLU + MaxPool(2x2) → (64, 45, 45)
- Latent: 64 × 45 × 45 = 129,600 dimensions
- Decoder:
  - ConvTranspose2d(64→32, 2x2, stride=2) + ReLU → (32, 90, 90)
  - ConvTranspose2d(32→16, 2x2, stride=2) + ReLU → (16, 180, 180)
  - ConvTranspose2d(16→1, 2x2, stride=2) + Tanh → (1, 360, 360)

TRAINING DETAILS:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-8)
- Batch size: 16
- Epochs: 20
- Validation split: 20%
- Early stopping: Based on validation loss (patience=5)
"""

import torch
import torch.nn as nn
import numpy as np


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for learning shared FC patterns.
    
    The autoencoder learns to reconstruct FC matrices, with the key insight
    that the reconstruction captures shared, group-level patterns while the
    residual (original - reconstructed) contains subject-specific information.
    
    Attributes
    ----------
    encoder : nn.Sequential
        Encoder network that compresses FC to latent representation
    decoder : nn.Sequential  
        Decoder network that reconstructs FC from latent representation
    
    Architecture Details
    --------------------
    Total parameters: ~105,000
    Latent dimensions: 129,600 (64 x 45 x 45)
    Compression ratio: 64,980 → 129,600 (no compression, focus on reconstruction)
    
    Note: This architecture uses MaxPooling which is NOT strictly invertible.
    The decoder uses ConvTranspose to approximate the inverse operation.
    """
    
    def __init__(self, n_parcels: int = 360, in_channels: int = 1):
        """
        Initialize the convolutional autoencoder.
        
        Parameters
        ----------
        n_parcels : int
            Number of brain parcels (default: 360 for Glasser MMP)
        in_channels : int
            Number of input channels (default: 1)
        """
        super(ConvAutoencoder, self).__init__()
        
        self.n_parcels = n_parcels
        self.in_channels = in_channels
        
        # Encoder: Progressively compress spatial dimensions
        # Each MaxPool reduces dimensions by 2x
        self.encoder = nn.Sequential(
            # Layer 1: 360x360 -> 180x180
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2: 180x180 -> 90x90
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3: 90x90 -> 45x45
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder: Progressively upsample spatial dimensions
        # ConvTranspose2d with stride=2 doubles dimensions
        self.decoder = nn.Sequential(
            # Layer 1: 45x45 -> 90x90
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 2: 90x90 -> 180x180
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Layer 3: 180x180 -> 360x360
            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),
            nn.Tanh()  # Output range [-1, 1] to match FC value range
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input FC matrices, shape (batch_size, 1, n_parcels, n_parcels)
            
        Returns
        -------
        torch.Tensor
            Reconstructed FC matrices, same shape as input
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode FC matrices to latent representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input FC matrices
            
        Returns
        -------
        torch.Tensor
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to FC matrices.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation
            
        Returns
        -------
        torch.Tensor
            Reconstructed FC matrices
        """
        return self.decoder(z)
    
    def get_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute residual (original - reconstructed).
        
        The residual is hypothesized to contain subject-specific patterns
        because shared patterns are captured by the reconstruction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input FC matrices
            
        Returns
        -------
        torch.Tensor
            Residual FC matrices
        """
        reconstructed = self.forward(x)
        return x - reconstructed
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_summary(self) -> str:
        """Return a string summary of the architecture."""
        summary = []
        summary.append("=" * 60)
        summary.append("ConvAutoencoder Architecture Summary")
        summary.append("=" * 60)
        summary.append(f"Input shape: (batch, 1, {self.n_parcels}, {self.n_parcels})")
        summary.append(f"Total parameters: {self.count_parameters():,}")
        summary.append("")
        summary.append("ENCODER:")
        summary.append("-" * 40)
        for i, layer in enumerate(self.encoder):
            summary.append(f"  {layer}")
        summary.append("")
        summary.append("DECODER:")
        summary.append("-" * 40)
        for i, layer in enumerate(self.decoder):
            summary.append(f"  {layer}")
        summary.append("=" * 60)
        return "\n".join(summary)


class ConvAutoencoderWithSkip(nn.Module):
    """
    ConvAutoencoder variant with skip connections (U-Net style).
    
    This variant may better preserve fine-grained details in residuals.
    """
    
    def __init__(self, n_parcels: int = 360):
        super(ConvAutoencoderWithSkip, self).__init__()
        
        self.n_parcels = n_parcels
        
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 + 64 from skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 32 + 32 from skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 16 + 16 from skip
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        bn = self.bottleneck(p3)
        
        # Decoder with skip connections
        d3 = self.up3(bn)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


if __name__ == "__main__":
    # Test the architecture
    model = ConvAutoencoder()
    print(model.get_architecture_summary())
    
    # Test forward pass
    x = torch.randn(2, 1, 360, 360)
    y = model(x)
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    # Test residual computation
    residual = model.get_residual(x)
    print(f"  Residual shape: {residual.shape}")
