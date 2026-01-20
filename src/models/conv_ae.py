import torch
import torch.nn as nn
import numpy as np

# Define the convolutional autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (batch_size, 16, 360, 360)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch_size, 16, 180, 180)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (batch_size, 32, 180, 180)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, 90, 90)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (batch_size, 64, 90, 90)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (batch_size, 64, 45, 45)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (batch_size, 32, 90, 90)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # (batch_size, 16, 180, 180)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # (batch_size, 1, 360, 360)
            nn.Tanh()  # To bring output between -1 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
