import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoderFP(nn.Module):
    def __init__(self):
        super(ConvAutoencoderFP, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (batch_size, 16, 50, 50)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch_size, 16, 25, 25)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (batch_size, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, 12, 12)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (batch_size, 64, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # (batch_size, 64, 6, 6)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [16, 32, 12, 12]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # [16, 16, 24, 24]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # [16, 1, 48, 48]
            nn.ReLU(),
            # Additional layer to get to 50x50
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1),  # [16, 1, 50, 50]
            nn.Sigmoid()
        )

    def forward(self, x):
        target_size = (x.size(2), x.size(3))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = F.interpolate(decoded, size=target_size, mode='bilinear', align_corners=True)
        return decoded

    def get_encoded_features(self, x):
        if x.size(0) != 16:
            x = x.repeat(16, 1, 1, 1)
        return self.encoder(x)
