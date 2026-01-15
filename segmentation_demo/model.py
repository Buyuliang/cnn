# model.py
import torch
import torch.nn as nn

class CNN2DSegmentation(nn.Module):
    """
    简单 2D CNN，用于二值图像分割
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(32, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x
