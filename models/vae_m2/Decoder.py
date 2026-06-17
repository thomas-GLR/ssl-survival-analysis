import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder: p_theta(x | y, z)

    Input  : concatenation of z (latent_dim) and y_onehot (num_classes)
    Output : reconstructed signal (B, input_dim)

    FC -> ConvTranspose1d x3
      first 2 layers: ReLU
      last layer    : Linear (no activation)
    """

    def __init__(self, input_dim: int = 1024, latent_dim: int = 128,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.input_dim = input_dim
        decoder_in_dim = latent_dim + num_classes

        # We need to map back to (B, 64, 256) to mirror the encoder
        conv_in_len  = input_dim // 4   # 256
        self.flat_dim = 64 * conv_in_len

        self.fc = nn.Linear(decoder_in_dim, self.flat_dim)

        self.deconv_block = nn.Sequential(
            # Transpose Conv 1: (B, 64, 256) -> (B, 32, 512)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Transpose Conv 2: (B, 32, 512) -> (B, 16, 1024)
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Transpose Conv 3: (B, 16, 1024) -> (B, 1, 1024)  -- linear activation
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: Tensor, y_onehot: Tensor) -> Tensor:
        """
        z       : (B, latent_dim)
        y_onehot: (B, num_classes)
        Returns reconstructed signal (B, input_dim)
        """
        zy = torch.cat([z, y_onehot], dim=-1)       # (B, latent_dim + num_classes)
        h  = F.relu(self.fc(zy))                    # (B, flat_dim)
        h  = h.view(h.size(0), 64, -1)              # (B, 64, 256)
        x_hat = self.deconv_block(h)                # (B, 1, input_dim)
        return x_hat.squeeze(1)                     # (B, input_dim)
