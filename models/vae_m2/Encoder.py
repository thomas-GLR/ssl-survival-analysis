from typing import Tuple

import torch.nn as nn
from torch import Tensor

class Encoder(nn.Module):
    """
    Encoder: q_phi(z | x)  ->  (mu_z, log_var_z)

    2 x Conv1d (ReLU + BN + Dropout) -> FC -> mu, log_var
    Input shape : (B, 1, input_dim)
    Output      : mu (B, latent_dim), log_var (B, latent_dim)
    """

    def __init__(self, input_dim: int = 1024, latent_dim: int = 128,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            # Conv layer 1
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 32, 512)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Conv layer 2
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 256)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Flattened size after two stride-2 convolutions over input_dim
        conv_out_len = input_dim // 4          # 1024 -> 256
        flat_dim = 64 * conv_out_len           # 64 * 256 = 16 384

        self.fc_mu      = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x : (B, input_dim)  raw vibration segment
        Returns mu, log_var each of shape (B, latent_dim)
        """
        h = self.conv_block(x.unsqueeze(1))   # (B, 64, 256)
        h = h.flatten(1)                       # (B, 16384)
        return self.fc_mu(h), self.fc_log_var(h)

