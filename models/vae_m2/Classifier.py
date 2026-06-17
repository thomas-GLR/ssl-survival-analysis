import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class Classifier(nn.Module):
    """
    Classifier: q_phi(y | x)  ->  class probabilities

    2 x (Conv1d + MaxPool1d + Dropout + ReLU) -> FC -> Softmax
    Input shape : (B, 1, input_dim)
    Output      : (B, num_classes)  -- probabilities
    """

    def __init__(self, input_dim: int = 1024, num_classes: int = 10,
                 dropout: float = 0.25) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 1024)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                            # (B, 32,  512)
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1), # (B, 64,  512)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                            # (B, 64,  256)
            nn.Dropout(dropout),
        )
        conv_out_len = input_dim // 4   # 1024 -> 256
        flat_dim = 64 * conv_out_len    # 16384

        self.fc = nn.Linear(flat_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, input_dim)
        Returns class probabilities (B, num_classes)
        """
        h = self.conv_block(x.unsqueeze(1))   # (B, 64, 256)
        h = h.flatten(1)                       # (B, 16384)
        return F.softmax(self.fc(h), dim=-1)   # (B, num_classes)
