from torch import nn


class RULMLP(nn.Module):
    """Four-layer RUL regressor over flattened (batch, time, features) sequences."""

    def __init__(self, num_features: int, sequence_len: int):
        super().__init__()
        self.n_features = num_features
        self.sequence_length = sequence_len
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_len * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, sequence_length, n_features]")
        if x.shape[1:] != (self.sequence_length, self.n_features):
            raise ValueError("Input sequence length or feature count is incorrect")
        return self.regressor(x)
