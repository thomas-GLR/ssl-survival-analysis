from torch import nn


class RULCNN(nn.Module):
    def __init__(self, num_features: int, sequence_len: int):
        super().__init__()

        self.n_features = num_features
        self.sequence_length = sequence_len

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_features,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.sequence_length, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        # Input: [batch_size, sequence_length, n_features]
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [batch, sequence_length, n_features]"
            )
        if x.shape[1:] != (self.sequence_length, self.n_features):
            raise ValueError("Input sequence length or feature count is incorrect")

        # Conv1d expects: [batch_size, n_features, sequence_length]
        x = x.transpose(1, 2)
        x = self.conv(x)

        # Output: [batch_size, 1], matching the other RUL ensemble models.
        return self.regressor(x)
