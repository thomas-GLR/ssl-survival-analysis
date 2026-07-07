import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, num_features, output_dim=1):
        super(CNN1D, self).__init__()

        # Conv1D layers expect input shape: (batch_size, in_channels, sequence_length)
        # in_channels = number of sensors (features)
        # sequence_length = sliding window size

        self.conv_block = nn.Sequential(
            # First Conv Layer: extracts basic low-level features
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Second Conv Layer: captures mid-level temporal combinations
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Third Conv Layer: high-level abstractions
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Global Average Pooling flattens the sequence dimension safely
            # regardless of the window_size configuration.
            nn.AdaptiveAvgPool1d(1)
        )

        # Fully Connected layers for RUL Regression
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Crucial for regularizing semi-supervised pseudo-labels
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        # PyTorch Conv1d needs: (batch_size, num_features, window_size)
        x = x.transpose(1, 2)

        features = self.conv_block(x)  # Shape: (batch_size, 128, 1)
        features = torch.flatten(features, 1)  # Shape: (batch_size, 128)

        rul_prediction = self.regressor(features)
        return rul_prediction
