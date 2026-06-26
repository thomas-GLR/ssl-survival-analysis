import torch
import torch.nn as nn


class TransformerFeatures(nn.Module):
    def __init__(
            self,
            feature_num,
            d_model,
            transformer_encoder_head_num,
            fc_layer_dim,
            fc_dropout
    ):
        super().__init__()
        if d_model % transformer_encoder_head_num != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by transformer_encoder_head_num ({transformer_encoder_head_num})"
            )

        self.feature_num = feature_num
        self.d_model = d_model

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        self.output_dim = 1

        self.transformer_encoder_head_num = transformer_encoder_head_num

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_encoder_head_num,
            batch_first=True,
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # fc layers
        self.linear = nn.Sequential(
            nn.Linear(self.feature_num, self.fc_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # The permutation enables the transformer to attend across feature channels
        # (each channel becomes a token, its time series is the token's d_model embedding).
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        # No re-permute: gap pools over the last dim (d_model / time), keeping feature_num
        # as the surviving channel dimension, so the output is length-invariant like CNN1D's.
        x = self.gap(x)
        x = self.flatten(x)

        x = self.linear(x)

        return x
