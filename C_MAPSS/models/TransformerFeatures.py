import torch.nn as nn


class TransformerFeatures(nn.Module):
    def __init__(
            self,
            feature_num,
            sequence_len,
            transformer_encoder_head_num,
            fc_layer_dim,
            fc_dropout,
            num_layers=2,
    ):
        super().__init__()
        # Tokens are feature channels and each token's embedding is that sensor's raw time
        # series, so the model dimension is fixed to sequence_len (not a free hyperparameter).
        if sequence_len % transformer_encoder_head_num != 0:
            raise ValueError(
                f"sequence_len ({sequence_len}) must be divisible by "
                f"transformer_encoder_head_num ({transformer_encoder_head_num})"
            )

        self.feature_num = feature_num
        self.sequence_len = sequence_len
        self.d_model = sequence_len

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        self.output_dim = 1

        self.transformer_encoder_head_num = transformer_encoder_head_num

        # transformer encoder (stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_encoder_head_num,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # fc layers (linear output; no terminal ReLU so negative pre-activations still learn)
        self.linear = nn.Sequential(
            nn.Linear(self.feature_num, self.fc_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
        )

    def forward(self, x):
        # The permutation enables the transformer to attend across feature channels
        # (each channel becomes a token, its time series is the token's d_model embedding).
        # No positional encoding: feature channels have no inherent order.
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        # No re-permute: gap pools over the last dim (d_model / time), keeping feature_num
        # as the surviving channel dimension, so the output is length-invariant like CNN1D's.
        x = self.gap(x)
        x = self.flatten(x)

        x = self.linear(x)

        return x
