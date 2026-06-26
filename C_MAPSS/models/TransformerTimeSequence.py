import torch
import torch.nn as nn


class TransformerTimeSequence(nn.Module):
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

        # Unlike TransformerFeatures (which permutes so feature channels become
        # tokens), this model keeps time steps as tokens: x arrives as
        # (batch, seq_len, feature_num), so each token's raw embedding is
        # feature_num-wide. That generally doesn't match d_model (a free
        # hyperparameter, validated above against transformer_encoder_head_num),
        # so we project each token from feature_num -> d_model first, exactly
        # like the input embedding layer in a standard sequence transformer.
        self.input_projection = nn.Linear(self.feature_num, self.d_model)

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
            nn.Linear(self.d_model, self.fc_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_num) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        # Permute to channels-first so gap pools over seq_len (time) and keeps
        # d_model as the surviving channel axis, matching the GAP convention
        # used across C_MAPSS/models (see CNN1D.py / TransformerFeatures.py).
        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = self.flatten(x)

        x = self.linear(x)

        return x
