import torch.nn as nn

from C_MAPSS.models.layers import AttentionPooling, LearnedPositionalEncoding


class TransformerTimeSequence(nn.Module):
    def __init__(
            self,
            feature_num,
            sequence_len,
            d_model,
            transformer_encoder_head_num,
            fc_layer_dim,
            fc_dropout,
            num_layers=2,
    ):
        super().__init__()
        if d_model % transformer_encoder_head_num != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by transformer_encoder_head_num ({transformer_encoder_head_num})"
            )

        self.feature_num = feature_num
        self.sequence_len = sequence_len
        self.d_model = d_model

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        self.output_dim = 1

        self.transformer_encoder_head_num = transformer_encoder_head_num

        # Tokens are time steps: x arrives as (batch, seq_len, feature_num), so each token's
        # raw embedding is feature_num-wide. That generally doesn't match d_model (a free
        # hyperparameter, validated above against transformer_encoder_head_num), so we project
        # each token from feature_num -> d_model first, like a standard sequence transformer.
        self.input_projection = nn.Linear(self.feature_num, self.d_model)

        # Self-attention is permutation-invariant over tokens, so without positional
        # information the model would treat the window as an unordered bag of time steps and
        # lose the degradation trend RUL depends on. Add a learned position embedding.
        self.positional_encoding = LearnedPositionalEncoding(self.sequence_len, self.d_model)

        # transformer encoder (stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_encoder_head_num,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling over the time axis: a learned weighted sum that can emphasise the
        # most informative (e.g. most recent) steps instead of averaging them equally.
        self.attention_pooling = AttentionPooling(self.d_model)

        # fc layers (linear output; no terminal ReLU so negative pre-activations still learn)
        self.linear = nn.Sequential(
            nn.Linear(self.d_model, self.fc_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_num) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.attention_pooling(x)  # (batch, d_model)

        x = self.linear(x)

        return x
