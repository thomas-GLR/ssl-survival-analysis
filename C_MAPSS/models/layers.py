import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Adds a learned position embedding to a (batch, seq_len, d_model) tensor.

    Without positional information a Transformer encoder is permutation-invariant over
    its tokens. For a time-sequence model (tokens = time steps) that would discard the
    temporal order the RUL trend depends on, so we add a learned embedding per position.
    """

    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.position_embedding(positions)


class AttentionPooling(nn.Module):
    """Pools a (batch, seq_len, d_model) sequence into (batch, d_model).

    A learned attention score per time step (softmax over the time axis) forms a weighted
    sum, so the readout can emphasise the most informative steps instead of averaging them
    equally like global average pooling.
    """

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, h):
        # h: (batch, seq_len, d_model)
        scores = torch.softmax(self.attn(h), dim=1)  # (batch, seq_len, 1)
        return (scores * h).sum(dim=1)  # (batch, d_model)
