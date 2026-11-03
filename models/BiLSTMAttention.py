import torch
from torch import nn


class BiLSTMAttention(nn.Module):
    """Four-layer bidirectional LSTM with temporal attention for RUL regression.

    Adapted from Wang et al. (EMNLP 2016), https://aclanthology.org/D16-1058/.
    Uses additive attention and context/final-state fusion without aspect inputs.
    The bidirectional four-layer encoder is a task-specific extension, with
    hidden_dim units PER DIRECTION in each layer. Inputs are fixed-size windows.
    """

    def __init__(self, num_features: int, sequence_len: int,
                 hidden_dim: int = 128, lstm_num_layers: int = 4):
        super().__init__()
        self.n_features = num_features
        self.sequence_length = sequence_len
        self.lstm = nn.LSTM(
            input_size=num_features, hidden_size=hidden_dim,
            num_layers=lstm_num_layers, batch_first=True, bidirectional=True,
        )
        encoded_dim = 2 * hidden_dim
        self.attention_projection = nn.Linear(encoded_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.context_projection = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.final_projection = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.regressor = nn.Linear(encoded_dim, 1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, sequence_length, n_features]")
        if x.shape[1:] != (self.sequence_length, self.n_features):
            raise ValueError("Input sequence length or feature count is incorrect")
        outputs, (hidden, _) = self.lstm(x)
        scores = self.attention_score(torch.tanh(self.attention_projection(outputs)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * outputs).sum(dim=1)
        # Last layer's forward and backward summaries (not outputs[:, -1]).
        final_state = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        representation = torch.tanh(
            self.context_projection(context) + self.final_projection(final_state)
        )
        return self.regressor(representation).squeeze(-1)
