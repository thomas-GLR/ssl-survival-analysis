import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        num_layers,
        latent_dim,
        seq_len,
        dropout,
        norm_outputs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.norm_outputs = norm_outputs

        self.layers = self._build_encoder()

    @property
    def padding(self):
        left = self.kernel_size - 1
        right = 0

        return left, right

    def _build_encoder(self):
        sequence = [
            nn.Conv1d(self.in_channels, self.base_filters, self.kernel_size, bias=False),
            nn.BatchNorm1d(self.base_filters),
            nn.ReLU(True),
        ]
        for i in range(1, self.num_layers):
            in_filters = min(i * self.base_filters, 64)
            out_filters = min((i + 1) * self.base_filters, 64)
            use_padding = i % 2 == 1
            sequence.extend(self._build_conv_layer(in_filters, out_filters, use_padding))

        cut_off = self.num_layers // 2 * (self.kernel_size - (self.kernel_size % 2))
        flat_dim = (self.seq_len - cut_off) * min(self.num_layers * self.base_filters, 64)
        sequence.extend([nn.Flatten(), nn.Linear(flat_dim, self.latent_dim)])

        return nn.Sequential(*sequence)

    def _build_conv_layer(self, in_filters, out_filters, use_padding):
        layer = []
        if use_padding:
            layer.append(nn.ConstantPad1d(self.padding, 0.0))
        layer.extend(
            [
                nn.Conv1d(in_filters, out_filters, self.kernel_size, bias=False),
                nn.BatchNorm1d(out_filters),
                nn.ReLU(True),
                nn.Dropout2d(p=self.dropout),
            ]
        )

        return layer

    def forward(self, inputs):
        """
        Forward function of the encoder model.

        :param inputs: inputs for Conv1d should be of shape (N, C, L) where :
            - N = Batch size
            - C = Number of channels (input features)
            - L = Sequence length
        :return: outputs of the encoder model where the size corresponds to the latent dimension
        """
        outputs = self.layers(inputs)
        if self.norm_outputs:
            outputs = outputs / torch.norm(outputs, dim=1, keepdim=True)

        return outputs