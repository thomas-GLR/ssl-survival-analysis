import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        num_layers,
        latent_dim,
        seq_len,
        dropout,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout

        self.layers = self._build_decoder()

    @property
    def padding(self):
        left = self.kernel_size - 1
        right = 0

        return left, right

    def _build_decoder(self):
        cut_off = self.num_layers // 2 * (self.kernel_size - (self.kernel_size % 2))
        max_filters = min(self.num_layers * self.base_filters, 64)
        reduced_seq_len = self.seq_len - cut_off
        flat_dim = reduced_seq_len * max_filters

        sequence = [
            nn.Linear(self.latent_dim, flat_dim),
            nn.BatchNorm1d(flat_dim),
            nn.ReLU(True),
            layers.DeFlatten(reduced_seq_len, max_filters),
        ]
        for i in range(self.num_layers - 1, 0, -1):
            in_filters = min((i + 1) * self.base_filters, 64)
            out_filters = min(i * self.base_filters, 64)
            use_padding = i % 2 == 1
            sequence.extend(self._build_conv_layer(in_filters, out_filters, use_padding))

        sequence.extend(
            [
                nn.ConvTranspose1d(self.base_filters, self.in_channels, self.kernel_size),
                nn.Tanh(),
            ]
        )

        return nn.Sequential(*sequence)

    def _build_conv_layer(self, in_filters, out_filters, use_padding):
        layer = []
        if use_padding:
            layer.extend(
                [
                    nn.ConstantPad1d(self.padding, 0.0),
                    nn.Conv1d(in_filters, out_filters, self.kernel_size, bias=False),
                ]
            )
        else:
            layer.append(
                nn.ConvTranspose1d(in_filters, out_filters, self.kernel_size, bias=False)
            )
        layer.extend(
            [
                nn.BatchNorm1d(out_filters),
                nn.ReLU(True),
                nn.Dropout2d(p=self.dropout),
            ]
        )

        return layer

    def forward(self, inputs):
        outputs = inputs
        for m in self.layers:
            outputs = m(outputs)

        return outputs
