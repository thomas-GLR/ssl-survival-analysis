from torch import nn

from models.self_supervised.base.Decoder import Decoder
from models.self_supervised.base.Encoder import Encoder


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.encoder = Encoder(
            self.in_channels,
            self.base_filters,
            self.kernel_size,
            self.num_layers,
            self.latent_dim,
            self.seq_len,
            self.dropout,
            norm_outputs=True,
        )
        self.decoder = Decoder(
            self.in_channels,
            self.base_filters,
            self.kernel_size,
            self.num_layers,
            self.latent_dim,
            self.seq_len,
            self.dropout,
        )

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        outputs = self.decoder(latent_code)

        return outputs
