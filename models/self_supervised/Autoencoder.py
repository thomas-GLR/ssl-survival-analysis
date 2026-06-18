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
        dropout,
        domain_tradeoff: float = 0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.domain_tradeoff = domain_tradeoff

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

    def compute_loss(self, anchors, outputs, inputs):
        regression_loss = nn.MSELoss()(outputs, inputs)

        if self.domain_tradeoff > 0:
            # TODO Check if it used in the original paper and if yes need to replace domain_disc
            # batch_size = anchors.shape[0]
            # domain_pred = self.domain_disc(embeddings[:batch_size])
            # domain_loss = nn.BCEWithLogitsLoss()(domain_pred, domain_labels)
            domain_loss = 0
        else:
            domain_loss = 0

        return regression_loss + self.domain_tradeoff * domain_loss
