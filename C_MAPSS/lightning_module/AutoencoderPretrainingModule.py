import torch

from C_MAPSS.lightning_module.UnsupervisedPretrainingModule import UnsupervisedPretrainingModule
from models.self_supervised.base.Decoder import Decoder


class AutoencoderPretrainingModule(UnsupervisedPretrainingModule):
    def __init__(
        self,
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout,
        lr,
        weight_decay,
    ):
        super().__init__(
            in_channels=in_channels,
            seq_len=seq_len,
            num_layers=num_layers,
            kernel_size=kernel_size,
            base_filters=base_filters,
            latent_dim=latent_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
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

        self.hparams["mode"] = "autoencoder"

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        outputs = self.decoder(latent_code)

        return outputs

    def _get_losses(self, batch):
        anchors, queries, true_distances = batch
        combined = torch.cat([anchors, queries])

        embeddings = self.encoder(combined)
        outputs = self.decoder(embeddings)

        regression_loss = self.criterion_regression(outputs, combined)

        return regression_loss
