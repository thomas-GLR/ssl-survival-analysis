import torch
from lightning import LightningModule

from C_MAPSS.lightning_module import metrics
from C_MAPSS.lightning_module.mixins import LoadEncoderMixin, DataHparamsMixin
from models.self_supervised.base.BaselineRegressor import BaselineRegressor
from models.self_supervised.base.Encoder import Encoder


class BaselineModule(LightningModule, LoadEncoderMixin, DataHparamsMixin):
    def __init__(
            self,
            in_channels: int,
            seq_len: int,
            latent_dim: int,
            base_filters: int,
            kernel_size: int,
            num_layers: int,
            dropout: float,
            lr: float,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lr = lr

        self.encoder = Encoder(
            in_channels=self.in_channels,
            base_filters=self.base_filters,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            dropout=self.dropout,
            norm_outputs=False,
        )

        self.regressor = BaselineRegressor(latent_dim)

        self.regression_metrics = metrics.RMSELoss()

        self.save_hyperparameters()

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters()},
            {"params": self.regressor.parameters()},
        ]
        return torch.optim.Adam(param_groups, lr=self.lr)

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        prediction = self.regressor(latent_code)

        return prediction

    def training_step(self, batch, batch_idx):
        source, source_labels = batch
        predictions = self(source)
        loss = self.regression_metrics(predictions, source_labels)

        self.log("train/regression_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def on_test_epoch_start(self):
        self._reset_all_metrics()

    def _reset_all_metrics(self):
        self.regression_metrics.reset()

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch)

    def test_step(self, batch, batch_idx):
        self._evaluate(batch)

    def _evaluate(self, batch):
        features, labels = batch
        predictions = self(features)
        self.regression_metrics.update(predictions, labels)

    def on_validation_epoch_end(self):
        self.log("val/regression_loss", self.regression_metrics.compute())

    def on_test_epoch_end(self):
        self.log("test/regression_loss", self.regression_metrics.compute())
