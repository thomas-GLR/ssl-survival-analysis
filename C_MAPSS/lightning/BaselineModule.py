import pytorch_lightning as pl
import torch

from C_MAPSS.lightning import metrics
from C_MAPSS.lightning.mixins import LoadEncoderMixin, DataHparamsMixin
from models.self_supervised.base.BaselineRegressor import BaselineRegressor
from models.self_supervised.base.Encoder import Encoder


class BaselineModule(pl.LightningModule, LoadEncoderMixin, DataHparamsMixin):
    def __init__(
        self,
        encoder: Encoder,
        in_channels: int,
        seq_len: int,
        latent_dim: int,
        lr: float,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.lr = lr

        self.encoder = encoder
        self.regressor = BaselineRegressor(latent_dim)

        self.criterion_regression = metrics.RMSELoss()

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
        loss = self.criterion_regression(predictions, source_labels)

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
