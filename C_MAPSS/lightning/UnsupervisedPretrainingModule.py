from abc import abstractmethod, ABC

import pytorch_lightning as pl
import torch
import torch.nn as nn

from C_MAPSS.lightning import metrics
from models.self_supervised.base.Encoder import Encoder


class UnsupervisedPretrainingModule(pl.LightningModule, ABC):
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
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.encoder = self._get_encoder()

        self.criterion_regression = nn.MSELoss()
        self.regression_metric = metrics.SimpleMetric()
        self.rmse_loss_metric = metrics.RMSELoss()

    def _get_encoder(self):
        return Encoder(
            self.in_channels,
            self.base_filters,
            self.kernel_size,
            self.num_layers,
            self.latent_dim,
            self.seq_len,
            self.dropout,
            norm_outputs=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def training_step(self, batch, batch_idx):
        regression_loss = self._get_losses(batch)
        loss = regression_loss
        self.log("train/regression_loss", regression_loss)

        return loss

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def _reset_all_metrics(self):
        self.regression_metric.reset()
        self.rmse_loss_metric.reset()

    def validation_step(self, batch, batch_idx):
        regression_loss = self._get_losses(batch)
        batch_size = batch[0].shape[0]
        self.regression_metric.update(regression_loss, batch_size)
        self.rmse_loss_metric.update(regression_loss, batch_size)

    def on_validation_epoch_end(self):
        regression_loss = self.regression_metric.compute()
        rmse_loss = self.rmse_loss_metric.compute()

        self.log("val/regression_loss", regression_loss)
        self.log("val/rmse_loss", rmse_loss)

    @abstractmethod
    def _get_losses(self, batch) -> torch.Tensor:
        pass