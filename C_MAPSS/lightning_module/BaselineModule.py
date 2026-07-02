import numpy as np
import torch
from lightning import LightningModule
from torchmetrics.functional import mean_squared_error

from C_MAPSS.lightning_module import metrics
from C_MAPSS.lightning_module.mixins import LoadEncoderMixin, DataHparamsMixin
from models.self_supervised.base.BaselineRegressor import BaselineRegressor
from models.self_supervised.base.Encoder import Encoder

def _cmapss_score(predict: np.ndarray, label: np.ndarray) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)

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

        # On CMAPSS storing outputs is not a big problem as there is not too much data.
        # But on bigest dataset there is a risk of memory overflow
        self.training_step_outputs = []
        self.training_step_targets = []
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

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

        predictions_reshaped = predictions.view(-1, source_labels.shape[1]) if source_labels.dim() > 1 else predictions.view(-1)

        loss = self.regression_metrics(predictions_reshaped, source_labels)

        self.training_step_outputs.extend(predictions_reshaped.detach())
        self.training_step_targets.extend(source_labels.detach())

        self.log("train/regression_loss", loss)

        return loss

    def on_train_epoch_end(self):
        outputs = torch.stack(self.training_step_outputs)
        targets = torch.stack(self.training_step_targets)

        rmse = mean_squared_error(outputs, targets, squared=False)

        self.training_step_outputs.clear()
        self.training_step_targets.clear()

        self.log('train_rmse', rmse, prog_bar=True)

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def on_test_epoch_start(self):
        self._reset_all_metrics()

    def _reset_all_metrics(self):
        self.regression_metrics.reset()

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, mode: str):
        features, labels = batch
        predictions = self(features)

        predictions_reshaped = predictions.view(-1, labels.shape[1]) if labels.dim() > 1 else predictions.view(-1)

        match mode:
            case "test":
                self.test_step_outputs.extend(predictions_reshaped)
                self.test_step_targets.extend(labels)
            case "val":
                self.validation_step_outputs.extend(predictions_reshaped.detach())
                self.validation_step_targets.extend(labels.detach())
            case _:
                raise ValueError(f"Unknown mode \"{mode}\", please select between modes (test, val)")

        self.regression_metrics.update(predictions_reshaped, labels)

    def on_validation_epoch_end(self):
        self.log("val/regression_loss", self.regression_metrics.compute())

        outputs = torch.stack(self.validation_step_outputs)
        targets = torch.stack(self.validation_step_targets)

        rmse = mean_squared_error(outputs, targets, squared=False)
        score = _cmapss_score(outputs.cpu().numpy().flatten(), targets.cpu().numpy().flatten())

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

        self.log('val_loss', rmse ** 2, prog_bar=True)
        self.log('val_rmse', rmse)
        self.log('val_score', score)

    def on_test_epoch_end(self):
        self.log("test/regression_loss", self.regression_metrics.compute())

        outputs = torch.tensor(self.test_step_outputs)
        targets = torch.tensor(self.test_step_targets)

        rmse = mean_squared_error(
            outputs,
            targets,
            squared=False
        )

        np_outputs, np_targets = outputs.cpu().numpy(), targets.cpu().numpy()

        score = _cmapss_score(np_outputs, np_targets)

        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        self.log('test_rmse', rmse)
        self.log('test_score', score)
