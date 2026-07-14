import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics.functional import mean_squared_error


def _cmapss_score(predict: np.ndarray, label: np.ndarray) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)


class BasicLightningModule(LightningModule):
    def __init__(
            self,
            lr,
            model: nn.Module,
            target_mean: float = 0.0,
            target_std: float = 1.0,
    ):
        super(BasicLightningModule, self).__init__()
        # We need to ignore model to prevent UnpicklingError since PyTorch 2.6 with weights_only=True
        self.save_hyperparameters(ignore=['model'])
        self.net = model
        self.lr = lr
        # RUL target standardization. The network learns/predicts in normalized
        # target space (loss is computed there), which keeps the MSE gradient on
        # an O(1) scale so the optimizer isn't dominated by the raw target
        # magnitude (mean ~117, std ~84). Predictions are de-normalized back to
        # real RUL units everywhere they are stored / returned, so all reported
        # metrics and saved predictions stay in original units. The defaults
        # (0, 1) make this a no-op for callers that don't pass stats.
        # These are hyperparameters (saved via save_hyperparameters) so they are
        # restored on load_from_checkpoint, which matters because predictions are
        # generated after reloading the best checkpoint.
        self.target_mean = target_mean
        self.target_std = target_std
        # On CMAPSS storing outputs is not a big problem as there is not too much data.
        # But on bigest dataset there is a risk of memory overflow
        self.training_step_outputs = []
        self.training_step_targets = []
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        # forward returns predictions in REAL RUL units (de-normalized). The
        # network itself learns in normalized target space (training_step feeds
        # self.net directly and normalizes y), but external callers of the module
        # -- notably Coprog._predict, which calls module(x) -> forward -- expect
        # real-unit predictions to stay consistent with the raw labels/targets
        # they compare against. The plain Lightning path never calls forward for
        # inference (it uses predict_step / test_step, which de-normalize too).
        return self._denorm(self.net(x))

    def _denorm(self, t: torch.Tensor) -> torch.Tensor:
        """De-normalize a prediction from standardized target space back to real RUL units.

        :param t: tensor of predictions in normalized target space.
        :return: tensor of predictions in original RUL units.
        """
        return t * self.target_std + self.target_mean

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)
        # Loss is computed in normalized target space; predictions are stored
        # de-normalized so train_rmse stays in real RUL units.
        y_norm = (y - self.target_mean) / self.target_std
        loss = F.mse_loss(preds, y_norm)
        self.training_step_outputs.extend(self._denorm(preds).detach())
        self.training_step_targets.extend(y.detach())
        return loss

    def on_train_epoch_end(self):
        outputs = torch.stack(self.training_step_outputs)
        targets = torch.stack(self.training_step_targets)

        rmse = mean_squared_error(outputs, targets, squared=False)

        self.training_step_outputs.clear()
        self.training_step_targets.clear()

        self.log('train_rmse', rmse, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)
        # Store de-normalized predictions so val_rmse / val_loss (monitored by
        # early stopping and checkpointing) stay in real RUL units.
        self.validation_step_outputs.extend(self._denorm(preds).detach())
        self.validation_step_targets.extend(y.detach())

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y = batch
        preds = self._denorm(self.net(x))
        self.test_step_outputs.extend(preds)
        self.test_step_targets.extend(y)

    def on_test_epoch_end(self):
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

    def on_validation_epoch_end(self):
        outputs = torch.stack(self.validation_step_outputs)
        targets = torch.stack(self.validation_step_targets)

        rmse = mean_squared_error(outputs, targets, squared=False)
        score = _cmapss_score(outputs.cpu().numpy().flatten(), targets.cpu().numpy().flatten())

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

        self.log('val_loss', rmse ** 2, prog_bar=True)
        self.log('val_rmse', rmse)
        self.log('val_score', score)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        """Run inference on a (features, target) batch and return both.

        :param batch: tuple of (features, target) tensors from the predict dataloader.
        :param batch_idx: index of the batch (unused, required by Lightning's signature).
        :return: tuple of (predictions, targets) tensors.
        """
        x, y = batch
        preds = self._denorm(self.net(x))
        return preds, y
