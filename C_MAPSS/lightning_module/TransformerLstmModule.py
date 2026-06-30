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


class TransformerLstmModule(LightningModule):
    def __init__(
            self,
            lr,
            model: nn.Module,
    ):
        super(TransformerLstmModule, self).__init__()
        # We need to ignore model to prevent UnpicklingError since PyTorch 2.6 with weights_only=True
        self.save_hyperparameters(ignore=['model'])
        self.net = model
        self.lr = lr
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = F.mse_loss(x, y)
        self.log('train_rmse', torch.sqrt(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)
        self.validation_step_outputs.extend(preds.detach())
        self.validation_step_targets.extend(y.detach())

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y = batch
        x = self.net(x)
        self.test_step_outputs.extend(x)
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
