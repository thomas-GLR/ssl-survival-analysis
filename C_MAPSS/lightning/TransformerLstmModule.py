import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics.functional import mean_squared_error


def _cmapss_score(predict: np.ndarray, label: np.ndarray) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)


class TransformerLstmModule(pl.LightningModule):
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
        self.validation_step_losses = []
        self.validation_step_lengths = []
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
        x = self.net(x)
        loss = F.mse_loss(x, y, reduction='sum')
        self.validation_step_losses.append(loss)
        self.validation_step_lengths.append(len(y))

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
        # Calculate the average loss
        mse = torch.sum(torch.tensor(self.validation_step_losses)) / torch.sum(torch.tensor(self.validation_step_lengths))
        rmse = torch.sqrt(mse)
        # Clear the lists
        self.validation_step_losses.clear()
        self.validation_step_lengths.clear()
        # Log the results
        self.log('val_loss', mse, prog_bar=True)
        self.log('val_rmse', rmse)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer
