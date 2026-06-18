import torch
from torch import nn

from models.self_supervised.base.BaselineRegressor import BaselineRegressor
from models.self_supervised.base.Encoder import Encoder


class Baseline(nn.Module):
    def __init__(
            self,
            encoder: Encoder
    ):
        super().__init__()

        self.encoder = encoder
        self.latent_dim = encoder.latent_dim

        self.regressor = BaselineRegressor(self.latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent_code = self.encoder(inputs)
        prediction = self.regressor(latent_code)

        return prediction

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        criterion = nn.MSELoss()
        # RMSE Loss as in the original paper
        loss = torch.sqrt(criterion(predictions, targets))
        return loss
