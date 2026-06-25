import torch
from torch import nn as nn
from torchmetrics import Metric


class RMSELoss(Metric):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

        self.add_state("losses", default=[], dist_reduce_fx=None)
        self.add_state("sizes", default=[], dist_reduce_fx=None)

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        summed_square = nn.functional.mse_loss(inputs, targets, reduction="sum")
        batch_size = inputs.shape[0]

        self.losses.append(summed_square)
        self.sizes.append(torch.tensor(batch_size, dtype=torch.float))

    def compute(self) -> torch.Tensor:
        if len(self.losses) == 0:
            raise RuntimeError("RMSE metric was not used. Computation impossible.")
        summed_squares = torch.stack(self.losses)
        batch_sizes = torch.stack(self.sizes)
        rmse = torch.sqrt(summed_squares.sum() / batch_sizes.sum())

        return rmse

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


class SimpleMetric(Metric):
    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction not in ["sum", "mean"]:
            raise ValueError(f"Unsupported reduction {reduction}")
        self.reduction = reduction

        self.add_state("losses", default=[], dist_reduce_fx=None)
        self.add_state("sizes", default=[], dist_reduce_fx=None)

    def update(self, loss: torch.Tensor, batch_size: int):
        self.losses.append(loss)
        self.sizes.append(torch.tensor(batch_size, dtype=torch.float))

    def compute(self) -> torch.Tensor:
        if len(self.losses) == 0:
            raise RuntimeError("Metric was not updated. Computation impossible.")
        if self.reduction == "mean":
            return self._weighted_mean()
        return self._sum()

    def _weighted_mean(self):
        losses = torch.stack(self.losses)
        weights = torch.stack(self.sizes)
        weights = weights / weights.sum()
        return torch.sum(losses * weights)

    def _sum(self):
        losses = torch.stack(self.losses)
        return torch.sum(losses)
