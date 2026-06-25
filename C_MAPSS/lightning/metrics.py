import torch
from torch import nn as nn
from torchmetrics import Metric


class RMSELoss(Metric):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

        self.add_state("losses", default=[], dist_reduce_fx=None)
        self.add_state("sizes", default=[], dist_reduce_fx=None)
        self.add_state("sample_counter", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        summed_square = nn.functional.mse_loss(inputs, targets, reduction="sum")
        batch_size = inputs.shape[0]

        self.losses[self.sample_counter] = summed_square
        self.sizes[self.sample_counter] = batch_size
        self.sample_counter.add_(1)

    def compute(self) -> torch.Tensor:
        if self.sample_counter == 0:
            raise RuntimeError("RMSE metric was not used. Computation impossible.")
        summed_squares = self.losses[: self.sample_counter]
        batch_sizes = self.sizes[: self.sample_counter]
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
        self.add_state("sample_counter", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, loss: torch.Tensor, batch_size: int):
        self.losses[self.sample_counter] = loss
        self.sizes[self.sample_counter] = batch_size
        self.sample_counter.add_(1)

    def compute(self) -> torch.Tensor:
        if self.sample_counter == 0:
            raise RuntimeError("RMSE metric was not used. Computation impossible.")
        if self.reduction == "mean":
            loss = self._weighted_mean()
        else:
            loss = self._sum()

        return loss

    def _weighted_mean(self):
        weights = self.sizes[: self.sample_counter]
        weights = weights / weights.sum()
        loss = self.losses[: self.sample_counter]
        loss = torch.sum(loss * weights)

        return loss

    def _sum(self):
        loss = self.losses[: self.sample_counter]
        loss = torch.sum(loss)

        return loss


class RULScore:
    def __init__(self, pos_factor=10, neg_factor=-13):
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor

    def __call__(self, inputs, targets):
        dist = inputs - targets
        for i, d in enumerate(dist):
            dist[i] = (d / self.neg_factor) if d < 0 else (d / self.pos_factor)
        dist = torch.exp(dist) - 1
        score = dist.sum()

        return score
