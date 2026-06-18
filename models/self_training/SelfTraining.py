import torch
import torch.nn as nn


class SelfTraining:
    def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-3,
            epochs: int = 20,
            batch_size: int = 32,
            device: torch.device | None = None,
            shuffle_dataloader: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_dataloader = shuffle_dataloader


    def train(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor
    ) -> None:
        remaining_suspension = suspension_data.clone()
