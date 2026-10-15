from __future__ import annotations

from dataclasses import dataclass, field

import torch

from scania.dataset import ScaniaDataModule


@dataclass
class CoTrainingData:
    feature_num: int
    features_uncensored: torch.Tensor
    targets_uncensored: torch.Tensor
    features_censored: torch.Tensor
    ids_censored: torch.Tensor
    test_features: torch.Tensor
    test_targets: torch.Tensor
    val_features: torch.Tensor
    val_targets: torch.Tensor
    scania_data_module: ScaniaDataModule | None = field(repr=False, compare=False)
