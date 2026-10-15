from __future__ import annotations

from dataclasses import dataclass, field

import torch

from co_training import CoTrainingData


@dataclass
class ConformalCoTrainingEnsembleData(CoTrainingData):
    censored_lower_bounds: torch.Tensor | None
    censored_time_steps: torch.Tensor | None
    calib_features: torch.Tensor | None
    calib_targets: torch.Tensor | None