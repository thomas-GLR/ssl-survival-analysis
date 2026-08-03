"""Metrics for the 5-class Scania failure-urgency ordinal problem.

Every metric here takes **class-index** tensors (values in ``{0, ..., NUM_CLASSES - 1}``, any
shape, any numeric dtype) and returns a plain ``float``, so they drop straight into the
``criteria_callback`` / ``score_callback`` slots the co-training ensembles already expose.

Two families are provided:

- *label-agnostic* ordinal metrics (:func:`accuracy`, :func:`f1_macro`, :func:`mae`,
  :func:`rmse`). ``mae`` / ``rmse`` are meaningful here precisely because the classes are
  **ordered**: mistaking class 0 for class 4 is worse than mistaking it for class 1.
- the *Scania cost* (:func:`scania_cost`), the company-defined asymmetric cost of a
  confusion matrix. Missing a faulty truck (predicting a lower urgency than the truth) is far
  more expensive than an unnecessary workshop check, which is why the lower triangle of
  :data:`SCANIA_COST_MATRIX` dwarfs its upper triangle.

:data:`ORDINAL_METRICS` and :data:`METRIC_DIRECTION` are the single source of truth used by
``CoTrainingEnsembleOrdinalRegression`` to resolve a ``ComputingWeightMode`` /
``ImprovingMetricMode`` into "which callable, and is bigger better?".
"""

from typing import Callable

import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score

# Number of ordered failure-urgency classes (see ScaniaClassificationDataset: 0 is "t > 48",
# 4 is "t <= 6"). Kept in sync with the cost matrix below, which must stay square.
NUM_CLASSES = 5

# Cost of predicting class m (column) when the actual class is n (row); the diagonal is free.
# Values are defined by Scania's experts (Sci. Data 12:493, table 1). n < m is a false
# positive (an unnecessary check, single-digit cost); n > m is a false negative (a missed or
# late alarm, 200-500) — the asymmetry is the whole point of the metric.
SCANIA_COST_MATRIX = torch.tensor(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=torch.float64,
)


def _as_class_index(t: torch.Tensor) -> torch.Tensor:
    """Flatten a prediction/target tensor to a 1-D ``long`` vector of class indices.

    Args:
        t: Class indices of any shape and numeric dtype (the datasets store them as
            ``float32``, argmax outputs are already integral).

    Returns:
        A 1-D ``long`` tensor on the CPU.

    Raises:
        ValueError: If any value falls outside ``[0, NUM_CLASSES - 1]``, which would silently
            corrupt the cost lookup and the torchmetrics calls.
    """
    flat = t.detach().cpu().reshape(-1).round().long()
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= NUM_CLASSES):
        raise ValueError(
            f"class indices must be in [0, {NUM_CLASSES - 1}], got "
            f"[{int(flat.min())}, {int(flat.max())}]."
        )
    return flat


def scania_cost(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Total Scania cost of a set of predictions (lower is better).

    Sums ``SCANIA_COST_MATRIX[actual, predicted]`` over every sample, i.e. the
    ``Total_cost = Cost_n_m * No instances`` of the Scania Component X paper.

    Args:
        preds: Predicted class indices.
        target: Actual class indices, same number of elements as ``preds``.

    Returns:
        The summed cost.
    """
    p = _as_class_index(preds)
    t = _as_class_index(target)
    return float(SCANIA_COST_MATRIX[t, p].sum())


def accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Micro accuracy over the 5 classes (higher is better).

    Args:
        preds: Predicted class indices.
        target: Actual class indices.

    Returns:
        The fraction of exactly-correct predictions.
    """
    return float(
        multiclass_accuracy(
            _as_class_index(preds), _as_class_index(target),
            num_classes=NUM_CLASSES, average="micro",
        )
    )


def f1_macro(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Macro-averaged F1 over the 5 classes (higher is better).

    Macro (not micro) averaging is used because the urgency classes are heavily imbalanced —
    most readouts are far from failure — and the rare urgent classes are the ones that matter.

    Args:
        preds: Predicted class indices.
        target: Actual class indices.

    Returns:
        The unweighted mean of the per-class F1 scores.
    """
    return float(
        multiclass_f1_score(
            _as_class_index(preds), _as_class_index(target),
            num_classes=NUM_CLASSES, average="macro",
        )
    )


def mae(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error on the class index (lower is better).

    Args:
        preds: Predicted class indices.
        target: Actual class indices.

    Returns:
        The mean absolute ordinal distance between prediction and truth.
    """
    p = _as_class_index(preds).double()
    t = _as_class_index(target).double()
    return float((p - t).abs().mean())


def rmse(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error on the class index (lower is better).

    Args:
        preds: Predicted class indices.
        target: Actual class indices.

    Returns:
        The root mean squared ordinal distance between prediction and truth.
    """
    p = _as_class_index(preds).double()
    t = _as_class_index(target).double()
    return float(((p - t) ** 2).mean().sqrt())


# Name -> callable, and name -> whether a *bigger* value is better. The ensemble resolves its
# ComputingWeightMode / ImprovingMetricMode through these two dicts so weighting and
# keep-best-model can never disagree on a metric's direction.
ORDINAL_METRICS: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "mae": mae,
    "rmse": rmse,
    "scania_cost": scania_cost,
}

METRIC_DIRECTION: dict[str, str] = {
    "accuracy": "max",
    "f1_macro": "max",
    "mae": "min",
    "rmse": "min",
    "scania_cost": "min",
}
