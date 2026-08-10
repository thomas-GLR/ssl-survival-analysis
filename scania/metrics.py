"""The official Scania Component X score, for the RUL-regression pipeline.

This module holds the *real* Scania metric, which replaces the C-MAPSS-style asymmetric
exponential score (``a1=13, a2=10``) that used to be reported under the name "Scania score"
everywhere on the Scania side. That formula is a RUL-regression metric with no connection to
Scania Component X; the challenge defines its metric on **failure-urgency classes** instead.

How it works: the models in this repository predict a remaining useful life, so both the
prediction and the target are binned into one of :data:`NUM_CLASSES` ordered urgency classes by
:func:`rul_to_class`, and the pair is charged the corresponding entry of
:data:`SCANIA_COST_MATRIX`. The result is a **sum** over samples (the ``Total_cost`` of the
Scania Component X paper), so it scales with the number of samples and is only comparable
between runs evaluated on the same split. **Lower is better** -- same direction as the score it
replaces, so nothing that consumes a score direction needed to change.

The score is only meaningful for **failure (uncensored) samples**, where the true RUL is known
exactly. Every caller in the pipeline already evaluates on uncensored rows only (the Lightning
val/test dataloaders go through ``ScaniaDataset.get_data_loader_without_censored_data``, and the
co-training tensors come from ``get_censored_split_tensors``); :func:`scania_score` additionally
drops any NaN pair defensively, since a censored row carries ``rul = NaN``.

The class definition and the cost matrix are shared with the ordinal classification pipeline on
the ``classification-problem`` branch (``models/classification/ordinal_metrics.py`` and
``scania/dataset/ScaniaClassificationDataset.py``).

This module lives directly under ``scania/`` rather than in ``scania/utils/`` on purpose:
``scania/utils/__init__.py`` imports the training orchestrators, which import
``scania.lightning_module``, so importing it from ``BasicLightningModule`` would be a circular
import. ``scania/__init__.py`` is empty, so ``scania.metrics`` is safe to import from anywhere.
"""

import numpy as np
import pandas as pd

# Number of ordered failure-urgency classes. Kept in sync with the cost matrix below, which
# must stay square.
NUM_CLASSES = 5

# Left-open, right-closed bins on the remaining useful life t (in time steps):
# t > 48 -> 0 ; 24 < t <= 48 -> 1 ; 12 < t <= 24 -> 2 ; 6 < t <= 12 -> 3 ; t <= 6 -> 4.
# Higher class index = more urgent.
_RUL_CLASS_BIN_EDGES = [-np.inf, 6, 12, 24, 48, np.inf]
_RUL_CLASS_LABELS = [4, 3, 2, 1, 0]

# Cost of predicting class m (column) when the actual class is n (row); the diagonal is free.
# Values are defined by Scania's experts (Sci. Data 12:493, table 1). n < m is a false positive
# (an unnecessary workshop check, single-digit cost); n > m is a false negative (a missed or late
# alarm, 200-500) -- the asymmetry is the whole point of the metric.
SCANIA_COST_MATRIX = np.array(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=np.float64,
)


def rul_to_class(rul: np.ndarray) -> np.ndarray:
    """Bin remaining useful lives into the 5 Scania failure-urgency classes.

    Uses left-open, right-closed intervals (``pd.cut`` with ``right=True``, the default), so a
    value falling exactly on an edge belongs to the *less* urgent class: ``t = 48`` is class 1
    and ``t = 6`` is class 4. Values below the lowest edge -- including negative RUL, which a
    regression model can predict -- fall into class 4, the most urgent one, which is the correct
    reading of "the component is already due".

    Args:
        rul: Remaining useful lives in time steps, any shape. ``NaN`` entries (censored rows,
            whose true RUL is unknown) are propagated as ``NaN``.

    Returns:
        Class labels as ``float64`` in ``[0.0, 4.0]``, same shape as ``rul`` flattened to 1-D.
        The dtype is float rather than int so ``NaN`` survives for the caller to filter.
    """
    values = pd.Series(np.asarray(rul, dtype=np.float64).reshape(-1))
    classes = pd.cut(values, bins=_RUL_CLASS_BIN_EDGES, labels=_RUL_CLASS_LABELS, right=True)
    return classes.astype(np.float64).to_numpy()


def scania_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Total Scania cost of a set of RUL predictions (lower is better).

    Converts both sides to failure-urgency classes with :func:`rul_to_class` and sums
    ``SCANIA_COST_MATRIX[actual_class, predicted_class]`` over every sample, i.e. the
    ``Total_cost = Cost_n_m * No instances`` of the Scania Component X paper.

    Args:
        predictions: Predicted RUL in time steps, any shape.
        targets: True RUL in time steps, same number of elements as ``predictions``.

    Returns:
        The summed cost. ``0.0`` if no usable sample remains (empty input, or every pair
        containing a ``NaN``).

    Raises:
        ValueError: If ``predictions`` and ``targets`` do not hold the same number of elements.
    """
    predicted_classes = rul_to_class(predictions)
    target_classes = rul_to_class(targets)

    if predicted_classes.size != target_classes.size:
        raise ValueError(
            f"predictions and targets must have the same number of elements, got "
            f"{predicted_classes.size} and {target_classes.size}."
        )

    # A censored row has no known RUL (NaN), so it has no ground-truth class and cannot be
    # scored. Callers already pass uncensored rows only; this is a guard against silently
    # indexing the cost matrix with a NaN.
    usable = ~(np.isnan(predicted_classes) | np.isnan(target_classes))
    if not usable.any():
        return 0.0

    rows = target_classes[usable].astype(np.int64)
    columns = predicted_classes[usable].astype(np.int64)

    return float(SCANIA_COST_MATRIX[rows, columns].sum())
