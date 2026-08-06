"""
Scania Component X classification dataset.

``ScaniaClassificationDataset`` predicts a 5-class failure-urgency label from
a vehicle's operational readouts, sharing all preprocessing (windowing,
normalization, censoring bookkeeping) with :class:`ScaniaBaseDataset`.

Let ``t = length_of_study_time_step - time_step`` (the same quantity
``ScaniaRegressionDataset.count_rul`` uses as RUL). The 5 classes are, in
increasing order of urgency (left-open, right-closed intervals):

    class 0: t > 48
    class 1: 24 < t <= 48
    class 2: 12 < t <= 24
    class 3: 6  < t <= 12
    class 4: t <= 6

``_compute_labels`` auto-detects which of two label sources to use, based on
whether ``CLASS_LABEL`` is already a column of the input dataframe:

- **Mode A (TTE-derived, train)**: no ``CLASS_LABEL`` column yet.
  ``length_of_study_time_step``/``time_step``/``is_censored`` are used to
  derive the class the same way RUL is derived: ``CLASS_UPPER_BOUND`` is
  always set to ``rul_to_class(t)`` (mathematically valid even for censored
  rows -- since the true remaining time ``t_true >= t`` and class is a
  non-increasing step function of ``t``, ``class(t)`` is an upper bound on
  the true unknown class), and ``CLASS_LABEL`` is the same value, NaN'd for
  censored rows (the true class is unknown there).
- **Mode B (provided-label, val/test)**: ``CLASS_LABEL`` is already present
  (pre-merged by the caller from ``validation_labels.csv``/``test_labels.csv``
  -- future work, not implemented by this class). Used as-is, no NaN'ing:
  those files contain only failure/event vehicles, so every label is known
  ground truth. ``CLASS_UPPER_BOUND`` is simply set equal to ``CLASS_LABEL``.
"""

import numpy as np
import pandas as pd

from constants.scania_component_x_columns import LENGTH_OF_STUDY_TIME_STEP, TIME_STEP, CLASS_LABEL
from scania.dataset.ScaniaBaseDataset import ScaniaBaseDataset, IS_CENSORED

# Column produced by this class / expected in the pre-processed frame (Mode B).
CLASS_UPPER_BOUND = "class_upper_bound"

# Left-open, right-closed bins on t = length_of_study_time_step - time_step.
# t > 48 -> 0 ; 24 < t <= 48 -> 1 ; 12 < t <= 24 -> 2 ; 6 < t <= 12 -> 3 ; t <= 6 -> 4
_RUL_CLASS_BIN_EDGES = [-np.inf, 6, 12, 24, 48, np.inf]
_RUL_CLASS_LABELS = [4, 3, 2, 1, 0]


def rul_to_class(t: np.ndarray | pd.Series) -> np.ndarray:
    """Bin a remaining-time quantity into the 5 Scania failure-urgency classes.

    Uses left-open, right-closed intervals (``pd.cut`` with ``right=True``,
    the default): class 0 is ``t > 48`` and class 4 is ``t <= 6``.

    :param t: array/Series of ``length_of_study_time_step - time_step`` values.
    :return: ``np.ndarray`` of float64 class labels (0.0-4.0; float so callers
        can NaN individual entries downstream without a dtype change).
    """
    t = pd.Series(np.asarray(t, dtype=np.float64))
    classes = pd.cut(t, bins=_RUL_CLASS_BIN_EDGES, labels=_RUL_CLASS_LABELS, right=True)
    return classes.astype(np.float64).to_numpy()


class ScaniaClassificationDataset(ScaniaBaseDataset):
    """5-class failure-urgency classification variant of :class:`ScaniaBaseDataset`."""

    LABEL_COL = CLASS_LABEL
    BOUND_COL = CLASS_UPPER_BOUND

    def _compute_labels(self) -> None:
        """Populate ``CLASS_LABEL``/``CLASS_UPPER_BOUND`` (see module docstring)."""
        df = self.df
        if CLASS_LABEL in df.columns:
            # --- Mode B: provided ground-truth labels (val/test) --------- #
            df[CLASS_LABEL] = df[CLASS_LABEL].astype(np.float64)
            df[CLASS_UPPER_BOUND] = df[CLASS_LABEL]
        else:
            # --- Mode A: TTE-derived labels (train) ----------------------- #
            t = df[LENGTH_OF_STUDY_TIME_STEP] - df[TIME_STEP]
            classes = rul_to_class(t)
            df[CLASS_UPPER_BOUND] = classes
            df[CLASS_LABEL] = classes.copy()
            df.loc[df[IS_CENSORED] == 1, CLASS_LABEL] = np.nan
        self.df = df
