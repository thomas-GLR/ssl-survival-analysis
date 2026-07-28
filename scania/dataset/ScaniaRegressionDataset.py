"""
Scania Component X regression dataset.

``ScaniaRegressionDataset`` predicts the Remaining Useful Life (RUL) of a
vehicle from its operational readouts, sharing all preprocessing (windowing,
normalization, censoring bookkeeping) with :class:`ScaniaBaseDataset`.

- Censoring is real (``in_study_repair == 0``), never synthetically generated.
- Censored rows have **no** RUL (label is NaN); a ``rul_lower_bound`` (the time
  observed until the end of the study) is kept instead, so a model can enforce
  "predicted RUL >= observed survival time".
- The test split is additionally built with ``only_final=True`` (set
  internally by ``ScaniaRegressionDataModule``, never exposed publicly), so
  evaluation sees exactly one window per vehicle -- its last, possibly
  truncated, readout -- instead of every sliding-window stride.
"""

import numpy as np

from constants.scania_component_x_columns import LENGTH_OF_STUDY_TIME_STEP, TIME_STEP
from scania.dataset.ScaniaBaseDataset import ScaniaBaseDataset, IS_CENSORED, ZHistFeatureNormalizer  # noqa: F401 (re-exports)

# Columns produced by this class / expected in the pre-processed frame.
RUL = "rul"
RUL_LOWER_BOUND = "rul_lower_bound"


class ScaniaRegressionDataset(ScaniaBaseDataset):
    """RUL regression variant of :class:`ScaniaBaseDataset`."""

    LABEL_COL = RUL
    BOUND_COL = RUL_LOWER_BOUND

    def _compute_labels(self) -> None:
        self.count_rul()

    def count_rul(self) -> None:
        """Compute the RUL target and its always-valid lower bound.

        Sets ``self.df[RUL_LOWER_BOUND]`` for every row (the time observed
        until the end of the study -- for uncensored vehicles this is the
        true RUL, for censored vehicles it is only a lower bound on it) and
        ``self.df[RUL]`` (the same value, NaN'd for censored rows since the
        true RUL is unknown there).
        """
        df = self.df

        # Time observed until the end of the study. For uncensored vehicles the
        # study ends at failure, so this is the true RUL. For censored vehicles
        # the true RUL is unknown but is at least this large (a lower bound).
        time_to_study_end = df[LENGTH_OF_STUDY_TIME_STEP] - df[TIME_STEP]

        df[RUL_LOWER_BOUND] = time_to_study_end
        df[RUL] = time_to_study_end.astype(np.float64)

        # Censored data has no known RUL -> NaN label (never used as a training
        # target; the uncensored-only loaders drop these rows).
        df.loc[df[IS_CENSORED] == 1, RUL] = np.nan

        self.df = df
