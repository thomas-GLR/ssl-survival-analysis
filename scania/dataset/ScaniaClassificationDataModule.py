"""
LightningDataModule for the Scania Component X classification pipeline.

Unlike the regression module, val/test are **not** carved out of the train
file -- they come from the fixed ``validation_operational_readouts.csv`` +
``validation_labels.csv`` and ``test_operational_readouts.csv`` +
``test_labels.csv`` files, which already carry ground-truth ``class_label``
(and contain only failure/event vehicles -- no censoring). The calibration
set, when enabled, is instead carved out of the train pool (same
``train_operational_readouts.csv`` + ``train_tte.csv`` regression draws its
train/val/test/calib from):

    1. build the train pool exactly like the regression module does (read,
       NaN-fill, counter-mode transform, merge TTE, derive is_censored) --
       see ``ScaniaBaseDataModule._read_and_transform_readouts`` /
       ``_merge_tte_and_censor``.
    2. draw a calibration set from the pool's *uncensored* vehicles only
       (censored vehicles have no known class_label to calibrate against,
       so they always stay in train), stratified by each vehicle's final
       (last-readout) class label so calib mirrors the overall
       uncensored-train class distribution. No length-quantile
       stratification (unlike regression's calib/val/test split).
    3. optionally subsample a fraction of the *remaining* (train) vehicles
       (data_fraction < 1.0), stratified by is_censored. Unlike the
       regression module, data_fraction shrinks the train split only: calib
       is carved out first and val/test come from separate fixed files, so
       all three evaluation-side splits keep their full size whatever
       data_fraction is set to.
    4. no tail truncation anywhere (train, calib, val, test) -- val/test's
       last readout already reflects the label file's evaluation point.
    5. build a ScaniaClassificationDataset per split; z-score params are fit
       on train only. val/test use only_final=True (one window per vehicle,
       matching their one-row-per-vehicle label file); train/calib keep
       every window.
    6. cache the processed splits (own cache subdirectory, separate from the
       regression module's) so later runs skip preprocessing.

Shared plumbing lives in ``ScaniaBaseDataModule``.
"""

import os
import time

import numpy as np
import pandas as pd

from constants.scania_component_x_columns import (
    VEHICLE_ID,
    TIME_STEP,
    LENGTH_OF_STUDY_TIME_STEP,
    CLASS_LABEL,
)
from scania.dataset.ScaniaBaseDataModule import ScaniaBaseDataModule, READOUTS_FILE
from scania.dataset.ScaniaBaseDataset import IS_CENSORED
from scania.dataset.ScaniaClassificationDataset import ScaniaClassificationDataset, rul_to_class

VALIDATION_READOUTS_FILE = "validation_operational_readouts.csv"
VALIDATION_LABELS_FILE = "validation_labels.csv"
TEST_READOUTS_FILE = "test_operational_readouts.csv"
TEST_LABELS_FILE = "test_labels.csv"


class ScaniaClassificationDataModule(ScaniaBaseDataModule):
    DATASET_CLASS = ScaniaClassificationDataset
    DEFAULT_CACHE_SUBDIR = "scania_cache_classification"

    def __init__(
            self,
            data_dir: str,
            batch_size: int | None,
            sequence_len: int,
            seed: int | None = None,
            data_fraction: float = 1.0,
            calib_rate: float = 0.0,
            norm_type: str | None = "z-score",
            shuffle_loader: bool = True,
            cache_dir: str | None = None,
            num_workers: int = 0,
            pin_memory: bool = False,
            return_sequence_label: bool = False,
            counter_mode: str = "cumulative",
            include_histograms: bool = False,
            histogram_mode: str = "sum",
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            sequence_len=sequence_len,
            seed=seed,
            data_fraction=data_fraction,
            calib_rate=calib_rate,
            norm_type=norm_type,
            shuffle_loader=shuffle_loader,
            cache_dir=cache_dir,
            num_workers=num_workers,
            pin_memory=pin_memory,
            return_sequence_label=return_sequence_label,
            counter_mode=counter_mode,
            include_histograms=include_histograms,
            histogram_mode=histogram_mode,
        )

    def _preprocess_and_split(self) -> None:
        start = time.time()

        # --- train pool: same source files/steps as regression's pool ----- #
        readouts = self._read_and_transform_readouts(os.path.join(self.data_dir, READOUTS_FILE))
        readouts = self._merge_tte_and_censor(readouts)

        rng = np.random.default_rng(self.seed)
        vehicle_status = readouts[[VEHICLE_ID, IS_CENSORED]].drop_duplicates(VEHICLE_ID)

        # 2. calib is carved out of the *full* pool, before data_fraction, so
        #    that shrinking the dataset leaves the evaluation-side splits
        #    untouched: val/test come from fixed files and are never
        #    subsampled, and calib must behave the same way (a conformal
        #    calibration set shrinking with data_fraction would widen the
        #    intervals for reasons unrelated to the model being calibrated).
        calib_ids: set = set()
        if "calib" in self._splits:
            calib_ids = self._draw_calib_ids(readouts, vehicle_status, rng)

        vids = readouts[VEHICLE_ID]
        calib_df = readouts[vids.isin(calib_ids)] if "calib" in self._splits else None
        train_df = readouts[~vids.isin(calib_ids)]

        # 3. data_fraction then shrinks the train split only.
        train_status = vehicle_status[~vehicle_status[VEHICLE_ID].isin(calib_ids)]
        train_df, _ = self._apply_data_fraction(train_df, train_status, rng)
        # No tail truncation anywhere in this module (train, calib, val, test).

        # --- val/test: fixed files, not carved out of the train pool ------ #
        val_df = self._read_labeled_split(VALIDATION_READOUTS_FILE, VALIDATION_LABELS_FILE)
        test_df = self._read_labeled_split(TEST_READOUTS_FILE, TEST_LABELS_FILE)

        # --- build datasets; norm fit on train, reused everywhere else ---- #
        self.train_set = self.DATASET_CLASS(
            train_df,
            norm_type=self.norm_type,
            norm_params=None,
            hist_norm_params=None,
            zhist_norm_params=None,
            **self._dataset_kwargs(),
        )
        self.norm_params = self.train_set.norm_params
        self.hist_norm_params = self.train_set.hist_norm_params
        self.zhist_norm_params = self.train_set.zhist_norm_params

        self.val_set = self.DATASET_CLASS(
            val_df,
            norm_type=self.norm_type,
            norm_params=self.norm_params,
            hist_norm_params=self.hist_norm_params,
            zhist_norm_params=self.zhist_norm_params,
            only_final=True,
            **self._dataset_kwargs(),
        )
        self.test_set = self.DATASET_CLASS(
            test_df,
            norm_type=self.norm_type,
            norm_params=self.norm_params,
            hist_norm_params=self.hist_norm_params,
            zhist_norm_params=self.zhist_norm_params,
            only_final=True,
            **self._dataset_kwargs(),
        )

        self.calib_set = None
        if calib_df is not None:
            self.calib_set = self.DATASET_CLASS(
                calib_df,
                norm_type=self.norm_type,
                norm_params=self.norm_params,
                hist_norm_params=self.hist_norm_params,
                zhist_norm_params=self.zhist_norm_params,
                **self._dataset_kwargs(),
            )

        split_names = "train/val/test" + ("/calib" if calib_df is not None else "")
        split_counts = [len(train_df[VEHICLE_ID].unique()), len(val_df[VEHICLE_ID].unique()),
                        len(test_df[VEHICLE_ID].unique())]
        if calib_df is not None:
            split_counts.append(len(calib_df[VEHICLE_ID].unique()))
        print(f"[Scania] Preprocessing done in {time.time() - start:.1f}s | "
              f"vehicles {split_names} = {'/'.join(str(c) for c in split_counts)}")

    # ------------------------------------------------------------------ #
    # Calibration carve-out (uncensored-only, stratified by final class)
    # ------------------------------------------------------------------ #
    def _draw_calib_ids(
            self, readouts: pd.DataFrame, vehicle_status: pd.DataFrame, rng: np.random.Generator,
    ) -> set:
        """Draw ``self.calib_rate`` of the uncensored train-pool vehicles into calib.

        Censored vehicles are excluded before stratifying, so they can never
        be drawn (they always stay in train). Each uncensored vehicle is
        assigned a stratum by its final-readout class (``rul_to_class`` of
        ``length_of_study_time_step - time_step`` at that vehicle's last row,
        the same quantity/binning ``ScaniaClassificationDataset``'s TTE-derived
        mode uses per-row), so calib mirrors the overall uncensored-train
        class distribution. No length-quantile stratification.

        Called on the *full* train pool, before any data_fraction
        subsampling, so the calibration set keeps its full size regardless
        of ``self.data_fraction`` (see the module docstring, step 2).

        :param readouts: full train-pool readouts, sorted by
            (vehicle_id, time_step).
        :param vehicle_status: one row per vehicle with VEHICLE_ID + IS_CENSORED
            (same vehicles as ``readouts``).
        :param rng: shared random generator (consumes one permutation per
            class stratum).
        :return: set of vehicle ids drawn into calib.
        """
        uncensored_ids = vehicle_status.loc[vehicle_status[IS_CENSORED] == 0, VEHICLE_ID]
        pool = readouts[readouts[VEHICLE_ID].isin(uncensored_ids)]

        last_rows = pool.groupby(VEHICLE_ID, as_index=False).tail(1)
        t_final = last_rows[LENGTH_OF_STUDY_TIME_STEP] - last_rows[TIME_STEP]
        final_class = pd.Series(rul_to_class(t_final), index=last_rows[VEHICLE_ID].to_numpy())

        strata = [
            final_class.index[final_class == c].to_numpy()
            for c in sorted(final_class.dropna().unique())
        ]
        id_sets = self._split_ids_by_rate(strata, {"calib": self.calib_rate}, rng)
        return id_sets["calib"]

    # ------------------------------------------------------------------ #
    # Val/test: fixed, pre-labeled files (no TTE, no censoring)
    # ------------------------------------------------------------------ #
    def _read_labeled_split(self, readouts_file: str, labels_file: str) -> pd.DataFrame:
        """Read one validation/test readouts CSV and merge its ground-truth labels.

        No TTE file exists for these splits, so ``is_censored`` is set to a
        constant 0 (they contain only failure/event vehicles) purely so
        ``ScaniaBaseDataset._gen_sequence`` has the column it unconditionally
        reads. The merged ``class_label`` column makes
        ``ScaniaClassificationDataset`` auto-detect its provided-label mode.

        :param readouts_file: file name (relative to ``self.data_dir``) of the
            operational-readouts CSV.
        :param labels_file: file name (relative to ``self.data_dir``) of the
            ``vehicle_id, class_label`` labels CSV.
        :return: the merged dataframe.
        """
        df = self._read_and_transform_readouts(os.path.join(self.data_dir, readouts_file))
        labels = pd.read_csv(os.path.join(self.data_dir, labels_file), usecols=[VEHICLE_ID, CLASS_LABEL])
        df = df.merge(labels, on=VEHICLE_ID, how="inner")
        df[IS_CENSORED] = 0
        return df

    # ------------------------------------------------------------------ #
    # Caching hooks
    # ------------------------------------------------------------------ #
    def _cache_config(self) -> dict:
        """Params that change the cached CSV content (invalidate the cache).

        Unlike the regression module, ``sequence_len`` is deliberately
        omitted: nothing in this module's preprocessing depends on it (there
        is no truncation step), it only affects windowing at
        dataset-construction time (already covered by ``_dataset_kwargs()``).

        ``cache_version`` is bumped whenever the split algorithm changes in a
        way no other key captures -- v2: calib is now drawn before (instead of
        after) data_fraction subsampling, so both the calib/train membership
        and the rng draw order differ for the same config.
        """
        return {
            "feature_cols": self.feature_cols,
            "norm_type": self.norm_type,
            "calib_rate": self.calib_rate,
            "seed": self.seed,
            "counter_mode": self.counter_mode,
            "data_fraction": self.data_fraction,
            "include_histograms": self.include_histograms,
            "histogram_mode": self.histogram_mode,
            "cache_version": 2,
        }

    def _cache_columns(self, split: str) -> list[str]:
        if split in ("val", "test"):
            return [VEHICLE_ID, TIME_STEP] + self.feature_cols + [CLASS_LABEL, IS_CENSORED]
        return [VEHICLE_ID, TIME_STEP] + self.feature_cols + [LENGTH_OF_STUDY_TIME_STEP, IS_CENSORED]

    def _only_final_for_cached_split(self, split: str) -> bool:
        return split in ("val", "test")
