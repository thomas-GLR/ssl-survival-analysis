"""
LightningDataModule for the Scania Component X regression pipeline.

Uses only the training files (``train_operational_readouts.csv`` +
``train_tte.csv``) and produces train/val/test splits *by vehicle* out of them
(the standalone ``validation_*`` / ``test_*`` files and ``train_specifications``
are intentionally ignored).

Pipeline (see the project plan for the rationale):
    1. read the two train files, keep vehicle_id + time_step + counter columns
    2. per-vehicle NaN fill (ffill then bfill) of the raw cumulative counters
    3. per-vehicle differencing counter -> per-step delta (first row = 0)
    4. merge time-to-event info, derive is_censored (in_study_repair == 0)
    4b. optionally subsample a fraction of vehicles (data_fraction < 1.0),
        stratified by is_censored so the censored/uncensored ratio is
        preserved regardless of the `stratify` split setting
    5. split vehicles into train/val/test/(optional calib) -- all rows of a
       vehicle stay together. Stratified (when `stratify=True`) by the cross
       of is_censored and a raw-length quantile bucket (vehicles with exactly
       1 readout get their own bucket, since they can't be quantile-split).
       `calib_rate` is an opt-in sibling of `val_rate`/`test_rate`: a fully
       separate held-out set for conformal calibration (crepes), decoupled
       from the early-stopping validation set.
    5b. truncate a random per-vehicle number of trailing readouts of
        uncensored (failure) vehicles in val/test/calib only, so the RUL
        target near end-of-life isn't trivially ~0 (train is never
        truncated; censored vehicles are never truncated)
    6. build a ScaniaRegressionDataset per split; z-score params are fit on train only;
       the val and test datasets additionally use only_final=True (one window
       per vehicle, mirroring CMAPSS), so early stopping is measured on the
       same kind of set as the final evaluation; train/calib keep every window
    7. cache the processed splits so later runs skip preprocessing

The module exposes the standard ``train/val/test_dataloader`` (uncensored
``(x, y)`` batches) plus convenience accessors for the co-training (Coprog) and
self-supervised paradigms. Shared plumbing (raw-CSV reading, TTE merge,
caching, dataloaders, generic accessors) lives in ``ScaniaBaseDataModule``.
"""

import os
import time

import numpy as np
import pandas as pd

from constants.scania_component_x_columns import VEHICLE_ID, TIME_STEP, LENGTH_OF_STUDY_TIME_STEP
from scania.dataset.ScaniaBaseDataModule import ScaniaBaseDataModule, READOUTS_FILE
from scania.dataset.ScaniaBaseDataset import IS_CENSORED
from scania.dataset.ScaniaRegressionDataset import ScaniaRegressionDataset


class ScaniaRegressionDataModule(ScaniaBaseDataModule):
    DATASET_CLASS = ScaniaRegressionDataset
    DEFAULT_CACHE_SUBDIR = "scania_cache_regression"

    def __init__(
            self,
            data_dir: str,
            batch_size: int | None,
            sequence_len: int,
            seed: int | None = None,
            data_fraction: float = 1.0,
            val_rate: float = 0.2,
            test_rate: float = 0.1,
            calib_rate: float = 0.0,
            stratify: bool = True,
            n_quantile_length_strata: int = 4,
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
        assert (
            0 <= val_rate < 1 and 0 <= test_rate < 1 and 0 <= calib_rate < 1
            and (val_rate + test_rate + calib_rate) < 1
        ), "val_rate/test_rate/calib_rate must be in [0, 1) and sum to < 1"
        assert n_quantile_length_strata >= 1, "n_quantile_length_strata must be >= 1"
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
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.stratify = stratify
        self.n_quantile_length_strata = n_quantile_length_strata

    def _preprocess_and_split(self) -> None:
        start = time.time()

        readouts = self._read_and_transform_readouts(os.path.join(self.data_dir, READOUTS_FILE))
        readouts = self._merge_tte_and_censor(readouts)

        # 5. split vehicles into train/val/test (all rows of a vehicle together),
        #    stratified by censoring status so the failure/censored proportion is
        #    the same in every split. Failures are rare (~2272 / 23550 vehicles),
        #    so a plain random split could leave val/test with very few failures.
        rng = np.random.default_rng(self.seed)

        # Vehicle-level censoring status (constant within a vehicle).
        vehicule_status = readouts[[VEHICLE_ID, IS_CENSORED]].drop_duplicates(VEHICLE_ID)

        # 4b. Optionally subsample a fraction of vehicles to shrink the dataset
        #     (e.g. for faster iteration on limited compute). Always stratified
        #     by is_censored -- independent of `self.stratify`, which only
        #     controls the train/val/test split below -- so the censored/
        #     uncensored ratio is preserved. Consumes rng draws before the
        #     split permutations and _truncate_uncensored_tail below.
        readouts, vehicule_status = self._apply_data_fraction(readouts, vehicule_status, rng)

        if self.stratify:
            # Fixed group order (is_censored 0/1, then "len1" before ascending length-
            # quantile bucket) keeps the sequential RNG deterministic.
            strata = self._length_stratify_groups(vehicule_status, readouts)
        else:
            strata = [vehicule_status[VEHICLE_ID].to_numpy()]

        id_sets = self._split_ids_by_rate(
            strata, {"test": self.test_rate, "val": self.val_rate, "calib": self.calib_rate}, rng,
        )
        test_ids, val_ids, calib_ids = id_sets["test"], id_sets["val"], id_sets["calib"]

        vehicules_ids = readouts[VEHICLE_ID]
        test_df = readouts[vehicules_ids.isin(test_ids)]
        val_df = readouts[vehicules_ids.isin(val_ids)]
        calib_df = readouts[vehicules_ids.isin(calib_ids)] if "calib" in self._splits else None
        train_df = readouts[~vehicules_ids.isin(test_ids | val_ids | calib_ids)]

        # 5b. Val/test/calib: truncate a random tail of trailing readouts per
        #     uncensored vehicle so the final kept window's RUL isn't
        #     trivially ~0 (see _truncate_uncensored_tail). Train is never
        #     truncated. Consumes further draws from the same `rng` used
        #     above for the vehicle split. calib is held out the same way
        #     val/test are (it feeds conformal calibration against known
        #     outcomes), so it gets the same treatment; it draws no extra
        #     rng entropy when disabled (calib_df is None).
        test_df = self._truncate_uncensored_tail(test_df, rng)
        val_df = self._truncate_uncensored_tail(val_df, rng)
        if calib_df is not None:
            calib_df = self._truncate_uncensored_tail(calib_df, rng)

        # 6. build datasets; z-score params fit on train, reused for val/test/calib.
        #    Val and test additionally use only_final=True (only the last window
        #    per vehicle kept), mirroring CMAPSS's use_only_final_on_test, so the
        #    early-stopping metric is computed on the same kind of set as the
        #    final evaluation. calib keeps every window -- conformal calibration
        #    needs a large residual pool, not a single window per vehicle.
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
            test_df, norm_type=self.norm_type,
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
    # Length-quantile x is_censored stratification (train/val/test/calib split)
    # ------------------------------------------------------------------ #
    def _length_stratify_groups(
            self, vehicule_status: pd.DataFrame, readouts: pd.DataFrame,
    ) -> list[np.ndarray]:
        """Vehicle-id strata crossing IS_CENSORED with a raw-length quantile bucket.

        Vehicles with exactly 1 raw readout row form their own "len1" stratum
        (they cannot be meaningfully quantile-split, and length == 1 is common
        enough in a censored survival dataset that including them in
        ``pd.qcut`` risks a degenerate all-identical-value bucket). Remaining
        length > 1 vehicles are split into ``self.n_quantile_length_strata``
        quantile buckets computed independently *within* each IS_CENSORED
        group -- censored and failure vehicles have different length
        distributions in a run-to-failure dataset, so nested binning keeps
        every (is_censored, length_bucket) cross-stratum representative of its
        own subpopulation instead of being dominated by whichever IS_CENSORED
        group is larger.

        ``pd.qcut(..., duplicates="drop")`` avoids raising on repeated bin
        edges (skewed/duplicated lengths); the one residual case it does not
        fully resolve -- returning all-NaN labels when every remaining length
        value in a group is identical -- is folded into bucket 0 via
        ``fillna(0)``.

        :param vehicule_status: One row per vehicle with VEHICLE_ID + IS_CENSORED
            (already filtered by data_fraction subsampling, if any).
        :param readouts: The (already subsampled) full readouts dataframe, used
            only to compute per-vehicle raw row counts.
        :return: List of 1-D vehicle-id numpy arrays, one per non-empty
            (is_censored, length_bucket) cross-stratum, in a fixed deterministic
            order (is_censored 0 then 1; within each, "len1" first, then
            ascending length-quantile bucket index).
        """
        lengths = readouts.groupby(VEHICLE_ID).size().rename("_length")
        vehicule_status = vehicule_status.merge(lengths, on=VEHICLE_ID, how="left")

        strata: list[np.ndarray] = []
        for _, group in vehicule_status.groupby(IS_CENSORED):  # fixed order: 0 then 1
            len1_mask = group["_length"] == 1
            len1_ids = group.loc[len1_mask, VEHICLE_ID].to_numpy()
            if len(len1_ids) > 0:
                strata.append(len1_ids)

            rest = group.loc[~len1_mask]
            if len(rest) == 0:
                continue
            bucket = pd.qcut(
                rest["_length"], q=self.n_quantile_length_strata, labels=False, duplicates="drop")
            bucket = bucket.fillna(0).astype(int)
            for bucket_id in sorted(bucket.unique()):
                strata.append(rest.loc[bucket == bucket_id, VEHICLE_ID].to_numpy())
        return strata

    # ------------------------------------------------------------------ #
    # Val/test tail truncation (avoid a trivially-~0 end-of-life RUL)
    # ------------------------------------------------------------------ #
    def _truncate_uncensored_tail(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Randomly drop trailing readouts from uncensored (failure) vehicles.

        Scania has no NASA-style withheld pre-failure tail like CMAPSS's test
        files, so a failure vehicle's last row IS its failure point and RUL is
        trivially ~0 there. This reproduces CMAPSS's effective end-of-life
        truncation: for each uncensored vehicle with more than
        ``sequence_len`` rows, drop a random number of trailing rows (drawn
        from ``rng``, so it is reproducible via ``self.seed`` without any new
        cache-config key -- ``rng`` is the same generator already used for the
        (optional) data_fraction subsampling and the train/val/test vehicle
        split). The vehicle keeps between
        ``sequence_len`` and its original row count, so it still yields a
        full window afterward.

        Censored vehicles and vehicles with <= ``sequence_len`` rows are left
        untouched: censored vehicles are already naturally truncated at real
        censoring time and their RUL target is NaN (only ``rul_lower_bound``
        is used for them), so truncating them further would only destroy
        signal without adding realism. Short vehicles already take the
        edge-padded path in ``ScaniaBaseDataset._gen_sequence`` and truncating
        them would push them below ``sequence_len``, breaking that invariant.

        :param df: readouts of a single split (val or test).
        :param rng: shared ``np.random.Generator``, consumed sequentially
            after the vehicle-split draws in ``_preprocess_and_split``.
        :return: a new dataframe with 0..(count - sequence_len) trailing rows
            removed per eligible uncensored vehicle.
        """
        seq_len = self.sequence_len
        df = df.sort_values([VEHICLE_ID, TIME_STEP]).reset_index(drop=True)
        rows_number = len(df)
        if rows_number == 0:
            return df

        vehicle_ids = df[VEHICLE_ID].to_numpy()
        censored = df[IS_CENSORED].to_numpy()

        starts = np.concatenate(([0], np.flatnonzero(np.diff(vehicle_ids) != 0) + 1))
        counts = np.diff(np.concatenate((starts, [rows_number])))
        censored_per_vehicle = censored[starts]

        k = np.zeros(len(counts), dtype=np.int64)
        eligible = (censored_per_vehicle == 0) & (counts > seq_len)
        if eligible.any():
            high = counts[eligible] - seq_len + 1  # exclusive upper bound
            k[eligible] = rng.integers(0, high)

        keep_len = counts - k
        row_pos_in_vehicle = np.arange(rows_number) - np.repeat(starts, counts)
        keep_mask = row_pos_in_vehicle < np.repeat(keep_len, counts)

        return df.loc[keep_mask].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Caching hooks
    # ------------------------------------------------------------------ #
    def _cache_config(self) -> dict:
        """Params that change the cached CSV content (invalidate the cache).

        ``sequence_len`` IS included here (unlike before this affected only
        windowing): the val/test truncation formula in
        ``_truncate_uncensored_tail`` bounds how many trailing rows are kept
        per vehicle relative to ``sequence_len``, so the cached row set
        itself now depends on it.

        ``cache_version`` is a generic marker bumped whenever the split
        algorithm (or any other cache-affecting logic not captured by a
        dedicated key below) changes -- ``_cache_is_valid`` only checks
        equality of this dict, so an algorithm change that doesn't touch any
        existing key/value would otherwise silently reuse a stale cache
        produced by the old code.
        """
        return {
            "feature_cols": self.feature_cols,
            "norm_type": self.norm_type,
            "val_rate": self.val_rate,
            "test_rate": self.test_rate,
            "calib_rate": self.calib_rate,
            "stratify": self.stratify,
            "n_quantile_length_strata": self.n_quantile_length_strata,
            "seed": self.seed,
            "sequence_len": self.sequence_len,
            "counter_mode": self.counter_mode,
            "data_fraction": self.data_fraction,
            "include_histograms": self.include_histograms,
            "histogram_mode": self.histogram_mode,
            "cache_version": 3,
        }

    def _cache_columns(self, split: str) -> list[str]:
        return [VEHICLE_ID, TIME_STEP] + self.feature_cols + [LENGTH_OF_STUDY_TIME_STEP, IS_CENSORED]

    def _only_final_for_cached_split(self, split: str) -> bool:
        return split in ("val", "test")
