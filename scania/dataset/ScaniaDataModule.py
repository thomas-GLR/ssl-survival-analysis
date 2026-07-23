"""
LightningDataModule for the Scania Component X dataset.

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
    6. build a ScaniaDataset per split; z-score params are fit on train only;
       the test dataset additionally uses only_final=True (one window per
       vehicle, mirroring CMAPSS); val/calib keep every window
    7. cache the processed splits so later runs skip preprocessing

The module exposes the standard ``train/val/test_dataloader`` (uncensored
``(x, y)`` batches) plus convenience accessors for the co-training (Coprog) and
self-supervised paradigms.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from constants.scania_component_x_columns import (
    VEHICLE_ID,
    TIME_STEP,
    LENGTH_OF_STUDY_TIME_STEP,
    IN_STUDY_REPAIR,
    COUNTER_COLUMNS,
    HISTOGRAM_COLUMNS,
    ZHIST_FEATURE_COLUMNS,
)
from scania.dataset.ScaniaDataset import ScaniaDataset, ZHistFeatureNormalizer, IS_CENSORED

READOUTS_FILE = "train_operational_readouts.csv"
TTE_FILE = "train_tte.csv"
MANIFEST_FILE = "manifest.json"
BASE_SPLITS = ("train", "val", "test")


class ScaniaDataModule(LightningDataModule):
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
        super().__init__()
        assert (
            0 <= val_rate < 1 and 0 <= test_rate < 1 and 0 <= calib_rate < 1
            and (val_rate + test_rate + calib_rate) < 1
        ), "val_rate/test_rate/calib_rate must be in [0, 1) and sum to < 1"
        assert n_quantile_length_strata >= 1, "n_quantile_length_strata must be >= 1"
        assert counter_mode in ("delta", "cumulative", "both"), \
            f"Unsupported counter_mode: {counter_mode}"
        assert 0 < data_fraction <= 1.0, "data_fraction must be in (0, 1]"
        assert histogram_mode in ("sum", "zhist"), \
            f"Unsupported histogram_mode: {histogram_mode}"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.seed = seed
        self.data_fraction = data_fraction
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.calib_rate = calib_rate
        self.stratify = stratify
        self.n_quantile_length_strata = n_quantile_length_strata
        self._splits: tuple[str, ...] = BASE_SPLITS + (("calib",) if self.calib_rate > 0 else ())
        self.norm_type = norm_type
        self.shuffle_loader = shuffle_loader
        self.cache_dir = cache_dir or os.path.join(data_dir, "scania_cache")
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.return_sequence_label = return_sequence_label
        self.counter_mode = counter_mode
        self.include_histograms = include_histograms
        self.histogram_mode = histogram_mode

        # Raw counter columns as they appear in the CSV (what we read/difference).
        self._base_counter_cols = list(COUNTER_COLUMNS)
        # Raw histogram bin columns to READ / NaN-fill from the CSV (a
        # distribution per feature; never differenced by counter_mode).
        self._raw_histogram_cols = list(HISTOGRAM_COLUMNS) if include_histograms else []
        # Histogram FEATURE columns fed to the model. In "sum" mode these are the
        # raw per-bin columns (sum-normalized by HistogramFeatureNormalizer); in
        # "zhist" mode each group is collapsed to one continuous zhist_<group>
        # feature (ZHistFeatureNormalizer), computed from the raw bins inside
        # ScaniaDataset.
        if not include_histograms:
            self._histogram_cols = []
        elif histogram_mode == "zhist":
            self._histogram_cols = list(ZHIST_FEATURE_COLUMNS)
        else:
            self._histogram_cols = list(HISTOGRAM_COLUMNS)
        # Feature columns actually fed to the model. In "both" mode the per-step
        # deltas are appended as separate "<counter>_delta" columns, doubling the
        # counter feature count; "delta"/"cumulative" keep the base columns (same
        # names, different values). Histogram feature columns, if enabled, are
        # appended last so feature ordering stays stable.
        if counter_mode == "both":
            counter_feature_cols = self._base_counter_cols + [f"{c}_delta" for c in self._base_counter_cols]
        else:
            counter_feature_cols = list(self._base_counter_cols)
        self.feature_cols = counter_feature_cols + self._histogram_cols

        self.train_set: ScaniaDataset | None = None
        self.val_set: ScaniaDataset | None = None
        self.test_set: ScaniaDataset | None = None
        self.calib_set: ScaniaDataset | None = None
        self.norm_params: np.ndarray | None = None
        self.hist_norm_params: dict[str, float] | None = None
        self.zhist_norm_params: dict | None = None

    @property
    def feature_num(self) -> int:
        return len(self.feature_cols)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:
            return  # already set up

        if self._cache_is_valid():
            print(f"[Scania] Loading preprocessed data from cache: {self.cache_dir}")
            self._load_from_cache()
        else:
            print("[Scania] No valid cache found, preprocessing from raw files...")
            self._preprocess_and_split()
            self._save_cache()

    def _dataset_kwargs(self) -> dict:
        return {
            "sequence_len": self.sequence_len,
            "feature_cols": self.feature_cols,
            "histogram_cols": self._histogram_cols,
            "histogram_mode": self.histogram_mode,
            "raw_histogram_cols": self._raw_histogram_cols,
            "return_sequence_label": self.return_sequence_label,
            "seed": self.seed,
        }

    def _preprocess_and_split(self) -> None:
        start = time.time()

        # Read the raw counter columns plus the histogram columns when enabled;
        # "both" mode's "_delta" feature columns are derived below, they do not
        # exist in the CSV.
        base_cols = self._base_counter_cols
        raw_cols = base_cols + self._raw_histogram_cols
        usecols = [VEHICLE_ID, TIME_STEP] + raw_cols
        readouts = pd.read_csv(os.path.join(self.data_dir, READOUTS_FILE), usecols=usecols)
        tte = pd.read_csv(
            os.path.join(self.data_dir, TTE_FILE),
            usecols=[VEHICLE_ID, LENGTH_OF_STUDY_TIME_STEP, IN_STUDY_REPAIR],
        )

        readouts = readouts.sort_values([VEHICLE_ID, TIME_STEP]).reset_index(drop=True)

        # 2. per-vehicle NaN fill of the raw cumulative counters and histograms
        readouts[raw_cols] = readouts.groupby(VEHICLE_ID)[raw_cols].ffill()
        readouts[raw_cols] = readouts.groupby(VEHICLE_ID)[raw_cols].bfill()
        # Some vehicles never report a given column at any timestep (the whole
        # vehicle-column is NaN, so ffill/bfill cannot fill it). This is common
        # for the histogram bins (e.g. the 167_* group) and would otherwise leak
        # NaN through the normalizers into the feature windows, making the
        # training loss NaN. "No reading" means zero counts for both cumulative
        # counters and histogram bins, so fill the residual with 0.
        readouts[raw_cols] = readouts[raw_cols].fillna(0.0)

        # 3. build the feature representation according to counter_mode.
        #    The raw counters are cumulative; the *cumulative* level is the
        #    monotonic aging signal most predictive of RUL. diff() yields NaN
        #    for the first row of each vehicle -> set to 0.
        if self.counter_mode == "cumulative":
            # Keep the cumulative counters as-is (no differencing).
            pass
        elif self.counter_mode == "delta":
            # Replace counters by their per-step delta (legacy behavior).
            readouts[base_cols] = readouts.groupby(VEHICLE_ID)[base_cols].diff().fillna(0.0)
        elif self.counter_mode == "both":
            # Keep the cumulative counters AND append the per-step deltas as new columns.
            delta_cols = [f"{c}_delta" for c in base_cols]
            readouts[delta_cols] = readouts.groupby(VEHICLE_ID)[base_cols].diff().fillna(0.0)

        # 4. merge TTE and derive the censoring flag
        readouts = readouts.merge(tte, on=VEHICLE_ID, how="inner")
        readouts[IS_CENSORED] = (readouts[IN_STUDY_REPAIR] == 0).astype(int)
        readouts = readouts.drop(columns=[IN_STUDY_REPAIR])

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
        if self.data_fraction < 1.0:
            kept_ids: set = set()
            for _, group in vehicule_status.groupby(IS_CENSORED):
                ids = rng.permutation(group[VEHICLE_ID].to_numpy())
                n_keep = max(1, round(len(ids) * self.data_fraction))
                kept_ids.update(ids[:n_keep].tolist())
            readouts = readouts[readouts[VEHICLE_ID].isin(kept_ids)]
            vehicule_status = vehicule_status[vehicule_status[VEHICLE_ID].isin(kept_ids)]

        if self.stratify:
            # Fixed group order (is_censored 0/1, then "len1" before ascending length-
            # quantile bucket) keeps the sequential RNG deterministic.
            strata = self._length_stratify_groups(vehicule_status, readouts)
        else:
            strata = [vehicule_status[VEHICLE_ID].to_numpy()]

        test_ids: set = set()
        val_ids: set = set()
        calib_ids: set = set()
        for ids in strata:
            ids = rng.permutation(ids)
            id_number = len(ids)
            id_number_test = int(self.test_rate * id_number)
            id_number_val = int(self.val_rate * id_number)
            id_number_calib = int(self.calib_rate * id_number)
            test_ids.update(ids[:id_number_test].tolist())
            val_ids.update(ids[id_number_test:id_number_test + id_number_val].tolist())
            calib_ids.update(
                ids[id_number_test + id_number_val:
                    id_number_test + id_number_val + id_number_calib].tolist())

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
        #    Test additionally uses only_final=True (only the last window per
        #    vehicle kept), mirroring CMAPSS's use_only_final_on_test. calib (like
        #    val) keeps every window -- conformal calibration needs a large residual
        #    pool, not a single window per vehicle.
        self.train_set = ScaniaDataset(train_df, norm_type=self.norm_type, norm_params=None,
                                       hist_norm_params=None, zhist_norm_params=None,
                                       **self._dataset_kwargs())
        self.norm_params = self.train_set.norm_params
        self.hist_norm_params = self.train_set.hist_norm_params
        self.zhist_norm_params = self.train_set.zhist_norm_params
        self.val_set = ScaniaDataset(val_df, norm_type=self.norm_type, norm_params=self.norm_params,
                                     hist_norm_params=self.hist_norm_params,
                                     zhist_norm_params=self.zhist_norm_params, **self._dataset_kwargs())
        self.test_set = ScaniaDataset(test_df, norm_type=self.norm_type, norm_params=self.norm_params,
                                      hist_norm_params=self.hist_norm_params,
                                      zhist_norm_params=self.zhist_norm_params, only_final=True,
                                      **self._dataset_kwargs())
        self.calib_set = None
        if calib_df is not None:
            self.calib_set = ScaniaDataset(calib_df, norm_type=self.norm_type, norm_params=self.norm_params,
                                           hist_norm_params=self.hist_norm_params,
                                           zhist_norm_params=self.zhist_norm_params, **self._dataset_kwargs())

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
        edge-padded path in ``ScaniaDataset._gen_sequence`` and truncating
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
    # Caching
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
            "cache_version": 2,
        }

    def _cache_columns(self) -> list[str]:
        return [VEHICLE_ID, TIME_STEP] + self.feature_cols + [LENGTH_OF_STUDY_TIME_STEP, IS_CENSORED]

    @staticmethod
    def _vehicle_censor_counts(ds: ScaniaDataset) -> dict:
        """Vehicle-level failure/censored counts for a built dataset (for the
        manifest / verification). is_censored is constant within a vehicle."""
        vids, first_idx = np.unique(ds.id_array, return_index=True)
        cens = ds.is_censored_array[first_idx]
        return {
            "failure": int((cens == 0).sum()),
            "censored": int((cens == 1).sum()),
        }

    def _cache_is_valid(self) -> bool:
        manifest_path = os.path.join(self.cache_dir, MANIFEST_FILE)
        if not os.path.exists(manifest_path):
            return False
        if not all(os.path.exists(os.path.join(self.cache_dir, f"{s}.csv")) for s in self._splits):
            return False
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return manifest.get("config") == self._cache_config()

    def _save_cache(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        cols = self._cache_columns()
        sizes = {}
        vehicle_counts = {}
        split_datasets = {"train": self.train_set, "val": self.val_set, "test": self.test_set}
        if self.calib_set is not None:
            split_datasets["calib"] = self.calib_set
        for name in self._splits:
            dataset = split_datasets[name]
            # ds.df holds the normalized features + the columns count_rul needs.
            dataset.df[cols].to_csv(os.path.join(self.cache_dir, f"{name}.csv"), index=False)
            sizes[name] = int(len(dataset))
            vehicle_counts[name] = self._vehicle_censor_counts(dataset)

        manifest = {
            "config": self._cache_config(),
            "norm_params": self.norm_params.tolist() if self.norm_params is not None else None,
            "hist_norm_params": self.hist_norm_params,
            "zhist_norm_params": (
                ZHistFeatureNormalizer.params_to_json(self.zhist_norm_params)
                if self.zhist_norm_params is not None else None
            ),
            "feature_cols": self.feature_cols,
            "window_counts": sizes,
            "vehicle_counts": vehicle_counts,
        }
        with open(os.path.join(self.cache_dir, MANIFEST_FILE), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[Scania] Cache written to {self.cache_dir}")

    def _load_from_cache(self) -> None:
        manifest_path = os.path.join(self.cache_dir, MANIFEST_FILE)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("norm_params") is not None:
            self.norm_params = np.asarray(manifest["norm_params"], dtype=np.float64)
        self.hist_norm_params = manifest.get("hist_norm_params")
        if manifest.get("zhist_norm_params") is not None:
            self.zhist_norm_params = ZHistFeatureNormalizer.params_from_json(manifest["zhist_norm_params"])

        sets = {}
        for name in self._splits:
            df = pd.read_csv(os.path.join(self.cache_dir, f"{name}.csv"))
            # Features are already normalized in the cache -> norm_type=None.
            # only_final mirrors _preprocess_and_split: test only.
            sets[name] = ScaniaDataset(
                df, norm_type=None, norm_params=None,
                only_final=(name == "test"),
                **self._dataset_kwargs(),
            )

        self.train_set, self.val_set, self.test_set = sets["train"], sets["val"], sets["test"]
        self.calib_set = sets.get("calib")
        if self.norm_params is not None:
            self.train_set.norm_params = self.norm_params

    # ------------------------------------------------------------------ #
    # Standard Lightning dataloaders (supervised, uncensored only)
    # ------------------------------------------------------------------ #
    def _loader(self, ds: ScaniaDataset, shuffle: bool) -> DataLoader:
        return ds.get_data_loader_without_censored_data(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=self.shuffle_loader)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)

    # ------------------------------------------------------------------ #
    # Convenience accessors for the other paradigms
    # ------------------------------------------------------------------ #
    def get_full_dataset(self, split: str = "train") -> ScaniaDataset:
        """Return the underlying ScaniaDataset for a split (self-supervised path,
        which needs censored + uncensored together via the is_censored flag)."""
        return self._get_set(split)

    def get_cotraining_tensors(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Co-training (Coprog) path: (feat_uncensored, target_uncensored,
        feat_censored, ids_censored) for the requested split."""
        return self._get_set(split).get_censored_split_tensors()

    def get_censored_lower_bounds(self, split: str = "train"):
        """(feat_censored, ids_censored, lower_bounds_censored) for the split."""
        return self._get_set(split).get_censored_lower_bounds()

    def _get_set(self, split: str) -> ScaniaDataset:
        if self.train_set is None:
            self.setup()
        mapping = {"train": self.train_set, "val": self.val_set, "test": self.test_set}
        if self.calib_set is not None:
            mapping["calib"] = self.calib_set
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}', expected one of {list(mapping)}")
        return mapping[split]
