"""
Shared LightningDataModule base for the Scania Component X pipelines.

Holds everything with no regression/classification-specific behavior: common
constructor params, feature-column bookkeeping, the raw-CSV read/NaN-fill/
counter-mode transform step, the TTE-merge/censoring-derivation step, the
data_fraction vehicle subsampling step, the generic "permute each stratum
once then slice sequentially by named rate" split helper, the caching
skeleton, the standard dataloaders and the co-training/self-supervised
accessors.

Subclasses (``ScaniaRegressionDataModule``, ``ScaniaClassificationDataModule``)
must set two class attributes (``DATASET_CLASS``, ``DEFAULT_CACHE_SUBDIR``) and
implement four hook methods (``_preprocess_and_split``, ``_cache_config``,
``_cache_columns``, ``_only_final_for_cached_split``) -- see each hook's
docstring below for its exact contract.
"""

import json
import os
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
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
from scania.dataset.ScaniaBaseDataset import ScaniaBaseDataset, ZHistFeatureNormalizer, IS_CENSORED

READOUTS_FILE = "train_operational_readouts.csv"
TTE_FILE = "train_tte.csv"
MANIFEST_FILE = "manifest.json"
BASE_SPLITS = ("train", "val", "test")


class ScaniaBaseDataModule(LightningDataModule):
    """Shared Scania LightningDataModule base (see module docstring)."""

    DATASET_CLASS: ClassVar[type]
    DEFAULT_CACHE_SUBDIR: ClassVar[str]

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
        super().__init__()
        assert 0 <= calib_rate < 1, "calib_rate must be in [0, 1)"
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
        self.calib_rate = calib_rate
        self._splits: tuple[str, ...] = BASE_SPLITS + (("calib",) if self.calib_rate > 0 else ())
        self.norm_type = norm_type
        self.shuffle_loader = shuffle_loader
        self.cache_dir = cache_dir or os.path.join(data_dir, self.DEFAULT_CACHE_SUBDIR)
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
        # the dataset class.
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

        self.train_set: ScaniaBaseDataset | None = None
        self.val_set: ScaniaBaseDataset | None = None
        self.test_set: ScaniaBaseDataset | None = None
        self.calib_set: ScaniaBaseDataset | None = None
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

    # ------------------------------------------------------------------ #
    # Shared raw-data preprocessing steps
    # ------------------------------------------------------------------ #
    def _read_and_transform_readouts(self, csv_path: str) -> pd.DataFrame:
        """Read one operational-readouts CSV and apply the shared cleanup steps.

        Reads vehicle_id + time_step + the configured raw counter/histogram
        columns, sorts by (vehicle_id, time_step), per-vehicle NaN-fills them
        (ffill then bfill, residual filled with 0 -- "no reading" means zero
        counts) and applies ``self.counter_mode`` (cumulative/delta/both) to
        the counter columns.

        :param csv_path: path to an operational-readouts CSV (train,
            validation or test -- all three share this exact raw schema).
        :return: the cleaned, sorted dataframe.
        """
        base_cols = self._base_counter_cols
        raw_cols = base_cols + self._raw_histogram_cols
        usecols = [VEHICLE_ID, TIME_STEP] + raw_cols
        df = pd.read_csv(csv_path, usecols=usecols)
        df = df.sort_values([VEHICLE_ID, TIME_STEP]).reset_index(drop=True)

        df[raw_cols] = df.groupby(VEHICLE_ID)[raw_cols].ffill()
        df[raw_cols] = df.groupby(VEHICLE_ID)[raw_cols].bfill()
        df[raw_cols] = df[raw_cols].fillna(0.0)

        if self.counter_mode == "cumulative":
            pass  # keep the cumulative counters as-is (no differencing)
        elif self.counter_mode == "delta":
            df[base_cols] = df.groupby(VEHICLE_ID)[base_cols].diff().fillna(0.0)
        elif self.counter_mode == "both":
            delta_cols = [f"{c}_delta" for c in base_cols]
            df[delta_cols] = df.groupby(VEHICLE_ID)[base_cols].diff().fillna(0.0)

        return df

    def _merge_tte_and_censor(self, readouts: pd.DataFrame) -> pd.DataFrame:
        """Merge ``train_tte.csv`` onto ``readouts`` and derive ``is_censored``.

        :param readouts: readouts dataframe (vehicle_id column required).
        :return: ``readouts`` merged with ``length_of_study_time_step`` and
            ``is_censored`` (``in_study_repair == 0``); ``in_study_repair`` is
            dropped.
        """
        tte = pd.read_csv(
            os.path.join(self.data_dir, TTE_FILE),
            usecols=[VEHICLE_ID, LENGTH_OF_STUDY_TIME_STEP, IN_STUDY_REPAIR],
        )
        readouts = readouts.merge(tte, on=VEHICLE_ID, how="inner")
        readouts[IS_CENSORED] = (readouts[IN_STUDY_REPAIR] == 0).astype(int)
        return readouts.drop(columns=[IN_STUDY_REPAIR])

    def _apply_data_fraction(
            self, readouts: pd.DataFrame, vehicle_status: pd.DataFrame, rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Optionally subsample a fraction of vehicles, stratified by IS_CENSORED.

        No-op (returns the inputs unchanged) when ``self.data_fraction >= 1.0``.
        Otherwise draws ``self.data_fraction`` of the vehicles independently
        within each IS_CENSORED group, so the censored/uncensored ratio is
        preserved. Consumes one ``rng.permutation`` draw per IS_CENSORED group.

        :param readouts: full readouts dataframe.
        :param vehicle_status: one row per vehicle with VEHICLE_ID + IS_CENSORED.
        :param rng: shared random generator (consumes draws sequentially).
        :return: ``(readouts, vehicle_status)`` filtered to the kept vehicles.
        """
        if self.data_fraction >= 1.0:
            return readouts, vehicle_status
        kept_ids: set = set()
        for _, group in vehicle_status.groupby(IS_CENSORED):
            ids = rng.permutation(group[VEHICLE_ID].to_numpy())
            n_keep = max(1, round(len(ids) * self.data_fraction))
            kept_ids.update(ids[:n_keep].tolist())
        readouts = readouts[readouts[VEHICLE_ID].isin(kept_ids)]
        vehicle_status = vehicle_status[vehicle_status[VEHICLE_ID].isin(kept_ids)]
        return readouts, vehicle_status

    @staticmethod
    def _split_ids_by_rate(
            strata: list[np.ndarray], rates: dict[str, float], rng: np.random.Generator,
    ) -> dict[str, set]:
        """Permute each stratum once, then slice it sequentially by named rate.

        For each stratum, ``rng.permutation`` is drawn once; the permuted ids
        are then handed out to each named split in ``rates``' iteration order,
        ``int(rate * len(stratum))`` ids at a time. The remaining ids (after
        every named rate has taken its share) are not returned -- callers
        treat them as the implicit "not drawn into any named split" remainder.

        :param strata: list of 1-D vehicle-id arrays (one per stratum).
        :param rates: ordered mapping of split name -> fraction of each
            stratum to draw into that split.
        :param rng: shared random generator (consumes one permutation per
            stratum, in stratum order).
        :return: mapping of split name -> set of drawn vehicle ids.
        """
        id_sets: dict[str, set] = {name: set() for name in rates}
        for ids in strata:
            ids = rng.permutation(ids)
            n = len(ids)
            offset = 0
            for name, rate in rates.items():
                take = int(rate * n)
                id_sets[name].update(ids[offset:offset + take].tolist())
                offset += take
        return id_sets

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _preprocess_and_split(self) -> None:
        """Build every split from raw files and set them on self.

        Must set ``self.train_set``, ``self.val_set``, ``self.test_set``,
        ``self.calib_set`` (``None`` if ``"calib"`` is not in ``self._splits``)
        -- each an instance of ``self.DATASET_CLASS`` -- plus
        ``self.norm_params``, ``self.hist_norm_params``, ``self.zhist_norm_params``
        (fit on train, reused for the other splits). Should also print a
        one-line ``"[Scania] Preprocessing done in ..."`` summary.
        """
        raise NotImplementedError

    def _cache_config(self) -> dict:
        """Params that change the cached CSV content (invalidates the cache
        when they differ from what's on disk). Must include ``cache_version``."""
        raise NotImplementedError

    def _cache_columns(self, split: str) -> list[str]:
        """Columns to write to / read back from ``<split>.csv``."""
        raise NotImplementedError

    def _only_final_for_cached_split(self, split: str) -> bool:
        """``only_final`` passed to ``self.DATASET_CLASS(...)`` when rebuilding
        ``split``'s dataset from its cached CSV."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Caching
    # ------------------------------------------------------------------ #
    @staticmethod
    def _vehicle_censor_counts(ds: ScaniaBaseDataset) -> dict:
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
        sizes = {}
        vehicle_counts = {}
        split_datasets = {"train": self.train_set, "val": self.val_set, "test": self.test_set}
        if self.calib_set is not None:
            split_datasets["calib"] = self.calib_set
        for name in self._splits:
            dataset = split_datasets[name]
            cols = self._cache_columns(name)
            # ds.df holds the normalized features + the columns _compute_labels needs.
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
            sets[name] = self.DATASET_CLASS(
                df, norm_type=None, norm_params=None,
                only_final=self._only_final_for_cached_split(name),
                **self._dataset_kwargs(),
            )

        self.train_set, self.val_set, self.test_set = sets["train"], sets["val"], sets["test"]
        self.calib_set = sets.get("calib")
        if self.norm_params is not None:
            self.train_set.norm_params = self.norm_params

    # ------------------------------------------------------------------ #
    # Standard Lightning dataloaders (supervised, uncensored only)
    # ------------------------------------------------------------------ #
    def _loader(self, ds: ScaniaBaseDataset, shuffle: bool) -> DataLoader:
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
    def get_full_dataset(self, split: str = "train") -> ScaniaBaseDataset:
        """Return the underlying dataset for a split (self-supervised path,
        which needs censored + uncensored together via the is_censored flag)."""
        return self._get_set(split)

    def get_cotraining_tensors(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Co-training (Coprog) path: (feat_uncensored, target_uncensored,
        feat_censored, ids_censored) for the requested split."""
        return self._get_set(split).get_censored_split_tensors()

    def get_censored_lower_bounds(self, split: str = "train"):
        """(feat_censored, ids_censored, bounds_censored) for the split."""
        return self._get_set(split).get_censored_lower_bounds()

    def _get_set(self, split: str) -> ScaniaBaseDataset:
        if self.train_set is None:
            self.setup()
        mapping = {"train": self.train_set, "val": self.val_set, "test": self.test_set}
        if self.calib_set is not None:
            mapping["calib"] = self.calib_set
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}', expected one of {list(mapping)}")
        return mapping[split]
