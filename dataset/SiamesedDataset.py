"""
Siamese-network (self-supervised pretraining) data pipeline for CMAPSS.

This module reproduces, without pytorch-lightning and without ray, the
pretext-task data pipeline described in:

    Krokotsch, T., Knaak, M., & Guhmann, C. (2022).
    "Improving Semi-Supervised Learning for Remaining Useful Lifetime
    Estimation Through Self-Supervision."
    Reference code: https://github.com/tilman151/self-supervised-ssl
    (see loader.py / cmapss.py / baseline.py in that repository).

WHAT THIS REPRODUCES
---------------------
The paper pretrains a siamese encoder on a *pretext task*: given two
windows sampled from the same engine run, predict how many cycles apart
they are (normalized by max_rul), while also exposing a "domain" label
(failed run vs. not-yet-failed/"broken" run) so a domain-adversarial
branch can be trained alongside it. This module ports that pairing logic
(`PairedCMAPSS` in the original repository) into a plain
`torch.utils.data.IterableDataset` (here: `SiamesePairDataset`).

WHAT IS DIFFERENT FROM THE ORIGINAL CODE
------------------------------------------
1. No pytorch-lightning, no ray: everything is plain `torch.utils.data`.
2. All preprocessing (windowing, max_rul clipping, normalization,
   operating-condition clustering) is delegated to the project's own
   `CMAPSSLoader` / `CMAPSSDataset` pipeline instead of the paper's
   `CMAPSSLoader`.
3. Most importantly, the "broken" (not-yet-failed, partially observed)
   domain is *not* recreated by this module. In the original repository,
   `PretrainingBaselineDataModule` builds it itself with
   `percent_broken` / `percent_fail_runs` truncation applied to whole
   runs. Here that censoring is generated upstream, directly by
   `CMAPSSDataset` (see `percent_of_censored_data` and
   `percent_of_broken_data` in CMAPSSDataset.__init__): a fraction of
   the units (`percent_of_censored_data`) are truncated to only their
   first `percent_of_broken_data` cycles, exactly the same idea as the
   paper's truncation, just performed once, upstream, by the dataset
   class shared with every other model in this project. `SiameseDataset`
   simply reads the `is_censored` flag that `CMAPSSDataset` attaches to
   every row to recover the two domains:
       is_censored == 1  ->  "broken" domain   (not observed to failure)
       is_censored == 0  ->  "fail" domain     (observed to failure)
   This is the same convention used by the original `PairedCMAPSS`,
   which assigns domain label 0 to the broken/unfailed domain and 1 to
   the failed domain.
4. Feature selection: the paper selects features by integer index into
   the 24 columns [op1, op2, op3, s1, ..., s21]. `CMAPSSDataset` selects
   features by column name, so `SiameseDataset.default_feature_cols()`
   translates the paper's default channel indices into column names.

NOTE ON THE "labeled" DISTANCE MODE
------------------------------------
`distance_mode='labeled'` (see `SiamesePairDataset`) builds the
anchor/query distance from the difference of their RUL labels instead of
their cycle-index difference. For "broken" (censored) units, the RUL
label of the truncated last cycle is still computed by `CMAPSSDataset`
as if the unit had failed there (CMAPSSDataset has no notion of a
"right-censored, true RUL unknown" label, only the boolean
`is_censored` flag). This only matters if `distance_mode='labeled'` is
used; the paper's default ('linear') only relies on cycle-index
differences and is unaffected.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from C_MAPSS.dataset.CMAPSSDataset import CMAPSSDataset
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader


class SiamesePairDataset(IterableDataset):
    """
    Samples (anchor, query, distance, domain_label) tuples used to
    pretrain a siamese encoder with the pretext task described in the
    paper. Ported almost verbatim from `cmapss.py::PairedCMAPSS` in the
    original repository, adapted to take pre-grouped per-run tensors and
    an explicit per-run domain array (derived from `is_censored`, see
    `SiameseDataset._group_windows_by_run_with_domain`) instead of a list
    of `CMAPSSLoader`-like sources.

    All anchor/query indices for a whole epoch are drawn in one
    vectorized `numpy` call in `__iter__`, and every batch is gathered
    from the flattened run tensors in a single indexing operation in
    `__next__` (the dataset yields whole batches, so it must be used with
    `DataLoader(..., batch_size=None)`). This avoids sampling and
    building pairs one Python-level `__next__` call at a time, which
    otherwise dominates wall-clock time and CPU usage while the GPU sits
    idle waiting for each batch.
    """

    def __init__(
            self,
            run_features: List[torch.Tensor],
            run_targets: List[torch.Tensor],
            max_rul: float,
            num_samples: int,
            min_distance: int,
            batch_size: int,
            deterministic: bool = False,
            mode: str = "linear",
    ):
        """
        :param run_features: one tensor per engine run, each of shape
            (n_windows_in_run, n_features, window_size) (channel-first).
        :param run_targets: one tensor per engine run, each of shape
            (n_windows_in_run,), the RUL label of every window.
        :param max_rul: RUL clipping value, used to normalize the
            anchor/query distance to [0, 1].
        :param num_samples: number of pairs sampled per epoch.
        :param min_distance: minimum number of cycles/windows between an
            anchor and its query.
        :param batch_size: number of pairs yielded per `__next__` call.
        :param deterministic: if True, re-seed the RNG identically at
            the start of every epoch (used for validation pairs).
        :param mode: 'linear', 'piecewise' or 'labeled' pairing strategy.
        """
        super().__init__()

        # Runs with too few windows cannot provide an anchor/query pair
        # respecting min_distance, so they are dropped upfront.
        long_enough = [len(features) > min_distance for features in run_features]
        features = [f for f, keep in zip(run_features, long_enough) if keep]
        labels = [t for t, keep in zip(run_targets, long_enough) if keep]

        if len(features) == 0:
            raise ValueError(
                "No run has more than min_distance windows; cannot build any "
                "anchor/query pair. Use a smaller min_distance or a smaller "
                "window_size."
            )

        self.min_distance = min_distance
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.mode = mode
        self._max_rul = max_rul

        # Flatten every run into one contiguous tensor so that an
        # arbitrary number of anchor/query pairs can be gathered with a
        # single vectorized indexing operation instead of one
        # Python-level lookup per pair.
        self._flat_features = torch.cat(features, dim=0)
        self._flat_targets = torch.cat(labels, dim=0)

        run_lengths = np.array([len(f) for f in features], dtype=np.int64)
        self._run_lengths = run_lengths
        self._run_offsets = np.concatenate([[0], np.cumsum(run_lengths)[:-1]])
        self._num_runs = len(features)

        self._rng = self._reset_rng()

        if mode == "linear":
            self._sample_pairs = self._sample_linear
        elif mode == "piecewise":
            self._sample_pairs = self._sample_piecewise
        elif mode == "labeled":
            self._sample_pairs = self._sample_labeled
        else:
            raise ValueError(f"Unknown distance mode {mode}.")

        self._epoch_anchor_idx: Optional[np.ndarray] = None
        self._epoch_query_idx: Optional[np.ndarray] = None
        self._epoch_distance: Optional[torch.Tensor] = None
        self._current_batch = 0

    @staticmethod
    def _reset_rng() -> np.random.Generator:
        return np.random.default_rng(seed=42)

    def __len__(self) -> int:
        """Number of batches yielded per epoch (the dataset batches itself)."""
        return -(-self.num_samples // self.batch_size)  # ceil division

    def __iter__(self):
        if self.deterministic:
            self._rng = self._reset_rng()

        anchor_idx, query_idx, distance = self._sample_pairs(self.num_samples)

        self._epoch_anchor_idx = anchor_idx
        self._epoch_query_idx = query_idx
        distance = torch.as_tensor(distance, dtype=torch.float) / self._max_rul
        self._epoch_distance = torch.clamp_max(distance, max=1)  # max distance is max_rul
        self._current_batch = 0

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self._current_batch * self.batch_size
        if start >= self.num_samples:
            raise StopIteration

        end = min(start + self.batch_size, self.num_samples)
        self._current_batch += 1

        anchors = self._flat_features[self._epoch_anchor_idx[start:end]]
        queries = self._flat_features[self._epoch_query_idx[start:end]]
        distances = self._epoch_distance[start:end]

        return anchors, queries, distances

    def _sample_linear(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        run_idx = self._rng.integers(0, self._num_runs, size=n)
        lengths = self._run_lengths[run_idx]

        anchor_rel = self._rng.integers(low=0, high=lengths - self.min_distance)
        end_rel = np.minimum(lengths, anchor_rel + self._max_rul)
        query_rel = self._rng.integers(low=anchor_rel + self.min_distance, high=end_rel)
        distance = (query_rel - anchor_rel).astype(np.float32)

        offsets = self._run_offsets[run_idx]
        return offsets + anchor_rel, offsets + query_rel, distance

    def _sample_piecewise(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        run_idx = self._rng.integers(0, self._num_runs, size=n)
        lengths = self._run_lengths[run_idx]
        middle_rel = lengths // 2

        anchor_rel = self._rng.integers(low=0, high=lengths - self.min_distance)
        end_rel = np.where(anchor_rel < (middle_rel - self.min_distance), middle_rel, lengths)
        query_rel = self._rng.integers(low=anchor_rel + self.min_distance, high=end_rel)
        distance = np.where(anchor_rel > middle_rel, query_rel - anchor_rel, 0).astype(np.float32)

        offsets = self._run_offsets[run_idx]
        return offsets + anchor_rel, offsets + query_rel, distance

    def _sample_labeled(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        run_idx = self._rng.integers(0, self._num_runs, size=n)
        lengths = self._run_lengths[run_idx]

        anchor_rel = self._rng.integers(low=0, high=lengths - self.min_distance)
        query_rel = self._rng.integers(low=anchor_rel + self.min_distance, high=lengths)

        offsets = self._run_offsets[run_idx]
        anchor_idx = offsets + anchor_rel
        query_idx = offsets + query_rel
        # RUL label difference is the negative of the time-step difference.
        distance = (self._flat_targets[anchor_idx] - self._flat_targets[query_idx]).numpy()

        return anchor_idx, query_idx, distance


class SiameseDataset:
    """
    Builds the CMAPSS data pipeline needed to pretrain a siamese encoder
    with the self-supervised pretext task from Krokotsch et al. (2022).

    This class produces plain `torch.utils.data.DataLoader`
    objects ready to be consumed by a hand-written PyTorch training loop.

    Reuses `CMAPSSLoader` / `CMAPSSDataset` for every preprocessing step
    (windowing, normalization, max_rul clipping, operating-condition
    clustering), including the generation of censored ("broken",
    not-yet-failed) units, which is delegated entirely to
    `CMAPSSDataset.percent_of_censored_data` / `percent_of_broken_data`
    (see module docstring for details).
    """

    # Window size used per sub-dataset in the paper.
    WINDOW_SIZES = {"FD001": 30, "FD002": 20, "FD003": 30, "FD004": 15}

    # 0-based indices into [op1, op2, op3, s1, ..., s21] (24 columns),
    # selected by the paper according to https://doi.org/10.1016/j.ress.2017.11.021
    _DEFAULT_CHANNEL_INDICES = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]

    # ============================================================================
    # FEATURE HELPERS
    # ============================================================================

    @classmethod
    def default_feature_cols(cls) -> List[str]:
        """Translate the paper's default channel indices into the column
        names expected by CMAPSSDataset's `include_cols`."""
        all_cols = CMAPSSDataset.OPERATION_COLS + CMAPSSDataset.SENSOR_COLS
        return [all_cols[i] for i in cls._DEFAULT_CHANNEL_INDICES]

    # ============================================================================
    # FACTORY METHOD
    # ============================================================================

    @classmethod
    def from_cmapss(
            cls,
            dataset_root: str,
            seed: int | None,
            sub_dataset: str = "FD001",
            window_size: Optional[int] = None,
            max_rul: int = 125,
            min_distance: int = 1,
            feature_select: Optional[List[str]] = None,
            exclude_cols: Optional[List[str]] = None,
            norm_type: str = "-1-1",
            cluster_operations: bool = False,
            norm_by_operations: bool = False,
            validation_rate: float = 0.2,
            use_only_final_on_test=True,
            use_max_rul_on_test=False,
            use_max_rul_on_valid=True,
            percent_of_censored_data: float = 0.5,
            percent_of_broken_data: Optional[float] = None,
            distance_mode: str = "linear",
            num_samples: int = 50000,
            num_val_samples: int = 25000,
            batch_size: int = 64,
            num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Build the data loaders needed to pretrain a siamese network with
        the self-supervised pairwise-distance pretext task from:

            Krokotsch, T., Knaak, M., & Guhmann, C. (2022). "Improving
            Semi-Supervised Learning for Remaining Useful Lifetime
            Estimation Through Self-Supervision."

        :param seed: Set the seed to reproduce restults
        :param dataset_root: root directory containing the raw CMAPSS txt files.
        :param sub_dataset: sub-dataset to pretrain on, 'FD001' to 'FD004'.
        :param window_size: window length; defaults to the paper's per-fd value.
        :param max_rul: RUL clipping value (piece-wise linear RUL function),
            also used to normalize the anchor/query distance to [0, 1].
        :param min_distance: minimum number of cycles between anchor and query.
        :param feature_select: feature column names to use; defaults to the
            paper's selection translated to column names.
        :param norm_type: normalization scheme forwarded to CMAPSSDataset
            ('0-1', '-1-1' or 'z-score'). The paper uses '-1-1'.
        :param cluster_operations: forwarded to CMAPSSDataset; cluster the
            3 operating settings with KMeans before normalizing.
        :param norm_by_operations: forwarded to CMAPSSDataset; normalize
            separately per operating-condition cluster.
        :param validation_rate: forwarded to CMAPSSLoader; fraction of
            engine units held out as the validation split.
        :param percent_of_censored_data: forwarded to CMAPSSDataset; fraction
            of train/validation units that are censored ("broken" domain,
            i.e. truncated before failure). May be 0.0 when you want to run
            the pair sampler without a censored domain.
        :param percent_of_broken_data: forwarded to CMAPSSDataset; fraction
            of cycles kept for each censored unit. None means a random
            fraction is drawn independently for every censored unit, same
            as in the article.
        :param distance_mode: 'linear', 'piecewise' or 'labeled' pairing
            strategy, see `SiamesePairDataset`.
        :param num_samples: number of pairs sampled per training epoch.
        :param num_val_samples: number of pairs sampled for the validation
            pair loader (paper hardcodes 25000).
        :param batch_size: batch size for all returned loaders.
        :param num_workers: number of DataLoader workers.

        :return: (train_pair_loader, val_pair_loader, source_val_loader, test_dataset)
            - train_pair_loader: yields (anchor, query, distance, domain_label)
              batches sampled from the train split, used to pretrain the
              siamese network on the pretext task.
            - val_pair_loader: same format, deterministic, sampled from the
              validation split, used to monitor the pretext task.
            - source_val_loader: standard (features, RUL) batches built from
              the "broken" (censored) domain of the validation split, used
              to monitor downstream RUL performance during pretraining, as
              in the original paper.
            - test_dataset: the raw `CMAPSSDataset` test split (never
              censored, see CMAPSSLoader), provided for the downstream
              supervised fine-tuning/evaluation step that follows
              pretraining; it is not used by the pretext task itself.
        """
        assert sub_dataset in cls.WINDOW_SIZES, f"sub_dataset must be one of {list(cls.WINDOW_SIZES)}, got {sub_dataset}"
        assert 0 <= percent_of_censored_data <= 1, (
            "percent_of_censored_data must be between 0 and 1 (see CMAPSSDataset.percent_of_censored_data)."
        )

        window_size = window_size or cls.WINDOW_SIZES[sub_dataset]
        # feature_select = list(feature_select) if feature_select is not None else cls.default_feature_cols()
        feature_select = list(feature_select) if feature_select else []

        # --- Step 1: run the project's own preprocessing pipeline ---------------
        # This applies normalization, max_rul clipping, optional operating-
        # condition clustering, and (for train/valid only) censored-data
        # generation, all delegated to CMAPSSDataset/CMAPSSLoader.
        train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(
            dataset_root=dataset_root,
            sub_dataset=sub_dataset,
            sequence_len=window_size,
            seed=seed,
            max_rul=max_rul,
            return_sequence_label=False,
            norm_type=norm_type,
            cluster_operations=cluster_operations,
            norm_by_operations=norm_by_operations,
            include_cols=feature_select,
            exclude_cols=exclude_cols,
            return_id=False,
            validation_rate=validation_rate,
            use_only_final_on_test=use_only_final_on_test,
            use_max_rul_on_test=use_max_rul_on_test,
            use_max_rul_on_valid=use_max_rul_on_valid,
            percent_of_censored_data=percent_of_censored_data,
            percent_of_broken_data=percent_of_broken_data,
        )

        # --- Step 2: regroup windows by run and recover the domain label --------
        # from the 'is_censored' flag CMAPSSDataset attached to every row.
        train_features, train_targets = cls._group_windows_by_run(train_dataset)
        val_features, val_targets = cls._group_windows_by_run(valid_dataset)

        # --- Step 3: paired (siamese) datasets -----------------------------------
        train_pairs = SiamesePairDataset(
            train_features, train_targets,
            max_rul=max_rul,
            num_samples=num_samples,
            min_distance=min_distance,
            batch_size=batch_size,
            deterministic=False,
            mode=distance_mode,
        )
        val_pairs = SiamesePairDataset(
            val_features, val_targets,
            max_rul=max_rul,
            num_samples=num_val_samples,
            min_distance=1,
            batch_size=batch_size,
            deterministic=True,
            mode=distance_mode,
        )

        # `SiamesePairDataset` batches itself (see class docstring), so the
        # DataLoader must not re-batch/collate on top of it.
        train_pair_loader = DataLoader(
            train_pairs, batch_size=None, pin_memory=True, num_workers=num_workers
        )
        val_pair_loader = DataLoader(
            val_pairs, batch_size=None, pin_memory=True, num_workers=num_workers
        )

        return train_pair_loader, val_pair_loader

    # ============================================================================
    # INTERNAL HELPERS
    # ============================================================================

    @staticmethod
    def _group_windows_by_run(
            dataset: CMAPSSDataset,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Re-group the flat array of windows produced by
        `CMAPSSDataset._gen_sequence()` back into one array per engine run,
        and attach a domain label (0 = "broken"/censored, 1 = "fail") to
        each run, read from the 'is_censored' column CMAPSSDataset attaches
        to `dataset.df`.

        `CMAPSSDataset` stacks the windows of every unit into a single flat
        array, but the siamese pretext task needs to know run boundaries to
        sample an anchor and a query window from the *same* run. The run id
        of each window is still available in `dataset.id_array`, so it is
        used to recover the grouping.

        :param dataset: a windowed CMAPSSDataset (train or validation split).
        :return: (run_features, run_targets, run_domains).
            run_features[i] is a tensor of shape (n_windows_in_run,
            n_features, window_size) (channel-first, as expected by the
            paper's CNN encoders); run_targets[i] has shape
            (n_windows_in_run,).
        """
        if not dataset.has_gen_sequence:
            raise RuntimeError(
                "Dataset windows have not been generated yet. Make sure this "
                "dataset was produced by CMAPSSLoader.get_datasets(), which "
                "calls _gen_sequence() automatically."
            )
        if "is_censored" not in dataset.df.columns:
            raise RuntimeError(
                "No 'is_censored' column found on the dataset. The siamese "
                "loader expects CMAPSSDataset to attach this column even when "
                "percent_of_censored_data is 0.0."
            )

        run_ids = np.unique(dataset.id_array)
        run_features, run_targets = [], []
        for run_id in run_ids:
            run_mask = dataset.id_array == run_id
            run_features.append(dataset.sequence_array[run_mask])
            run_targets.append(dataset.label_array[run_mask])

        run_features, run_targets = SiameseDataset._to_tensor(run_features, run_targets)

        return run_features, run_targets

    @staticmethod
    def _to_tensor(
            run_features: List[np.ndarray], run_targets: List[np.ndarray]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Convert per-run numpy windows to torch tensors and switch to the
        channel-first layout (n_features, window_size) expected by the
        paper's CNN encoders."""
        run_features = [
            torch.tensor(feat, dtype=torch.float32).permute(0, 2, 1) for feat in run_features
        ]
        run_targets = [torch.tensor(targ, dtype=torch.float32) for targ in run_targets]

        return run_features, run_targets

    @staticmethod
    def _build_source_val_loader(
            dataset: CMAPSSDataset, batch_size: int, num_workers: int
    ) -> DataLoader:
        """Build a flat (features, RUL) DataLoader over the windows of the
        "broken" (censored) domain only, used to monitor downstream RUL
        performance while pretraining the siamese network, as in the
        original paper's `source_val_dataloader`."""
        censored_by_id = dataset.df.groupby("id")["is_censored"].first()
        window_is_censored = np.array(
            [bool(censored_by_id.loc[unit_id]) for unit_id in dataset.id_array]
        )

        features = torch.tensor(
            dataset.sequence_array[window_is_censored], dtype=torch.float32
        ).permute(0, 2, 1)
        targets = torch.tensor(
            dataset.label_array[window_is_censored], dtype=torch.float32
        )

        tensor_dataset = TensorDataset(features, targets)

        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )