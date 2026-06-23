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

from typing import List, Optional, Tuple, Union

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
    """

    def __init__(
            self,
            run_features: List[torch.Tensor],
            run_targets: List[torch.Tensor],
            run_domains: np.ndarray,
            max_rul: float,
            num_samples: int,
            min_distance: int,
            deterministic: bool = False,
            mode: str = "linear",
    ):
        """
        :param run_features: one tensor per engine run, each of shape
            (n_windows_in_run, n_features, window_size) (channel-first).
        :param run_targets: one tensor per engine run, each of shape
            (n_windows_in_run,), the RUL label of every window.
        :param run_domains: array of shape (n_runs,), domain label of
            every run (0 = "broken"/censored domain, 1 = "fail" domain).
        :param max_rul: RUL clipping value, used to normalize the
            anchor/query distance to [0, 1].
        :param num_samples: number of pairs sampled per epoch.
        :param min_distance: minimum number of cycles/windows between an
            anchor and its query.
        :param deterministic: if True, re-seed the RNG identically at
            the start of every epoch (used for validation pairs).
        :param mode: 'linear', 'piecewise' or 'labeled' pairing strategy.
        """
        super().__init__()

        # Runs with too few windows cannot provide an anchor/query pair
        # respecting min_distance, so they are dropped upfront.
        long_enough = [len(features) > min_distance for features in run_features]
        self._features = [f for f, keep in zip(run_features, long_enough) if keep]
        self._labels = [t for t, keep in zip(run_targets, long_enough) if keep]
        self._run_domain_idx = np.asarray(
            [d for d, keep in zip(run_domains, long_enough) if keep]
        )

        if len(self._features) == 0:
            raise ValueError(
                "No run has more than min_distance windows; cannot build any "
                "anchor/query pair. Use a smaller min_distance or a smaller "
                "window_size."
            )

        self.min_distance = min_distance
        self.num_samples = num_samples
        self.deterministic = deterministic
        self.mode = mode
        self._max_rul = max_rul

        self._current_iteration = 0
        self._rng = self._reset_rng()

        if mode == "linear":
            self._get_pair_func = self._get_pair_idx
        elif mode == "piecewise":
            self._get_pair_func = self._get_pair_idx_piecewise
        elif mode == "labeled":
            self._get_pair_func = self._get_labeled_pair_idx
        else:
            raise ValueError(f"Unknown distance mode {mode}.")

    @staticmethod
    def _reset_rng() -> np.random.Generator:
        return np.random.default_rng(seed=42)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        self._current_iteration = 0
        if self.deterministic:
            self._rng = self._reset_rng()

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._current_iteration < self.num_samples:
            self._current_iteration += 1
            pair_idx = self._get_pair_func()
            return self._build_pair(*pair_idx)
        else:
            raise StopIteration

    def _get_pair_idx(self) -> Tuple[torch.Tensor, int, int, int, int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = self._run_domain_idx[chosen_run_idx]
        chosen_run = self._features[chosen_run_idx]

        run_length = chosen_run.shape[0]
        anchor_idx = self._rng.integers(low=0, high=run_length - self.min_distance)
        end_idx = min(run_length, anchor_idx + self._max_rul)
        query_idx = self._rng.integers(low=anchor_idx + self.min_distance, high=end_idx)
        distance = query_idx - anchor_idx

        return chosen_run, anchor_idx, query_idx, distance, domain_label

    def _get_pair_idx_piecewise(self) -> Tuple[torch.Tensor, int, int, int, int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = self._run_domain_idx[chosen_run_idx]
        chosen_run = self._features[chosen_run_idx]

        run_length = chosen_run.shape[0]
        middle_idx = run_length // 2
        anchor_idx = self._rng.integers(low=0, high=run_length - self.min_distance)
        end_idx = (
            middle_idx if anchor_idx < (middle_idx - self.min_distance) else run_length
        )
        query_idx = self._rng.integers(low=anchor_idx + self.min_distance, high=end_idx)
        distance = query_idx - anchor_idx if anchor_idx > middle_idx else 0

        return chosen_run, anchor_idx, query_idx, distance, domain_label

    def _get_labeled_pair_idx(self) -> Tuple[torch.Tensor, int, int, int, int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = self._run_domain_idx[chosen_run_idx]
        chosen_run = self._features[chosen_run_idx]
        chosen_labels = self._labels[chosen_run_idx]

        run_length = chosen_run.shape[0]
        anchor_idx = self._rng.integers(low=0, high=run_length - self.min_distance)
        query_idx = self._rng.integers(low=anchor_idx + self.min_distance, high=run_length)
        # RUL label difference is the negative of the time-step difference.
        distance = chosen_labels[anchor_idx] - chosen_labels[query_idx]

        return chosen_run, anchor_idx, query_idx, distance, domain_label

    def _build_pair(
            self,
            run: torch.Tensor,
            anchor_idx: int,
            query_idx: int,
            distance: Union[int, float, torch.Tensor],
            domain_label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = run[anchor_idx]
        query = run[query_idx]
        domain_label = torch.tensor(domain_label, dtype=torch.float)
        distance = torch.as_tensor(distance, dtype=torch.float) / self._max_rul
        distance = torch.clamp_max(distance, max=1)  # max distance is max_rul

        return anchor, query, distance, domain_label


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
            norm_type: str = "-1-1",
            cluster_operations: bool = False,
            norm_by_operations: bool = False,
            validation_rate: float = 0.2,
            percent_of_censored_data: float = 0.5,
            percent_of_broken_data: Optional[float] = None,
            distance_mode: str = "linear",
            num_samples: int = 50000,
            num_val_samples: int = 25000,
            batch_size: int = 64,
            num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, CMAPSSDataset]:
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
            i.e. truncated before failure). Required to be > 0: the pretext
            task needs both a "broken" and a "fail" domain to build its
            domain label and to expose unlabeled, partially observed runs.
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
        assert 0 < percent_of_censored_data <= 1, (
            "The siamese pretext task needs both a 'fail' and a 'broken' domain, so "
            "percent_of_censored_data must be > 0 (see CMAPSSDataset.percent_of_censored_data)."
        )

        window_size = window_size or cls.WINDOW_SIZES[sub_dataset]
        # feature_select = list(feature_select) if feature_select is not None else cls.default_feature_cols()
        feature_select = list(feature_select) if feature_select else None

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
            validation_rate=validation_rate,
            percent_of_censored_data=percent_of_censored_data,
            percent_of_broken_data=percent_of_broken_data,
        )

        # --- Step 2: regroup windows by run and recover the domain label --------
        # from the 'is_censored' flag CMAPSSDataset attached to every row.
        train_features, train_targets, train_domains = cls._group_windows_by_run_with_domain(train_dataset)
        val_features, val_targets, val_domains = cls._group_windows_by_run_with_domain(valid_dataset)

        # --- Step 3: paired (siamese) datasets -----------------------------------
        train_pairs = SiamesePairDataset(
            train_features, train_targets, train_domains,
            max_rul=max_rul,
            num_samples=num_samples,
            min_distance=min_distance,
            deterministic=False,
            mode=distance_mode,
        )
        val_pairs = SiamesePairDataset(
            val_features, val_targets, val_domains,
            max_rul=max_rul,
            num_samples=num_val_samples,
            min_distance=1,
            deterministic=True,
            mode=distance_mode,
        )

        train_pair_loader = DataLoader(
            train_pairs, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        val_pair_loader = DataLoader(
            val_pairs, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )

        # --- Step 4: plain supervised val loader on the "broken" domain ---------
        # Used in the paper to track downstream RUL performance while
        # pretraining the siamese network.
        source_val_loader = cls._build_source_val_loader(valid_dataset, batch_size, num_workers)

        return train_pair_loader, val_pair_loader, source_val_loader, test_dataset

    # ============================================================================
    # INTERNAL HELPERS
    # ============================================================================

    @staticmethod
    def _group_windows_by_run_with_domain(
            dataset: CMAPSSDataset,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], np.ndarray]:
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
            (n_windows_in_run,); run_domains has shape (n_runs,).
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
                "pretext task needs CMAPSSDataset to generate censored "
                "('broken') units; pass percent_of_censored_data > 0 to "
                "SiameseDataset.from_cmapss()."
            )

        # 'is_censored' is constant within a unit, so the first row of each
        # id gives the right value.
        censored_by_id = dataset.df.groupby("id")["is_censored"].first()

        run_ids = np.unique(dataset.id_array)
        run_features, run_targets, run_domains = [], [], []
        for run_id in run_ids:
            run_mask = dataset.id_array == run_id
            run_features.append(dataset.sequence_array[run_mask])
            run_targets.append(dataset.label_array[run_mask])

            is_censored = bool(censored_by_id.loc[run_id])
            # Matches the original PairedCMAPSS convention: the "broken"
            # (not observed to failure) domain is labeled 0, the "fail"
            # (observed to failure) domain is labeled 1.
            run_domains.append(0 if is_censored else 1)

        run_features, run_targets = SiameseDataset._to_tensor(run_features, run_targets)

        return run_features, run_targets, np.array(run_domains)

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