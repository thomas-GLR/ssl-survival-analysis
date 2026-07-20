"""
Scania Component X dataset.

A ``torch.utils.data.Dataset`` that turns pre-processed Scania operational
readouts into sliding-window sequences, mirroring the contract of
``C_MAPSS/dataset/CMAPSSDataset.py`` so the existing supervised / co-training
(Coprog) / self-supervised training code can consume Scania data unchanged.

The dataframe passed to ``__init__`` is expected to already be pre-processed by
``ScaniaDataModule`` (per-vehicle NaN fill + counter differencing done, TTE
merged), i.e. it holds, per readout row:

    vehicle_id, time_step, <counter delta features...>,
    length_of_study_time_step, is_censored

This class only computes the RUL target, normalizes the features (z-score) and
generates the windowed arrays.

Scania specifics vs C_MAPSS:
- Censoring is real (``in_study_repair == 0``), never synthetically generated.
- Censored rows have **no** RUL (label is NaN); a ``rul_lower_bound`` (the time
  observed until the end of the study) is kept instead, so a model can enforce
  "predicted RUL >= observed survival time".
- The test split is additionally built with ``only_final=True`` (set
  internally by ``ScaniaDataModule``, never exposed publicly), so evaluation
  sees exactly one window per vehicle -- its last, possibly truncated,
  readout -- instead of every sliding-window stride.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from constants.scania_component_x_columns import (
    VEHICLE_ID,
    TIME_STEP,
    LENGTH_OF_STUDY_TIME_STEP,
    COUNTER_COLUMNS,
)

# Columns produced by this class / expected in the pre-processed frame.
IS_CENSORED = "is_censored"
RUL = "rul"
RUL_LOWER_BOUND = "rul_lower_bound"


class HistogramFeatureNormalizer:
    """Sum-based normalizer for Scania histogram variables.

    Histogram variables are cumulative per-bin counts (a distribution over bins).
    Z-scoring them per bin would destroy the distribution shape, so instead each
    feature group is divided by the average per-row total of that group: bins are
    turned into fractions of the group's typical total. Fitting computes one
    scalar per feature group on the training split; the same scalars are reused
    on val/test to avoid leakage.

    The feature groups are derived from the column names by splitting on the last
    ``_`` (e.g. ``"397_35"`` belongs to group ``"397"``), matching the
    ``"<feature_id>_<bin>"`` naming convention used across the codebase.
    """

    def __init__(self, histogram_cols: list[str]):
        """
        :param histogram_cols:
            Flat list of histogram bin column names (e.g. ``"167_0"``,
            ``"167_1"``, ...). Grouped internally by feature id.
        """
        self.histogram_features: dict[str, list[str]] = {}
        for column in histogram_cols:
            feature = column.rsplit("_", 1)[0]
            self.histogram_features.setdefault(feature, []).append(column)
        # Per-feature normalization scalar, populated by ``fit``.
        self.normalization_params: dict[str, float] = {}

    def fit(self, x: pd.DataFrame) -> "HistogramFeatureNormalizer":
        """Compute, per feature group, the average per-row bin total (+ epsilon).

        :param x: dataframe holding at least the histogram bin columns.
        :return: self (fitted).
        """
        epsilon = 1e-6
        for feature, columns in self.histogram_features.items():
            # Average, over rows, of the total count across the group's bins.
            feature_sum = float(x[columns].sum(axis=1).mean())
            self.normalization_params[feature] = feature_sum + epsilon
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Divide each feature group's bin columns by its fitted scalar.

        :param x: dataframe holding the histogram bin columns.
        :return: a copy of ``x`` with the histogram columns normalized.
        """
        x_transformed = x.copy()
        for feature, columns in self.histogram_features.items():
            feature_sum = self.normalization_params[feature]
            x_transformed[columns] = x[columns] / feature_sum
        return x_transformed

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``x`` then return the normalized ``x`` (train-split convenience)."""
        self.fit(x)
        return self.transform(x)


class ScaniaDataset(Dataset):
    def __init__(
            self,
            data_df: pd.DataFrame,
            sequence_len: int,
            feature_cols: list[str] | None = None,
            norm_type: str | None = None,
            norm_params: np.ndarray | None = None,
            histogram_cols: list[str] | None = None,
            hist_norm_params: dict[str, float] | None = None,
            return_sequence_label: bool = False,
            return_id: bool = False,
            only_final: bool = False,
            seed: int | None = None,
    ):
        """
        :param data_df:
            Pre-processed Scania readouts (see module docstring). Must contain
            ``vehicle_id``, ``time_step``, the feature columns,
            ``length_of_study_time_step`` and ``is_censored``.
        :param sequence_len:
            Length of the sliding time window (stride 1). Short vehicles are
            left-edge-padded, same convention as CMAPSSDataset.
        :param feature_cols:
            Feature columns to use. Defaults to ``COUNTER_COLUMNS``.
        :param norm_type:
            ``'z-score'`` or ``None`` (no normalization).
        :param norm_params:
            Pre-computed z-score params of shape ``(n_zscore_features, 2)`` =
            ``[mean, std]``, aligned to the non-histogram feature columns. If
            ``None`` and ``norm_type`` is set, they are computed from this
            dataframe (use this only for the train split and pass
            ``train.norm_params`` to val/test to avoid leakage).
        :param histogram_cols:
            Subset of ``feature_cols`` holding histogram bin columns. These are
            excluded from the z-score and normalized instead by
            ``HistogramFeatureNormalizer`` (sum-based). Empty/None -> no
            histogram features.
        :param hist_norm_params:
            Pre-computed histogram normalization scalars (``{feature_id: sum}``).
            Same leakage contract as ``norm_params``: fit on train, reused for
            val/test. If ``None`` and histograms are present, they are fit here.
        :param return_sequence_label:
            If True, ``__getitem__``/label array return the RUL for every step
            of the window instead of only the last step.
        :param return_id:
            If True, ``__getitem__`` also returns the vehicle id.
        :param only_final:
            If True, keep only the last sliding window per vehicle (the
            window ending at that vehicle's final row) instead of every
            stride. Mirrors CMAPSS's ``only_final``. This is an internal
            flag: ``ScaniaDataModule`` sets it (only for its test split); it
            is never part of the module's public config surface.
        :param seed:
            Seeds numpy for reproducibility.
        """
        super().__init__()
        assert isinstance(data_df, pd.DataFrame), "data_df need pd.DataFrame"
        assert sequence_len > 0, "Need sequence_len > 0, got: " + str(sequence_len)

        if seed is not None:
            np.random.seed(seed)

        assert norm_type in (None, "z-score"), f"Unsupported norm_type: {norm_type}"

        self.df = data_df.copy()
        self.sequence_len = sequence_len
        self.feature_cols = list(feature_cols) if feature_cols else list(COUNTER_COLUMNS)
        self.norm_type = norm_type
        self.norm_params = norm_params
        self.histogram_cols = list(histogram_cols) if histogram_cols else []
        self.hist_norm_params = hist_norm_params
        # Columns z-scored by norm_type: every feature column that is not a
        # histogram bin (histograms get their own sum-based normalizer).
        histogram_set = set(self.histogram_cols)
        self._zscore_cols = [c for c in self.feature_cols if c not in histogram_set]
        self.return_sequence_label = return_sequence_label
        self.return_id = return_id
        self.only_final = only_final

        # Lazy windowing state populated by _gen_sequence(). Instead of a dense
        # (n_windows, seq_len, n_feat) array (which explodes to tens of GB once
        # histograms push n_feat to ~105), we keep the flat per-row feature
        # matrix once and the per-window start/length; each window is sliced (and
        # edge-padded for short vehicles) on demand in _get_window/_build_windows.
        self._feat_flat: np.ndarray | None = None   # (rows, n_feat) float32
        self._win_start: np.ndarray | None = None   # (N,) int64
        self._win_count: np.ndarray | None = None   # (N,) int32
        self._n_windows: int = 0

        # Small per-window metadata (no n_feat factor -> cheap, kept eager and
        # bit-identical to the previous implementation).
        self.label_array: np.ndarray | None = None
        self.lower_bound_array: np.ndarray | None = None
        self.id_array: np.ndarray | None = None
        self.is_censored_array: np.ndarray | None = None

        self.count_rul()

        if self.norm_type:
            self._normalization()

        self._gen_sequence()

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, i):
        """
        :return: (sequence, target[, id])
            sequence: FloatTensor (sequence_len, n_features)
            target:   FloatTensor (1,) last-step RUL, or (sequence_len,) if
                      return_sequence_label. NaN for censored windows.
        """
        seq = torch.from_numpy(self._get_window(i))
        if self.return_sequence_label:
            target = torch.FloatTensor(self.label_array[i])
        else:
            target = torch.FloatTensor([self.label_array[i]])
        items = [seq, target]
        if self.return_id:
            items.append(torch.LongTensor([int(self.id_array[i])]))
        return tuple(items)

    # ------------------------------------------------------------------ #
    # Target / RUL
    # ------------------------------------------------------------------ #
    def count_rul(self) -> None:
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

    # ------------------------------------------------------------------ #
    # Normalization (z-score for counters, sum-based for histograms)
    # ------------------------------------------------------------------ #
    def _normalization(self) -> None:
        """Normalize the feature columns in place.

        Counter (non-histogram) columns are z-scored; histogram columns are
        normalized by ``HistogramFeatureNormalizer``. Both param sets are fit
        here for the train split (``*_norm_params is None``) and reused as-is
        when passed in for val/test.
        """
        df = self.df

        # --- z-score the counter columns --------------------------------- #
        cols = self._zscore_cols
        if cols:
            df[cols] = df[cols].astype(np.float64)

            if self.norm_params is None:
                self.norm_params = self._gen_norm_params()

            mean = self.norm_params[:, 0]
            std = self.norm_params[:, 1]
            std_safe = np.where(std == 0, 1.0, std)
            df[cols] = (df[cols].to_numpy() - mean) / std_safe

        # --- sum-normalize the histogram columns ------------------------- #
        if self.histogram_cols:
            df[self.histogram_cols] = df[self.histogram_cols].astype(np.float64)
            normalizer = HistogramFeatureNormalizer(self.histogram_cols)
            if self.hist_norm_params is None:
                normalizer.fit(df)
                self.hist_norm_params = normalizer.normalization_params
            else:
                normalizer.normalization_params = self.hist_norm_params
            df[self.histogram_cols] = normalizer.transform(df[self.histogram_cols])

        self.df = df

    def _gen_norm_params(self) -> np.ndarray:
        vals = self.df[self._zscore_cols].to_numpy(dtype=np.float64)
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)
        return np.stack((mean, std), axis=1)

    # ------------------------------------------------------------------ #
    # Sequence generation
    # ------------------------------------------------------------------ #
    def _gen_sequence(self) -> None:
        """Compute the per-window index metadata (lazy windowing).

        Instead of materializing a dense ``(n_windows, seq_len, n_feat)`` array
        (which grows to tens of GB once histograms push ``n_feat`` to ~105), we
        keep the flat, sorted feature matrix ``self._feat_flat`` once and, per
        window, only its start row ``self._win_start`` and real length
        ``self._win_count`` (``seq_len`` for long vehicles, the vehicle length
        for short ones). Windows are sliced/edge-padded on demand in
        ``_get_window`` / ``_build_windows``.

        The index math is identical to the previous eager implementation: sort
        once, find contiguous per-vehicle boundaries, take every long-vehicle
        stride whose first and last row share a vehicle, and give each short
        vehicle a single left-edge-padded window. ``only_final`` still filters
        the long branch to each vehicle's final window. The small metadata
        arrays (``label_array`` / ``lower_bound_array`` / ``id_array`` /
        ``is_censored_array``) are built eagerly, long-then-short, so they are
        bit-identical to before and callers keep the same row order.
        """
        seq_len = self.sequence_len

        # Sort once so each vehicle's rows are contiguous and time-ordered.
        df = self.df.sort_values([VEHICLE_ID, TIME_STEP]).reset_index(drop=True)

        vehicules_ids = df[VEHICLE_ID].to_numpy()
        # Store the features flat and once, as float32 (the model consumes
        # float32 anyway); this is the only large array kept in memory.
        self._feat_flat = df[self.feature_cols].to_numpy(dtype=np.float32)
        rul = df[RUL].to_numpy(dtype=np.float64)
        rul_lower_bound = df[RUL_LOWER_BOUND].to_numpy(dtype=np.float64)
        censored = df[IS_CENSORED].to_numpy()
        rows_number = len(df)

        # Contiguous group boundaries (rows are grouped by vehicle after sort).
        starts = np.concatenate(([0], np.flatnonzero(np.diff(vehicules_ids) != 0) + 1))
        counts = np.diff(np.concatenate((starts, [rows_number])))

        start_chunks: list[np.ndarray] = []
        count_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        lb_chunks: list[np.ndarray] = []
        id_chunks: list[np.ndarray] = []
        cens_chunks: list[np.ndarray] = []

        # --- Long vehicles (count >= seq_len): all window starts at once ----- #
        if rows_number >= seq_len:
            sequence_group = np.arange(rows_number - seq_len + 1)
            last = sequence_group + seq_len - 1
            # A window is valid iff its first and last row belong to the same
            # vehicle (contiguity guarantees no interleaving in between).
            valid = vehicules_ids[sequence_group] == vehicules_ids[last]
            sequence_group = sequence_group[valid]
            last = last[valid]

            if self.only_final:
                # Keep, per vehicle, only the window whose last row is that
                # vehicle's final row in the sorted frame (mirrors CMAPSS's
                # only_final). starts/counts already give each vehicle's
                # final row index as starts + counts - 1.
                is_vehicle_last_row = np.zeros(rows_number, dtype=bool)
                is_vehicle_last_row[starts + counts - 1] = True
                final_mask = is_vehicle_last_row[last]
                sequence_group = sequence_group[final_mask]
                last = last[final_mask]

            if len(sequence_group):
                start_chunks.append(sequence_group.astype(np.int64))
                count_chunks.append(np.full(len(sequence_group), seq_len, dtype=np.int32))

                if self.return_sequence_label:
                    rul_sw = np.lib.stride_tricks.sliding_window_view(rul, seq_len, axis=0)
                    label_chunks.append(rul_sw[sequence_group])
                else:
                    label_chunks.append(rul[last])

                lb_chunks.append(rul_lower_bound[last])
                id_chunks.append(vehicules_ids[sequence_group])
                cens_chunks.append(censored[sequence_group])

        # --- Short vehicles (count < seq_len): one edge-padded window each --- #
        short_mask = counts < seq_len
        short_starts = starts[short_mask]
        short_counts = counts[short_mask]
        if len(short_starts):
            start_chunks.append(short_starts.astype(np.int64))
            count_chunks.append(short_counts.astype(np.int32))

            if self.return_sequence_label:
                for start, count in zip(short_starts, short_counts):
                    label_chunks.append(
                        np.pad(rul[start: start + count], (seq_len - count, 0), "edge")[np.newaxis]
                    )
            else:
                label_chunks.append(rul[short_starts + short_counts - 1])

            lb_chunks.append(rul_lower_bound[short_starts + short_counts - 1])
            id_chunks.append(vehicules_ids[short_starts])
            cens_chunks.append(censored[short_starts])

        self._win_start = np.concatenate(start_chunks, axis=0) if start_chunks else np.empty(0, np.int64)
        self._win_count = np.concatenate(count_chunks, axis=0) if count_chunks else np.empty(0, np.int32)
        self._n_windows = int(len(self._win_start))
        self.label_array = np.concatenate(label_chunks, axis=0)
        self.lower_bound_array = np.concatenate(lb_chunks, axis=0)
        self.id_array = np.concatenate(id_chunks, axis=0)
        # is_censored is constant within a vehicle.
        self.is_censored_array = np.concatenate(cens_chunks, axis=0).astype(int)

    # ------------------------------------------------------------------ #
    # On-demand window materialization
    # ------------------------------------------------------------------ #
    def _get_window(self, i: int) -> np.ndarray:
        """Return window ``i`` as a ``(seq_len, n_feat)`` float32 array.

        Long-vehicle windows are a contiguous slice of ``_feat_flat``; short
        vehicles (fewer real rows than ``seq_len``) are left-edge-padded, exactly
        reproducing the previous eager windowing.
        """
        start = int(self._win_start[i])
        count = int(self._win_count[i])
        block = self._feat_flat[start: start + count]
        if count < self.sequence_len:
            block = np.pad(block, ((self.sequence_len - count, 0), (0, 0)), "edge")
        return block

    def _build_windows(self, indices: np.ndarray) -> np.ndarray:
        """Materialize a ``(len(indices), seq_len, n_feat)`` float32 array.

        Used by the accessors that hand whole tensors to the co-training /
        self-supervised paradigms. Long windows are filled from a
        ``sliding_window_view`` in chunks so the transient copy stays small (it
        does not double the output); short windows are filled individually.

        :param indices: window indices to materialize (order preserved).
        :return: dense windows for exactly those indices.
        """
        indices = np.asarray(indices, dtype=np.int64)
        n = len(indices)
        seq_len = self.sequence_len
        n_feat = self._feat_flat.shape[1]
        out = np.empty((n, seq_len, n_feat), dtype=np.float32)
        if n == 0:
            return out

        counts = self._win_count[indices]
        is_long = counts == seq_len
        long_pos = np.flatnonzero(is_long)
        short_pos = np.flatnonzero(~is_long)

        if len(long_pos):
            sw = np.lib.stride_tricks.sliding_window_view(self._feat_flat, seq_len, axis=0)
            long_starts = self._win_start[indices[long_pos]]
            chunk = 8192
            for k in range(0, len(long_pos), chunk):
                sl = slice(k, k + chunk)
                # sw[start] -> (n_feat, seq_len); reorder to (seq_len, n_feat).
                out[long_pos[sl]] = sw[long_starts[sl]].transpose(0, 2, 1)

        for pos in short_pos:
            out[pos] = self._get_window(int(indices[pos]))

        return out

    # ------------------------------------------------------------------ #
    # Accessors used by the different training paradigms
    # ------------------------------------------------------------------ #
    def get_data_loader_without_censored_data(
            self,
            batch_size: int | None,
            shuffle: bool = False,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> DataLoader:
        """Supervised path: DataLoader over uncensored windows only, yields (x, y).

        Fully lazy: iterates a ``Subset`` of this dataset over the uncensored
        window indices, so no dense window array is materialized (``__getitem__``
        slices each window on demand). ``return_id`` is False on this path, so
        each item is ``(sequence, target)`` -- the same batch contract as before.
        """
        uncensored_idx = np.flatnonzero(self.is_censored_array == 0)
        dataset = Subset(self, uncensored_idx.tolist())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_censored_split_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Co-training (Coprog) path. Returns:
            (features_uncensored, targets_uncensored, features_censored, ids_censored)
        ``ids_censored`` is the per-window vehicle id array Coprog groups on.
        """
        uncensored_idx = np.flatnonzero(self.is_censored_array == 0)
        censored_idx = np.flatnonzero(self.is_censored_array == 1)

        target_uncensored = self.label_array[uncensored_idx]
        id_censored = self.id_array[censored_idx]

        features_uncensored = torch.from_numpy(self._build_windows(uncensored_idx))
        features_censored = torch.from_numpy(self._build_windows(censored_idx))
        ids_censored = torch.from_numpy(id_censored).long()

        if not self.return_sequence_label:
            target_uncensored = target_uncensored[:, np.newaxis]
        targets_uncensored = torch.from_numpy(target_uncensored).float()

        return features_uncensored, targets_uncensored, features_censored, ids_censored

    def get_features_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All windows and their labels (NaN targets for censored windows)."""
        features = torch.from_numpy(self._build_windows(np.arange(self._n_windows)))
        targets = self.label_array
        if not self.return_sequence_label:
            targets = targets[:, np.newaxis]
        return features, torch.from_numpy(targets).float()

    def get_censored_lower_bounds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Censored windows with their survival lower bound. Returns:
            (features_censored, ids_censored, lower_bounds_censored)
        ``lower_bounds_censored`` (N, 1) = time observed until end of study; a
        model's predicted RUL for these windows should be >= this value.
        """
        # Same censored-index order as get_censored_split_tensors so the returned
        # lower bounds stay row-aligned with that method's censored features
        # (relied on by CoTrainingEnsemble v2's monotone projection).
        censored_idx = np.flatnonzero(self.is_censored_array == 1)

        features_censored = torch.from_numpy(self._build_windows(censored_idx))
        ids_censored = torch.from_numpy(self.id_array[censored_idx]).long()
        lower_bounds = self.lower_bound_array[censored_idx][:, np.newaxis]
        lower_bounds_censored = torch.from_numpy(lower_bounds).float()

        return features_censored, ids_censored, lower_bounds_censored
