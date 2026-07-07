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
from torch.utils.data import Dataset, DataLoader, TensorDataset

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


class ScaniaDataset(Dataset):
    def __init__(
            self,
            data_df: pd.DataFrame,
            sequence_len: int,
            feature_cols: list[str] | None = None,
            norm_type: str | None = None,
            norm_params: np.ndarray | None = None,
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
            Pre-computed normalization params of shape ``(n_features, 2)`` =
            ``[mean, std]``. If ``None`` and ``norm_type`` is set, they are
            computed from this dataframe (use this only for the train split and
            pass ``train.norm_params`` to val/test to avoid leakage).
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
        self.return_sequence_label = return_sequence_label
        self.return_id = return_id
        self.only_final = only_final

        # Sequence arrays populated by _gen_sequence().
        self.sequence_array: np.ndarray | None = None
        self.label_array: np.ndarray | None = None
        self.lower_bound_array: np.ndarray | None = None
        self.id_array: np.ndarray | None = None
        self.is_censored_array: np.ndarray | None = None

        self.count_rul()

        if self.norm_type:
            self._normalization()

        self._gen_sequence()

    def __len__(self) -> int:
        return len(self.sequence_array)

    def __getitem__(self, i):
        """
        :return: (sequence, target[, id])
            sequence: FloatTensor (sequence_len, n_features)
            target:   FloatTensor (1,) last-step RUL, or (sequence_len,) if
                      return_sequence_label. NaN for censored windows.
        """
        seq = torch.FloatTensor(self.sequence_array[i])
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
    # Normalization (z-score)
    # ------------------------------------------------------------------ #
    def _normalization(self) -> None:
        df = self.df
        cols = self.feature_cols
        df[cols] = df[cols].astype(np.float64)

        if self.norm_params is None:
            self.norm_params = self._gen_norm_params()

        mean = self.norm_params[:, 0]
        std = self.norm_params[:, 1]
        std_safe = np.where(std == 0, 1.0, std)
        df[cols] = (df[cols].to_numpy() - mean) / std_safe
        self.df = df

    def _gen_norm_params(self) -> np.ndarray:
        vals = self.df[self.feature_cols].to_numpy(dtype=np.float64)
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)
        return np.stack((mean, std), axis=1)

    # ------------------------------------------------------------------ #
    # Sequence generation
    # ------------------------------------------------------------------ #
    def _gen_sequence(self) -> None:
        """Build the sliding-window arrays.

        Vectorized: instead of filtering the dataframe per vehicle (an
        O(vehicles x rows) rescan) and looping window-by-window, we sort once,
        find contiguous per-vehicle group boundaries, and generate every
        long-vehicle window in one shot with ``sliding_window_view`` masked so
        no window spans two vehicles. Vehicles shorter than ``seq_len`` are
        left-edge-padded in a small loop (there are few of them).

        If ``self.only_final`` is set, the long-vehicle branch is
        additionally filtered to keep, per vehicle, only the single window
        ending at that vehicle's last row (short vehicles already yield
        exactly one window each, so they need no extra handling).
        """
        n_feat = len(self.feature_cols)
        seq_len = self.sequence_len

        # Sort once so each vehicle's rows are contiguous and time-ordered.
        df = self.df.sort_values([VEHICLE_ID, TIME_STEP]).reset_index(drop=True)

        vehicules_ids = df[VEHICLE_ID].to_numpy()
        features = df[self.feature_cols].to_numpy(dtype=np.float64)
        rul = df[RUL].to_numpy(dtype=np.float64)
        rul_lower_bound = df[RUL_LOWER_BOUND].to_numpy(dtype=np.float64)
        censored = df[IS_CENSORED].to_numpy()
        rows_number = len(df)

        # Contiguous group boundaries (rows are grouped by vehicle after sort).
        starts = np.concatenate(([0], np.flatnonzero(np.diff(vehicules_ids) != 0) + 1))
        counts = np.diff(np.concatenate((starts, [rows_number])))

        seq_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        lb_chunks: list[np.ndarray] = []
        id_chunks: list[np.ndarray] = []
        cens_chunks: list[np.ndarray] = []

        # --- Long vehicles (count >= seq_len): all windows at once ---------- #
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
                # sliding_window_view -> (num_positions, n_feat, seq_len); index
                # the valid starts (single copy) then reorder to (M, seq_len, n_feat).
                sw = np.lib.stride_tricks.sliding_window_view(features, seq_len, axis=0)
                seq_chunks.append(sw[sequence_group].transpose(0, 2, 1))

                if self.return_sequence_label:
                    rul_sw = np.lib.stride_tricks.sliding_window_view(rul, seq_len, axis=0)
                    label_chunks.append(rul_sw[sequence_group])
                else:
                    label_chunks.append(rul[last])

                lb_chunks.append(rul_lower_bound[last])
                id_chunks.append(vehicules_ids[sequence_group])
                cens_chunks.append(censored[sequence_group])

        # --- Short vehicles (count < seq_len): one edge-padded window each --- #
        for start, count in zip(starts[counts < seq_len], counts[counts < seq_len]):
            sl = slice(start, start + count)
            pad = ((seq_len - count, 0), (0, 0))
            seq_chunks.append(np.pad(features[sl], pad, "edge")[np.newaxis])

            if self.return_sequence_label:
                label_chunks.append(np.pad(rul[sl], (seq_len - count, 0), "edge")[np.newaxis])
            else:
                label_chunks.append(rul[start + count - 1: start + count])

            lb_chunks.append(rul_lower_bound[start + count - 1: start + count])
            id_chunks.append(vehicules_ids[start: start + 1])
            cens_chunks.append(censored[start: start + 1])

        self.sequence_array = np.concatenate(seq_chunks, axis=0)
        self.label_array = np.concatenate(label_chunks, axis=0)
        self.lower_bound_array = np.concatenate(lb_chunks, axis=0)
        self.id_array = np.concatenate(id_chunks, axis=0)
        # is_censored is constant within a vehicle.
        self.is_censored_array = np.concatenate(cens_chunks, axis=0).astype(int)

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
        """Supervised path: DataLoader over uncensored windows only, yields (x, y)."""
        mask_uncensored = self.is_censored_array == 0

        feat = self.sequence_array[mask_uncensored]
        target = self.label_array[mask_uncensored]

        features = torch.from_numpy(feat).float()

        if not self.return_sequence_label:
            target = target[:, np.newaxis]
        targets = torch.from_numpy(target).float()

        dataset = TensorDataset(features, targets)
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
        mask_censored = self.is_censored_array == 1
        mask_uncensored = ~mask_censored

        feat_uncensored = self.sequence_array[mask_uncensored]
        target_uncensored = self.label_array[mask_uncensored]

        feat_censored = self.sequence_array[mask_censored]
        id_censored = self.id_array[mask_censored]

        features_uncensored = torch.from_numpy(feat_uncensored).float()
        features_censored = torch.from_numpy(feat_censored).float()
        ids_censored = torch.from_numpy(id_censored).long()

        if not self.return_sequence_label:
            target_uncensored = target_uncensored[:, np.newaxis]
        targets_uncensored = torch.from_numpy(target_uncensored).float()

        return features_uncensored, targets_uncensored, features_censored, ids_censored

    def get_features_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All windows and their labels (NaN targets for censored windows)."""
        features = torch.from_numpy(self.sequence_array).float()
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
        mask_censored = self.is_censored_array == 1

        features_censored = torch.from_numpy(self.sequence_array[mask_censored]).float()
        ids_censored = torch.from_numpy(self.id_array[mask_censored]).long()
        lower_bounds = self.lower_bound_array[mask_censored][:, np.newaxis]
        lower_bounds_censored = torch.from_numpy(lower_bounds).float()

        return features_censored, ids_censored, lower_bounds_censored
