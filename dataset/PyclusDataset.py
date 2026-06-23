import numpy as np
import pandas as pd

import constants.c_mapss_columns as cmapss_col
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader


class PyclusDataset:

    TIME_TO_EVENT_COLUMN = 'time_to_event'
    EVENT_COLUMN = 'event'

    def __init__(self, X, Y, ids, time_grid, feature_names_pyclus, true_rul=None):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.time_grid = time_grid
        self.feature_names_pyclus = feature_names_pyclus
        # Exact scalar RUL (time_to_event) per sample, kept aside from the binarized
        # Y targets so that the ground truth never has to be reconstructed (lossily)
        # from the survival vectors. None for datasets built outside of
        # _to_pyclus_format (e.g. constructed manually without this info).
        self.true_rul = true_rul

    # ============================================================================
    # FACTORY METHODS
    # ============================================================================

    @staticmethod
    def from_cmapss(
            dataset_root: str,
            seed: int | None,
            summarize_features: bool,
            sub_dataset: str = 'FD001',
            max_rul=None,
            norm_type=None,
            cluster_operations=False,
            norm_by_operations=False,
            validation_rate=0.2,
            use_max_rul_on_test=False,
            use_max_rul_on_valid=True,
            percent_of_censored_data: float = 0.0,
            percent_of_broken_data: float | None = None
    ):
        # We don't wan't to normalize if we summarize features with mean, std, slope...
        cluster_operations = False if summarize_features else cluster_operations
        norm_by_operations = False if summarize_features else norm_by_operations

        train_cmapss, test_cmapss, valid_cmapss = CMAPSSLoader.get_datasets(
            dataset_root=dataset_root,
            sub_dataset=sub_dataset,
            sequence_len=1,
            seed=seed,
            max_rul=max_rul,
            return_sequence_label=True,
            norm_type=norm_type,
            cluster_operations=cluster_operations,
            norm_by_operations=norm_by_operations,
            include_cols=None,
            exclude_cols=None,
            return_id=True,
            validation_rate=validation_rate,
            use_only_final_on_test=False,
            use_max_rul_on_test=use_max_rul_on_test,
            use_max_rul_on_valid=use_max_rul_on_valid,
            percent_of_censored_data=percent_of_censored_data,
            percent_of_broken_data=percent_of_broken_data
        )

        return PyclusDataset._transform_datasets_to_pyclus(
            train_cmapss, test_cmapss, valid_cmapss, cmapss_col.ID, cmapss_col.TIME, summarize_features
        )

    # ============================================================================
    # GENERIC METHOD TO TRANSFORM DATASET FOR PYCLUS
    # ============================================================================

    @staticmethod
    def _transform_datasets_to_pyclus(
            train_dataset,
            test_dataset,
            valid_dataset,
            id_col: str,
            time_col: str,
            summarize_features: bool,
            feature_cols=None
    ):
        """
        Transform 3 datasets (train, test, valid) into the pyclus format.

        :param train_dataset: Dataset with a 'df' column
        :param test_dataset: Dataset with a 'df' column
        :param valid_dataset: Dataset with a 'df' column
        :return: Tuple of (train_pyclus, test_pyclus, valid_pyclus)
        """
        if summarize_features:
            # Aggregate the data by unit (id) with summarized features
            PyclusDataset._group_by_id_with_summary_features(
                train_dataset, id_col, time_col, is_test_dataset=False, feature_cols=feature_cols
            )
            PyclusDataset._group_by_id_with_summary_features(
                test_dataset, id_col, time_col, is_test_dataset=True, feature_cols=feature_cols
            )

            if valid_dataset is not None:
                PyclusDataset._group_by_id_with_summary_features(
                    valid_dataset, id_col, time_col, is_test_dataset=False, feature_cols=feature_cols
                )
        else:
            PyclusDataset._keep_last_time_to_event_by_id(train_dataset, id_col, time_col)
            PyclusDataset._keep_last_time_to_event_by_id(test_dataset, id_col, time_col)

            if valid_dataset is not None:
                PyclusDataset._keep_last_time_to_event_by_id(valid_dataset, id_col, time_col)

        # Build the time grid from the train dataset
        time_grid = np.sort(train_dataset.df[PyclusDataset.TIME_TO_EVENT_COLUMN].unique())

        # Convert each dataset to pyclus format
        train_X, train_Y, train_ids, train_features, train_true_rul = PyclusDataset._to_pyclus_format(
            train_dataset, time_grid, time_col, feature_cols
        )
        test_X, test_Y, test_ids, test_features, test_true_rul = PyclusDataset._to_pyclus_format(
            test_dataset, time_grid, time_col, feature_cols
        )

        val_pyclus_dataset = None

        if valid_dataset is not None:
            valid_X, valid_Y, valid_ids, valid_features, valid_true_rul = PyclusDataset._to_pyclus_format(
                valid_dataset, time_grid, time_col, feature_cols
            )

            val_pyclus_dataset = PyclusDataset(
                valid_X, valid_Y, valid_ids, time_grid, valid_features, true_rul=valid_true_rul
            )

        # Create the PyclusDataset instances
        return (
            PyclusDataset(train_X, train_Y, train_ids, time_grid, train_features, true_rul=train_true_rul),
            PyclusDataset(test_X, test_Y, test_ids, time_grid, test_features, true_rul=test_true_rul),
            val_pyclus_dataset
        )

    @staticmethod
    def _keep_last_time_to_event_by_id(
            dataset,
            id_col: str,
            time_col: str,
    ):
        """
        For each unit (id), keep only the row with the last time_to_event RUL value.
        Modifies dataset.df directly.

        :param dataset: Dataset with attribute 'df'
        :param id_col: Name of the id column
        """
        df = dataset.df

        df[PyclusDataset.TIME_TO_EVENT_COLUMN] = df[time_col]
        # If censored then the event didn't occur so 0 otherwise 1
        df[PyclusDataset.EVENT_COLUMN] = df['is_censored'].apply(lambda x: 0 if x == 1 else 1)

        idx_max_time_to_event = df.groupby(id_col)[time_col].idxmax()
        dataset.df = df.loc[idx_max_time_to_event].reset_index(drop=True)

    @staticmethod
    def _group_by_id_with_summary_features(
            dataset,
            id_col: str,
            time_col: str,
            is_test_dataset: bool = False,
            feature_cols=None
    ) -> None:
        """
        Aggregate data by unit (id) by computing summarized features
        (mean, last, slope, max for each column).
        Modifies dataset.df directly.

        :param dataset: Dataset with attributes 'df' and 'feature_cols'
        :param is_test_dataset: Boolean to mark test data
        :param feature_cols: List of columns to summarize (optional)
        """
        if feature_cols is None:
            feature_cols = dataset.feature_cols

        df = dataset.df
        summary_features_by_id_df = df.groupby(id_col).apply(
            lambda grp: PyclusDataset._compute_summary_features_per_id(
                grp, feature_cols, time_col, is_test_dataset
            ),
            include_groups=False
        )
        dataset.df = summary_features_by_id_df


    @staticmethod
    def _compute_summary_features_per_id(
            group: pd.DataFrame,
            feature_cols: list,
            time_col: str,
            is_test_dataset: bool = False
    ) -> pd.Series:
        """
        Compute summarized features (mean, last, slope, max) for a unit.

        :param group: DataFrame grouped by id
        :param feature_cols: Columns to summarize
        :param is_test_dataset: Boolean to mark test data
        :return: pd.Series with the aggregated features
        """
        group = group.sort_values(time_col)
        row = {}

        for col in feature_cols:
            vals = group[col].dropna()
            times = group.loc[group[col].notna(), time_col]

            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_last"] = np.nan
                row[f"{col}_slope"] = np.nan
                row[f"{col}_max"] = np.nan
            else:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_last"] = vals.iloc[-1]
                row[f"{col}_max"] = vals.max()

                # Slope in linear regression if >= 2 values
                if len(vals) >= 2:
                    slope = np.polyfit(times, vals, 1)[0]
                else:
                    slope = 0.0
                row[f"{col}_slope"] = slope

        last_idx = group[time_col].idxmax()
        last_real_rul = group.loc[last_idx, 'real_rul']

        row[PyclusDataset.TIME_TO_EVENT_COLUMN] = group['rul'].max()
        row[PyclusDataset.EVENT_COLUMN] = 1 if (is_test_dataset or last_real_rul <= 0) else 0

        return pd.Series(row)


    @staticmethod
    def _build_survival_targets(
            df: pd.DataFrame,
            time_grid: np.ndarray,
    ) -> list:
        """
        Builds the multi-label targets for survival SSL-PCT.

        For each individual and each time t_j in time_grid:
            - 1   if the individual is known "alive" at t_j       (t_j <= time_to_event)
            - 0   if the event occurred before t_j                (t_j > time_to_event and event == 1)
            - '?' if the status is unknown at t_j (censoring)     (t_j > time_to_event and event == 0)

        :param df: DataFrame with columns time_col and event_col
        :param time_grid: Time grid (1D array)
        :param time_col: Name of the time-to-event column
        :param event_col: Name of the event column
        :return: List[List[Any]] shaped (n_samples, len(time_grid)), values 0/1/'?'
        """
        times = df[PyclusDataset.TIME_TO_EVENT_COLUMN].to_numpy().reshape(-1, 1)
        events = df[PyclusDataset.EVENT_COLUMN].to_numpy().reshape(-1, 1)
        grid = np.asarray(time_grid).reshape(1, -1)

        alive = grid <= times  # (n_samples, n_times)
        unknown_or_dead = np.where(events == 1, 0, np.nan)  # (n_samples, 1)

        targets = np.where(alive, 1, unknown_or_dead).astype(object)
        targets[pd.isna(targets)] = '?'

        return targets.tolist()


    @staticmethod
    def _to_pyclus_format(
            dataset,
            time_grid: np.ndarray,
            time_col: str,
            feature_cols=None
    ) -> tuple:
        """
        Convert a dataset to pyclus format (X, Y, ids, feature_names).

        :param dataset: Dataset with attribute 'df'
        :param time_grid: Time grid
        :param feature_cols: Columns to include (optional)
        :param time_col: Name of the time column
        :return: (X, Y, ids, feature_names, true_rul) ready for pyclus
        """
        df = dataset.df

        if feature_cols is None:
            exclude = {time_col, PyclusDataset.TIME_TO_EVENT_COLUMN, PyclusDataset.EVENT_COLUMN}
            feature_cols = [c for c in df.columns if c not in exclude]

        # X: Features in text format with '?' for missing values
        X = df[feature_cols].to_numpy(dtype=object)
        X[pd.isna(X)] = '?'
        X = X.tolist()

        # Y: Survival targets (0/1/'?')
        Y = PyclusDataset._build_survival_targets(
            df, time_grid
        )

        # Exact ground-truth RUL (time_to_event), kept un-discretized so it does not
        # have to be reconstructed from the binarized Y vector later on.
        true_rul = df[PyclusDataset.TIME_TO_EVENT_COLUMN].to_numpy(dtype=float)

        # Ids and feature names
        ids = df.index.to_numpy()

        return X, Y, ids, feature_cols, true_rul

    # ============================================================================
    # CONVERTING SURVIVAL TARGETS / PREDICTIONS BACK TO A SCALAR RUL
    # ============================================================================

    @staticmethod
    def _clean_survival_vector(y_vec, time_grid) -> tuple:
        """
        Remove unknown ('?') entries from a survival vector and sort it by time.

        :param y_vec: 1D array-like of values in {0, 1, '?'} (ground truth) or floats in [0, 1] (prediction)
        :param time_grid: 1D array-like, same length as y_vec
        :return: (t_valid, y_valid) sorted by time, with the unknown entries removed
        """
        y_arr = np.asarray(y_vec, dtype=object)
        y_arr = np.where(y_arr == '?', np.nan, y_arr).astype(float)
        t_arr = np.asarray(time_grid, dtype=float)

        valid = ~np.isnan(y_arr)
        t_valid = t_arr[valid]
        y_valid = y_arr[valid]

        order = np.argsort(t_valid)
        return t_valid[order], y_valid[order]

    @staticmethod
    def survival_vector_to_rul(
            y_vec,
            time_grid,
            method: str = 'threshold',
            threshold: float = 0.5,
            enforce_monotonic: bool = True
    ) -> float:
        """
        Convert ONE survival vector (ground truth or model prediction) into a scalar RUL estimate.

        Two extraction methods are available:
          - 'threshold': RUL = last time point of the (non-increasing) survival curve where
                         P(alive) >= threshold. This is EXACT for ground-truth vectors
                         (pure step functions of 1s then 0s) and behaves like a
                         "median survival time" estimator for continuous predictions.
          - 'auc': RUL = area under the survival curve (expected lifetime), via the trapezoidal
                   rule. Uses all the information in the curve but is more sensitive to noise.

        :param y_vec: 1D array-like of values in {0, 1, '?'} (target) or floats in [0, 1] (prediction)
        :param time_grid: 1D array-like, same length as y_vec
        :param method: 'threshold' or 'auc'
        :param threshold: probability threshold used by the 'threshold' method
        :param enforce_monotonic: if True, force the curve to be non-increasing (cumulative min)
                                   before extracting the RUL. Recommended for raw model predictions,
                                   which are not guaranteed to be monotonic.
        :return: scalar RUL estimate (np.nan if nothing is known about the sample)
        """
        t_valid, y_valid = PyclusDataset._clean_survival_vector(y_vec, time_grid)

        if len(t_valid) == 0:
            return np.nan

        if enforce_monotonic:
            y_valid = np.minimum.accumulate(y_valid)

        if method == 'threshold':
            alive_idx = np.where(y_valid >= threshold)[0]
            if len(alive_idx) == 0:
                return float(t_valid[0])
            return float(t_valid[alive_idx[-1]])
        elif method == 'auc':
            if len(t_valid) < 2:
                return float(t_valid[0] * y_valid[0])
            return float(np.trapz(y_valid, t_valid))
        else:
            raise ValueError(f"Unknown method '{method}', expected 'threshold' or 'auc'")

    @staticmethod
    def survival_targets_to_rul(Y, time_grid, **kwargs) -> np.ndarray:
        """
        Convert a full batch of survival vectors (targets Y or model predictions Y_hat)
        into an array of scalar RUL estimates.

        :param Y: array-like of shape (n_samples, n_times), values in {0, 1, '?'} or in [0, 1]
        :param time_grid: 1D array-like of shape (n_times,)
        :param kwargs: forwarded to survival_vector_to_rul (method, threshold, enforce_monotonic)
        :return: np.ndarray of shape (n_samples,)
        """
        return np.array([
            PyclusDataset.survival_vector_to_rul(row, time_grid, **kwargs)
            for row in Y
        ])

    def to_rul(self, **kwargs) -> np.ndarray:
        """
        Convenience instance method returning the ground-truth RUL values.

        If the dataset was built via _to_pyclus_format (the normal path), self.true_rul
        holds the EXACT scalar time_to_event for each sample and is returned directly
        (no information loss). Otherwise, it falls back to reconstructing an estimate
        from the binarized self.Y survival targets (see survival_targets_to_rul),
        which is only accurate up to the resolution of self.time_grid.
        """
        if self.true_rul is not None:
            return np.asarray(self.true_rul, dtype=float)
        return PyclusDataset.survival_targets_to_rul(self.Y, self.time_grid, **kwargs)