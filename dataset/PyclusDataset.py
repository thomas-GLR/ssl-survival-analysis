import numpy as np
import pandas as pd

import constants.c_mapss_columns as cmapss_col
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader


class PyclusDataset:

    TIME_TO_EVENT_COLUMN = 'time_to_event'
    EVENT_COLUMN = 'event'

    def __init__(self, X, Y, ids, time_grid, feature_names_pyclus):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.time_grid = time_grid
        self.feature_names_pyclus = feature_names_pyclus

    # ============================================================================
    # FACTORY METHODS
    # ============================================================================

    @staticmethod
    def from_cmapss(
            dataset_root: str,
            sub_dataset: str = 'FD001',
            max_rul=None,
            validation_rate=0.2,
            use_max_rul_on_test=False,
            use_max_rul_on_valid=True,
            percent_of_censored_data: float = 0.0,
            percent_of_broken_data: float | None = None
    ):
        train_cmapss, test_cmapss, valid_cmapss = CMAPSSLoader.get_datasets(
            dataset_root=dataset_root,
            sub_dataset=sub_dataset,
            sequence_len=1,
            max_rul=max_rul,
            return_sequence_label=True,
            norm_type=None,
            cluster_operations=False,
            norm_by_operations=False,
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
            train_cmapss, test_cmapss, valid_cmapss, cmapss_col.ID, cmapss_col.TIME
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
            feature_cols=None
    ):
        """
        Transform 3 datasets (train, test, valid) into the pyclus format.

        :param train_dataset: Dataset with a 'df' column
        :param test_dataset: Dataset with a 'df' column
        :param valid_dataset: Dataset with a 'df' column
        :return: Tuple of (train_pyclus, test_pyclus, valid_pyclus)
        """
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

        # Build the time grid from the train dataset
        time_grid = np.sort(train_dataset.df[PyclusDataset.TIME_TO_EVENT_COLUMN].unique())

        # Convert each dataset to pyclus format
        train_X, train_Y, train_ids, train_features = PyclusDataset._to_pyclus_format(
            train_dataset, time_grid, time_col, feature_cols
        )
        test_X, test_Y, test_ids, test_features = PyclusDataset._to_pyclus_format(
            test_dataset, time_grid, time_col, feature_cols
        )

        val_pyclus_dataset = None

        if valid_dataset is not None:
            valid_X, valid_Y, valid_ids, valid_features = PyclusDataset._to_pyclus_format(
                valid_dataset, time_grid, time_col, feature_cols
            )

            val_pyclus_dataset = PyclusDataset(valid_X, valid_Y, valid_ids, time_grid, valid_features)

        # Create the PyclusDataset instances
        return (
            PyclusDataset(train_X, train_Y, train_ids, time_grid, train_features),
            PyclusDataset(test_X, test_Y, test_ids, time_grid, test_features),
            val_pyclus_dataset
        )

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
        :return: (X, Y, ids, feature_names) ready for pyclus
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

        # Ids and feature names
        ids = df.index.to_numpy()

        return X, Y, ids, feature_cols
