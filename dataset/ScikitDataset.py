from typing import Optional

import numpy as np
import pandas as pd
from numpy import ndarray

import constants.c_mapss_columns as cmapss_col
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader


class ScikitDataset:
    TIME_TO_EVENT_COLUMN = 'time_to_event'
    EVENT_COLUMN = 'event'

    def __init__(self, X: ndarray, Y: ndarray, ids: ndarray, feature_names: list[str], rul: ndarray | None=None):
        """

        :param X: features
        :param Y: targets
        :param ids: list of ids for each unit
        :param feature_names: the name of the features
        :param rul: rul for the test dataset
        """
        self.X = X
        self.Y = Y
        self.ids = ids
        self.feature_names_pyclus = feature_names
        self.rul = rul

    # ============================================================================
    # FACTORY METHODS
    # ============================================================================

    @staticmethod
    def from_cmapss(
            dataset_root: str,
            sub_dataset: str,
            max_rul: int,
            seed: int | None,
            summarize_features: bool,
            include_cols: Optional[list[str]],
            exclude_cols: Optional[list[str]],
            norm_type="z-score",
            cluster_operations=True,
            norm_by_operations=True,
            use_max_rul_on_test=False,
            use_max_rul_on_valid=True,
            percent_of_censored_data: float = 0.,
            percent_of_broken_data: float | None = None
    ):
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
            include_cols=include_cols,
            exclude_cols=exclude_cols,
            return_id=True,
            validation_rate=0.,
            use_only_final_on_test=False,
            use_max_rul_on_test=use_max_rul_on_test,
            use_max_rul_on_valid=use_max_rul_on_valid,
            percent_of_censored_data=percent_of_censored_data,
            percent_of_broken_data=percent_of_broken_data
        )

        return ScikitDataset._transform_datasets_to_scikit(
            train_cmapss, test_cmapss, valid_cmapss, cmapss_col.ID, cmapss_col.TIME, summarize_features
        )

    @staticmethod
    def from_scania(data_module) -> tuple["ScikitDataset", "ScikitDataset", "ScikitDataset | None"]:
        """Build train/test/valid ScikitDatasets from a ScaniaDataModule for RSF.

        A Random Survival Forest cannot consume multivariate time series, so every
        readout row is treated as its own individual (no feature summarization, no
        keep-last-row). For a row at ``time_step = t`` of a vehicle:

        - ``Time``   = ``time_step`` (elapsed observed time). RSF predicts the total
          lifetime as ``∫ survival_fn``; RUL = ``predicted_total - Time``.
        - ``Status`` = ``True`` if the vehicle failed (``is_censored == 0``) else ``False``.
        - ``rul``    = ``length_of_study_time_step - time_step`` (true RUL for failures).

        Training keeps all rows (RSF handles censoring via ``Status``). The test set
        is restricted to uncensored (failure) rows so a true RUL exists for evaluation.

        :param data_module: A ``ScaniaDataModule`` (setup is triggered here if needed).
        :return: ``(train, test, valid)`` ScikitDatasets; ``valid`` is ``None`` if the
            module has no validation split.
        """
        # Local import to avoid any package-init import cycle (scania depends on dataset).
        from constants.scania_component_x_columns import VEHICLE_ID, TIME_STEP
        from scania.dataset.ScaniaDataset import IS_CENSORED, RUL_LOWER_BOUND

        data_module.setup()
        feature_cols = list(data_module.feature_cols)

        train = ScikitDataset._scania_split_to_scikit(
            data_module.train_set, feature_cols, IS_CENSORED, TIME_STEP, VEHICLE_ID,
            RUL_LOWER_BOUND, keep_uncensored_only=False,
        )
        test = ScikitDataset._scania_split_to_scikit(
            data_module.test_set, feature_cols, IS_CENSORED, TIME_STEP, VEHICLE_ID,
            RUL_LOWER_BOUND, keep_uncensored_only=True,
        )
        valid = None
        if data_module.val_set is not None:
            valid = ScikitDataset._scania_split_to_scikit(
                data_module.val_set, feature_cols, IS_CENSORED, TIME_STEP, VEHICLE_ID,
                RUL_LOWER_BOUND, keep_uncensored_only=False,
            )

        return train, test, valid

    @staticmethod
    def _scania_split_to_scikit(
            scania_dataset,
            feature_cols: list[str],
            is_censored_col: str,
            time_col: str,
            id_col: str,
            rul_col: str,
            keep_uncensored_only: bool,
    ) -> "ScikitDataset":
        """Turn one pre-processed ScaniaDataset split into a ScikitDataset.

        Each row of ``scania_dataset.df`` (features already z-score-normalized, RUL
        columns already computed) becomes one survival sample.

        :param scania_dataset: A built ``ScaniaDataset`` exposing ``.df``.
        :param feature_cols: Feature columns to use as ``X``.
        :param is_censored_col: Name of the censoring flag column (1 = censored).
        :param time_col: Name of the elapsed-time column used as survival ``Time``.
        :param id_col: Name of the per-individual id column.
        :param rul_col: Name of the column holding the true RUL / survival lower bound.
        :param keep_uncensored_only: If True, drop censored rows (test evaluation).
        :return: A ``ScikitDataset`` with ``X``, structured ``Y``, ``ids`` and ``rul``.
        """
        df = scania_dataset.df.copy()
        if keep_uncensored_only:
            df = df[df[is_censored_col] == 0].reset_index(drop=True)

        X = df[feature_cols].to_numpy(dtype=object)  # parity with from_cmapss
        status = (df[is_censored_col] == 0).to_numpy()  # True = event (failure) observed
        time = df[time_col].to_numpy(dtype=np.float64)
        Y = np.array(
            list(zip(status, time)),
            dtype=[('Status', '?'), ('Time', '<f8')],
        )
        ids = df[id_col].to_numpy()
        rul = df[rul_col].to_numpy(dtype=np.float64)

        return ScikitDataset(X, Y, ids, feature_cols, rul=rul)

    # ============================================================================
    # GENERIC METHOD TO TRANSFORM DATASET FOR PYCLUS
    # ============================================================================

    @staticmethod
    def _transform_datasets_to_scikit(
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
            ScikitDataset._group_by_id_with_summary_features(
                train_dataset, id_col, time_col, is_test_dataset=False, feature_cols=feature_cols
            )
            ScikitDataset._group_by_id_with_summary_features(
                test_dataset, id_col, time_col, is_test_dataset=True, feature_cols=feature_cols
            )

            if valid_dataset is not None:
                ScikitDataset._group_by_id_with_summary_features(
                    valid_dataset, id_col, time_col, is_test_dataset=False, feature_cols=feature_cols
                )
        else:
            ScikitDataset._create_time_to_event_and_event_column(train_dataset, time_col)
            ScikitDataset._create_time_to_event_and_event_column(test_dataset, time_col)

            if valid_dataset is not None:
                ScikitDataset._create_time_to_event_and_event_column(valid_dataset, time_col)

            ScikitDataset._keep_last_time_to_event_by_id(train_dataset, id_col)
            ScikitDataset._keep_last_time_to_event_by_id(test_dataset, id_col)

            if valid_dataset is not None:
                ScikitDataset._keep_last_time_to_event_by_id(valid_dataset, id_col)

        train_X, train_Y, train_ids, train_feature_cols = ScikitDataset._to_scikit_format(train_dataset, time_col,
                                                                                          feature_cols)
        test_X, test_Y, test_ids, test_feature_cols = ScikitDataset._to_scikit_format(test_dataset, time_col,
                                                                                      feature_cols)

        val_scikit_dataset = None

        if valid_dataset is not None:
            valid_X, valid_Y, valid_ids, valid_feature_cols = ScikitDataset._to_scikit_format(test_dataset, time_col,
                                                                                              feature_cols)

            val_scikit_dataset = ScikitDataset(valid_X, valid_Y, valid_ids, valid_feature_cols)

        # Create the ScikitDataset instances
        return (
            ScikitDataset(train_X, train_Y, train_ids, train_feature_cols), # Train
            ScikitDataset(test_X, test_Y, test_ids, test_feature_cols, test_dataset.final_rul), # Test
            val_scikit_dataset # Valid
        )

    @staticmethod
    def _create_time_to_event_and_event_column(dataset, time_col: str):
        df = dataset.df

        df[ScikitDataset.TIME_TO_EVENT_COLUMN] = df[time_col]
        # If censored then the event didn't occur so 0 otherwise 1
        df[ScikitDataset.EVENT_COLUMN] = df['is_censored'].apply(lambda x: 0 if x == 1 else 1)

        dataset.df = df


    @staticmethod
    def _keep_last_time_to_event_by_id(
            dataset,
            id_col: str,
    ):
        """
        For each unit (id), keep only the row with the last time_to_event RUL value.
        Modifies dataset.df directly.

        :param dataset: Dataset with attribute 'df'
        :param id_col: Name of the id column
        """
        df = dataset.df
        idx_max_time_to_event = df.groupby(id_col)[ScikitDataset.TIME_TO_EVENT_COLUMN].idxmax()
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
            lambda grp: ScikitDataset._compute_summary_features_per_id(
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
        #
        # row[ScikitDataset.TIME_TO_EVENT_COLUMN] = group['rul'].max()
        # row[ScikitDataset.EVENT_COLUMN] = 1 if (is_test_dataset or last_real_rul <= 0) else 0

        row[ScikitDataset.TIME_TO_EVENT_COLUMN] = group[time_col].max()
        row[ScikitDataset.EVENT_COLUMN] = 1 if (is_test_dataset or last_real_rul <= 0) else 0

        return pd.Series(row)


    @staticmethod
    def _to_scikit_format(
            dataset,
            time_col: str,
            feature_cols: list[str] | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, list[str]]:
        df = dataset.df

        if feature_cols is None:
            exclude = {time_col, ScikitDataset.TIME_TO_EVENT_COLUMN, ScikitDataset.EVENT_COLUMN}
            feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols].to_numpy(dtype=object)

        Y = df[[ScikitDataset.EVENT_COLUMN, ScikitDataset.TIME_TO_EVENT_COLUMN]].to_numpy(dtype=object)

        Y = np.array(
            [(bool(status), time) for status, time in Y],
            dtype=[('Status', '?'), ('Time', '<f8')]
        )

        # Ids and feature names
        ids = df.index.to_numpy()

        return X, Y, ids, feature_cols
