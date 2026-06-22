import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader, TensorDataset

ID = 'id'
TIME = 'time'


class CMAPSSDataset(Dataset):
    """
    Most of the code come from the paper "Building of transformer-based RUL predictors supported by explainability
    techniques: Application on real industrial datasets" where the repository can be fined at this address : https://github.com/DintenR/Transformer-based-RUL-predictors
    """
    # [op1, op2, op3]
    OPERATION_COLS = ['op%d' % i for i in range(1, 3 + 1)]
    # [s1, s2, ..., s21]
    SENSOR_COLS = ['s%d' % i for i in range(1, 21 + 1)]
    # [id, time, op1, op2, op3, s1, s2, ..., s21]
    DATASET_COLS = [ID, TIME] + OPERATION_COLS + SENSOR_COLS

    def __init__(
            self,
            data_df,
            sequence_len=1,
            final_rul=None,
            norm_params=None,
            norm_type=None,
            max_rul=None,
            only_final=False,
            return_sequence_label=False,
            cluster_operations=False,
            norm_by_operations=False,
            include_cols=None,
            exclude_cols=None,
            return_id=False,
            kmeans_model=None,
            percent_of_censored_data: float=0.0,
            percent_of_broken_data: float | None=None):
        """

            C-MAPSS Dataset, create pytorch Dataset by pd.Dataframe use original txt file,
            PHM08 Challenge Dataset is also supported.
            C-MAPSS and PHM08 Dataset download: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

        :param data_df:
            Required, pd.Dataframe from 'train_FD00X.txt/test_FD00X.txt/train.txt/test.txt.

        :param sequence_len:
            sequence length of time window pre-progress, default is 1.
            e.g.: a unit has 200 cycles, seq_length=50 means generate data as
                data = [[cycle 0 - 50],
                        [cycle 1 - 51],
                        ...
                        [cycle 150-199]]
                        # shape(sequences_num, sequence_len, features_num) = shape(150, 50, 24)

        :param final_rul:
            An list or nparray denote the RUL for the last time cycle of each unit,
            which are all set to 0 for the training set, default is None.

        :param norm_params:
            An numpy array to set the normalization params manually, default is None.
            if not provide, it will calculate the params using provided data.
            this params can be use in the situation that normalize the training and test dataset together.
            the norm_params is shaped as (sensor, params),
            sensor represents the i-th sensor and params represents the params of the normalization methods,
                for min-max normalization, the value is [min, max].
                for z-score normalization, the value is [μ, σ].
            if cluster_operations and norm_by_operation are both set to True, the norm_params is shape as
            (op_type, sensor, params), op_type represents the j-th operation.

        :param norm_type:
            A string represents the normalization type, '0-1', '-1-1' or 'z-score', default is None.

        :param max_rul:
            Number, a piece-wise RUL function on RUL, RUL exceeding max_rul will be set to max_rul.
            the read RUL will be store in self.df['real_rul'].

        :param only_final:
            only use the last time window's data and label, use in test sets, default False

        :param return_sequence_label:
            return all RUL instead of only last RUL, default is False.

        :param cluster_operations:
            implement a K-Means cluster on three operational settings,
            an new column named 'op_type' will insert to the self.df, but not add to feature columns, default is False

        :param norm_by_operations:
            if is cluster operational settings,
            set this to True to normalize data by operation types, default is False.

        :param include_cols:
            use include_cols as features, e.g. ['s1', 's2'], default is None,
            means use all operations and sensors is feature.

        :param exclude_cols:
            exclude features, e.g. ['op3', 's2', 's3'], default is None

        :param kmeans_model:
            A KMeans model already fit to have the same cluster than the train dataset for valid and test

        :param percent_of_censored_data:
            percentage of censored data, default is 0.0, which means no censored data.
            The percentage of censored data dosen't apply to test dataset

        :param percent_of_broken_data:
            This is the percent of damage until the data is censored.
            Default is None which means the broken percent is random

        :param return_id:
            return unit id, default is False

        """
        super().__init__()
        assert isinstance(data_df, pd.DataFrame), 'data_df need pd.DataFrame'
        assert len(data_df.columns) >= 26, 'Invalid Dataframe input (columns < 26)'

        self.has_cluster_operations = False
        self.has_normalization = False
        self.has_gen_sequence = False
        self.has_count_rul = False

        # set self.df
        self.df = data_df
        if len(self.df.columns) >= 26:
            self.df = self.df.drop([26, 27], axis=1)
        self.df.columns = CMAPSSDataset.DATASET_COLS
        # sequence_len
        assert sequence_len > 0, 'Need sequence_len > 0, got:' + str(sequence_len)
        self.sequence_len = sequence_len

        # feature cols define
        if include_cols is not None:
            self.feature_cols = include_cols
        else:
            self.feature_cols = CMAPSSDataset.OPERATION_COLS + CMAPSSDataset.SENSOR_COLS

        if exclude_cols is not None:
            for v in exclude_cols:
                if v in self.feature_cols:
                    self.feature_cols.remove(v)

        # final rul
        if final_rul is None:
            self.final_rul = np.zeros(self.df['id'].nunique())
        else:
            self.final_rul = final_rul

        # norm type
        self.norm_type = None
        # norm params
        self.norm_params = None
        # normalization by operations
        self.norm_by_operations = None

        self._set_norm_params(norm_type, norm_params, norm_by_operations)

        # max rul
        if max_rul is None:
            max_rul = 999999
        self.max_rul = max_rul

        # only final
        self.only_final = only_final

        self.percent_of_censored_data = percent_of_censored_data
        self.percent_of_broken_data = percent_of_broken_data

        # We need to create the censored data before count_rul(), normalization() and clustering()
        if percent_of_censored_data > 0:
            self._generate_censored_data()
        else:
            self.df['is_censored'] = 0

        # compute RUL
        self.count_rul()

        # cluster on operations
        self.cluster_operations = cluster_operations

        # return sequence_label
        self.return_sequence_label = return_sequence_label

        # return id
        self.return_id = return_id

        # sequence data
        self.sequence_array = None
        self.label_array = None
        self.id_array = None

        self.kmeans_model = None

        if self.cluster_operations:
            self._clustering_operations(kmeans_model)

        if self.norm_type:
            self._normalization()

        self._gen_sequence()

    def __len__(self):
        return len(self.sequence_array)

    def __getitem__(self, i):
        """
        :param i: get i_th data
        :return: sequence, target
            sequence: tensor([time, setting1, ... , sensor21])
            target: tensor([rul])
        """
        l = [torch.FloatTensor(self.sequence_array[i]), torch.FloatTensor([self.label_array[i]])]
        if self.return_id:
            l.append(torch.LongTensor([self.id_array[i]]))
        return tuple(l)

    def get_data_loader(self, loader_kwargs: dict) -> DataLoader:
        """
        :param loader_kwargs:
            kwargs pass to all DataLoader
        """

        return DataLoader(self, **loader_kwargs)

    def get_censored_split_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the features, targets, and ids separately for censored data (is_censored=1)
        and uncensored data (is_censored=0).
        """
        # Identify ids marked as censored in the DataFrame
        censored_ids = self.df[self.df['is_censored'] == 1]['id'].unique()

        # Create a mask to filter sequences/windows generated
        mask_censored = np.isin(self.id_array, censored_ids)
        mask_uncensored = ~mask_censored

        feat_uncensored = self.sequence_array[mask_uncensored]
        target_uncensored = self.label_array[mask_uncensored]

        feat_censored = self.sequence_array[mask_censored]
        id_censored = self.id_array[mask_censored]  # <-- Extract IDs for censored data

        features_uncensored = torch.from_numpy(feat_uncensored).float()
        features_censored = torch.from_numpy(feat_censored).float()
        ids_censored = torch.from_numpy(id_censored).long()  # <-- Convert to Tensor

        # Adjust the shape of the target to correspond to (N, 1)
        if not self.return_sequence_label:
            target_uncensored = target_uncensored[:, np.newaxis]

        targets_uncensored = torch.from_numpy(target_uncensored).float()

        return features_uncensored, targets_uncensored, features_censored, ids_censored

    def get_features_targets(self):
        features_tensor = torch.from_numpy(self.sequence_array).float()

        if not self.return_sequence_label:
            targets = self.label_array[:, np.newaxis]
        else:
            targets = self.label_array

        targets_tensor = torch.from_numpy(targets).float()

        return features_tensor, targets_tensor

    def get_data_loader_without_censored_data(
            self,
            batch_size: int,
            shuffle: bool=False,
            is_model_cnn: bool=False,
    ) -> DataLoader:
        # Identify ids marked as censored in the DataFrame
        censored_ids = self.df[self.df['is_censored'] == 1]['id'].unique()

        # Create a mask to filter sequences/windows generated
        # self.id_array contain the ID that correspond to each input of self.sequence_array
        mask_censored = np.isin(self.id_array, censored_ids)
        mask_uncensored = ~mask_censored

        feat_uncensored = self.sequence_array[mask_uncensored]
        target_uncensored = self.label_array[mask_uncensored]

        features_uncensored = torch.from_numpy(feat_uncensored).float()

        if is_model_cnn:
            # For conv1d the features (channels) should be in second place
            features_uncensored = features_uncensored.permute(0, 2, 1)

        if not self.return_sequence_label:
            target_uncensored = target_uncensored[:, np.newaxis]

        targets_uncensored = torch.from_numpy(target_uncensored).float()

        dataset = TensorDataset(features_uncensored, targets_uncensored)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    def count_rul(self):
        df = self.df

        final_rul = self.final_rul
        max_rul = self.max_rul
        time_series = df.groupby('id').size()
        rul_array = time_series.values + final_rul
        rul_df = pd.DataFrame({
            'id': time_series.index,
            'rul': rul_array,
            'real_rul': rul_array
        })
        df = pd.merge(df, rul_df)
        df['real_rul'] = df.apply(lambda x: x['rul'] - x['time'], axis=1)
        df['rul'] = df.apply(lambda x: max_rul if max_rul < x['real_rul'] else x['real_rul'], axis=1)

        # For the censored data we can't calculate the RUL as we don't know when the event occurs
        df.loc[df['is_censored'] == 1, ['rul', 'real_rul']] = np.nan

        self.df = df
        self.has_count_rul = True

    def _set_norm_params(self, norm_type, norm_params, norm_by_operations):
        # norm_params
        self.norm_params = norm_params

        # norm_type
        assert norm_type is None or norm_type in ['z-score', '0-1', '-1-1']
        self.norm_type = norm_type

        # norm by operations
        self.norm_by_operations = norm_by_operations

    def _clustering_operations(self, kmeans_model=None):
        df = self.df
        features = df[['op1', 'op2', 'op3']].values

        if kmeans_model is None:
            self.kmeans_model = KMeans(n_clusters=6, random_state=1).fit(features)
        else:
            self.kmeans_model = kmeans_model

        op_types = self.kmeans_model = kmeans_model.predict(features)
        df.insert(2, 'op_type', op_types)

        self.df = df
        self.has_cluster_operations = True

    def _normalization(self):
        if self.norm_type is None:
            return

        if self.cluster_operations and self.norm_by_operations and not self.has_cluster_operations:
            raise RuntimeError('need cluster operations before normalization when norm_by_operations is True')

        # TODO specific normalization cols
        df = self.df
        norm_cols = self.feature_cols
        norm_type = self.norm_type
        norm_by_operations = self.norm_by_operations
        if self.norm_params is None:
            self.norm_params = self._gen_norm_params(norm_type, norm_by_operations)
        norm_params = self.norm_params
        if norm_type == '0-1':
            min_norm, max_norm = 0, 1
            self.df = self._min_max_normalization(norm_cols, norm_params, min_norm, max_norm)
        if norm_type == '-1-1':
            min_norm, max_norm = -1, 1
            self.df = self._min_max_normalization(norm_cols, norm_params, min_norm, max_norm)
        if norm_type == 'z-score':
            self.df = self._z_score_normalization(norm_cols, norm_params)
        self.has_normalization = True

    def _min_max_normalization(self, cols, norm_params, min_norm, max_norm):
        df = self.df

        if len(norm_params.shape) == 2:
            for col_i, col in enumerate(cols):
                min_v, max_v = norm_params[col_i]
                if max_v == min_v:
                    df[col] = (min_norm + max_norm) / 2  # median
                else:
                    df[col] = (((max_norm - min_norm) * (df[col].values - min_v)) / (max_v - min_v)) + min_norm
        elif len(norm_params.shape) == 3:
            op_list = df['op_type'].unique()
            op_list.sort()
            for op_i, op in enumerate(op_list):
                for col_i, col in enumerate(cols):
                    min_v, max_v = norm_params[op_i, col_i]
                    if max_v == min_v:
                        df.loc[df['op_type'] == op, col] = (min_norm + max_norm) / 2  # median
                    else:
                        values = df[df['op_type'] == op][col].values
                        df.loc[df['op_type'] == op, col] = (((max_norm - min_norm) * (values - min_v)) / (
                                max_v - min_v)) + min_norm

        else:
            raise ValueError('norm_params shape error')
        return df

    def _z_score_normalization(self, cols, norm_params):
        df = self.df

        if len(norm_params.shape) == 2:
            for col_i, col in enumerate(cols):
                mean, standard = norm_params[col_i]
                values = df[col].values
                values = values - mean
                if standard != 0:
                    values = values / standard
                df[col] = values
        elif len(norm_params.shape) == 3:
            op_list = df['op_type'].unique()
            op_list.sort()
            for op_i, op in enumerate(op_list):
                for col_i, col in enumerate(cols):
                    mean, standard = norm_params[op_i, col_i]
                    values = df[df['op_type'] == op][col].values
                    values = (values - mean)
                    if standard != 0:
                        values = values / standard
                    df.loc[df['op_type'] == op, col] = values
        else:
            raise ValueError('norm_params shape error')
        return df

    def _gen_norm_params(self, norm_type, norm_by_operations=False):
        """
            Get normalization parameters.
            if normalization by conditions, need cluster operations first.

        :param norm_type: 0-1, -1-1 or z-score
        :param norm_by_operations:
        :return: normalize by each op_type
        """
        assert norm_type in ['0-1', '-1-1', 'z-score']

        if norm_by_operations:
            assert self.has_cluster_operations, \
                'need cluster operations before normalization when norm_by_operations is True'


        norm_cols = self.feature_cols

        if norm_by_operations:
            op_list = self.df['op_type'].unique()
            op_list.sort()
            params_list = []
            for op in op_list:
                sub_df = self.df[self.df['op_type'] == op]
                if norm_type == '0-1' or norm_type == '-1-1':
                    col_max = np.max(sub_df[norm_cols].values, axis=0)
                    col_min = np.min(sub_df[norm_cols].values, axis=0)
                    params_list.append(np.stack((col_min, col_max), axis=1))
                if norm_type == 'z-score':
                    mean = np.mean(sub_df[norm_cols].values, axis=0)
                    standard = np.std(sub_df[norm_cols].values, axis=0)
                    params_list.append(np.stack((mean, standard), axis=1))

            return np.stack(params_list, axis=0)
        else:
            df = self.df
            if norm_type in ['0-1', '-1-1']:
                col_max = np.max(df[norm_cols].values, axis=0)
                col_min = np.min(df[norm_cols].values, axis=0)

                return np.stack((col_min, col_max), axis=1)
            elif norm_type == 'z-score':
                mean = np.mean(df[norm_cols].values, axis=0)
                standard = np.std(df[norm_cols].values, axis=0)

                return np.stack((mean, standard), axis=1)

    def _gen_sequence(self):
        seq_cols = ['id'] + self.feature_cols + ['rul']
        seq_len = self.sequence_len
        all_array = []
        # print('gen_sequence')
        # print(self.df)
        for id in self.df['id'].unique():
            id_df = self.df[self.df['id'] == id].sort_values(by='time', ascending=True)
            id_array = id_df[seq_cols].values
            row_num = id_array.shape[0]
            if row_num >= seq_len:
                if self.only_final:
                    all_array.append(id_array[row_num - seq_len:])
                else:
                    for i in range(0, row_num - seq_len + 1):
                        all_array.append(id_array[i:i + seq_len])
            else:
                # row number < sequence length, only one sequence
                # pad width first time-cycle value
                all_array.append(np.pad(id_array, ((seq_len - id_array.shape[0], 0), (0, 0)), 'edge'))

        all_array = np.stack(all_array)

        self.sequence_array = all_array[:, :, 1:-1]

        if self.return_sequence_label:
            self.label_array = all_array[:, :, -1]
        else:
            self.label_array = all_array[:, -1, -1]

        self.id_array = all_array[:, 0, 0]

        self.has_gen_sequence = True

    def _generate_censored_data(self):
        if self.has_cluster_operations or self.has_normalization or self.has_gen_sequence or self.has_count_rul:
            raise RuntimeError('The censored data need to be generated before clustering operations, normalization, sequence generation and counting RUL')

        unique_ids = self.df['id'].unique()
        number_units = len(unique_ids)

        number_censored_units = int(number_units * self.percent_of_censored_data)
        censored_ids = np.random.choice(unique_ids, size=number_censored_units, replace=False)

        self.df['is_censored'] = 0
        self.df.loc[self.df['id'].isin(censored_ids), 'is_censored'] = 1

        # Sort by id/time first so "the first x% of rows" really means "the earliest x%
        # of cycles", regardless of the row order found in the raw txt file.
        self.df = self.df.sort_values(['id', 'time']).reset_index(drop=True)

        # For each censored unit, keep only the first percent_of_broken_data fraction of its
        # cycles (in time order), simulating a unit that has not (yet) reached failure.
        # Built as a boolean mask instead of groupby().apply() because pandas >= 2.2 strips
        # the 'id' grouping column out of the group passed to apply() by default, and pandas
        # 3.x removed the include_groups=True escape hatch entirely.
        keep_mask = np.ones(len(self.df), dtype=bool)
        for unit_id in censored_ids:
            unit_row_positions = np.flatnonzero(self.df['id'].to_numpy() == unit_id)

            pct_to_keep = self.percent_of_broken_data
            if pct_to_keep is None:
                pct_to_keep = np.random.default_rng().random()

            num_rows_to_keep = max(1, int(pct_to_keep * len(unit_row_positions)))
            keep_mask[unit_row_positions[num_rows_to_keep:]] = False

        self.df = self.df.loc[keep_mask].reset_index(drop=True)