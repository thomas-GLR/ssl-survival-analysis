import os

import numpy as np
import pandas as pd

from C_MAPSS.dataset.CMAPSSDataset import CMAPSSDataset


class CMAPSSLoader:

    @staticmethod
    def get_datasets(
            dataset_root,
            sub_dataset: str,
            sequence_len: int,
            seed: int | None,
            max_rul=None,
            return_sequence_label=False,
            norm_type=None,
            cluster_operations=False,
            norm_by_operations=False,
            include_cols=None,
            exclude_cols=None,
            return_id=False,
            val_rate=0.2,
            calib_rate: float = 0.0,
            use_only_final_on_test=True,
            use_max_rul_on_test=False,
            use_max_rul_on_valid=True,
            percent_of_censored_data: float = 0.0,
            percent_of_broken_data: float | None = None,
    ) -> tuple[CMAPSSDataset, CMAPSSDataset, CMAPSSDataset | None, CMAPSSDataset | None]:
        """
            Get train, valid, test dataset from dataset file.
            The parameter with the same name as in __init__ has the same effect, they are:
            sequence_len, max_rul, return_sequence_label, include_cols, exclude_cols, return id

        :param dataset_root:
            root directory of raw txt files

        :param sub_dataset:
            A string denote the dataset name, FD001/FD002/FD003/FD004

        :param sequence_len:
        :param max_rul:
        :param return_sequence_label:
        :param norm_type:
        :param cluster_operations:
        :param norm_by_operations:
        :param include_cols:
        :param exclude_cols:
        :param return_id:

        :param val_rate:
            Number of units used in the validation set as a percentage of the total training set, default is 0.2
            validation_rate = len(validation_dataset.df['id'].unique()) / len(full_train_dataset.df['id'].unique())

        :param use_only_final_on_test:
            set only_final on test dataset, default is True

        :param use_max_rul_on_test:
            use max_rul on test dataset

        :param use_max_rul_on_valid:
            use max_rul on validation dataset

        :param percent_of_censored_data:
            percentage of censored data, default is 0.0, which means no censored data.
            The percentage of censored data dosen't apply to test dataset

        :param percent_of_broken_data:
            This is the percent of damage until the data is censored. Default is 0.0

        :param seed:
            Set the seed to reproduce results
        """
        if seed is not None:
            np.random.seed(seed)

        if sub_dataset == 'PHM08':
            train_df = pd.read_csv(os.path.join(dataset_root, 'train.txt'), sep=' ', header=None)
            test_df = pd.read_csv(os.path.join(dataset_root, 'test.txt'.format(sub_dataset)), sep=' ', header=None)
            # PHM08 test dataset has 218 unit
            rul = np.empty(218)
            rul[:] = np.nan
        else:
            train_df = pd.read_csv(os.path.join(dataset_root, 'train_{:s}.txt'.format(sub_dataset)), sep=' ',
                                   header=None)
            test_df = pd.read_csv(os.path.join(dataset_root, 'test_{:s}.txt'.format(sub_dataset)), sep=' ',
                                  header=None)
            rul_df = pd.read_csv(os.path.join(dataset_root, 'RUL_{:s}.txt'.format(sub_dataset)), header=None)
            rul = rul_df.values.squeeze()

        valid_dataset = None
        calib_dataset = None

        assert (
            0 <= val_rate < 1 and 0 <= calib_rate < 1
            and (val_rate + calib_rate) < 1
        ), "val_rate/calib_rate must be in [0, 1) and sum to < 1"

        rng = np.random.default_rng(seed)

        ids = rng.permutation(train_df[0].unique())

        id_number = len(ids)
        id_number_val = int(val_rate * id_number)
        id_number_calib = int(calib_rate * id_number)

        val_ids = set(ids[:id_number_val].tolist())
        calib_ids = set(ids[id_number_val:id_number_val + id_number_calib].tolist())

        is_val = train_df[0].isin(val_ids)
        is_calib = train_df[0].isin(calib_ids)

        valid_df = train_df[is_val]
        calib_df = train_df[is_calib] if calib_rate > 0.0 else None
        train_df = train_df[~(is_val | is_calib)]

        if sub_dataset in ['FD001', 'FD003']:
            norm_by_operations = False
            cluster_operations = False

        common_dataset_kwargs = {
            'sequence_len': sequence_len,
            'max_rul': max_rul,
            'norm_type': norm_type,
            'include_cols': include_cols,
            'exclude_cols': exclude_cols,
            'cluster_operations': cluster_operations,
            'norm_by_operations': norm_by_operations,
            'return_sequence_label': return_sequence_label,
            'return_id': return_id,
            'seed': seed,
        }

        # Only the train and val dataset should have censored data
        train_val_dataset_kwargs = {
            'percent_of_censored_data': percent_of_censored_data,
            'percent_of_broken_data': percent_of_broken_data
        }

        train_val_dataset_kwargs.update(common_dataset_kwargs)

        # print
        train_dataset = CMAPSSDataset(
            train_df,
            **train_val_dataset_kwargs
        )
        common_dataset_kwargs['final_rul'] = rul
        if not use_max_rul_on_test and 'max_rul' in common_dataset_kwargs:
            common_dataset_kwargs.pop('max_rul')
        if use_only_final_on_test:
            common_dataset_kwargs['only_final'] = True

        test_dataset = CMAPSSDataset(
            test_df,
            kmeans_model=train_dataset.kmeans_model,
            norm_params=train_dataset.norm_params,
            **common_dataset_kwargs
        )

        if valid_df is not None:
            if 'final_rul' in train_val_dataset_kwargs:
                train_val_dataset_kwargs.pop('final_rul')
            if not use_max_rul_on_valid and 'max_rul' in train_val_dataset_kwargs:
                train_val_dataset_kwargs.pop('max_rul')
            if use_max_rul_on_valid and max_rul is not None:
                train_val_dataset_kwargs['max_rul'] = max_rul
            if 'only_final' in train_val_dataset_kwargs:
                train_val_dataset_kwargs.pop('only_final')
            valid_dataset = CMAPSSDataset(
                valid_df,
                kmeans_model=train_dataset.kmeans_model,
                norm_params=train_dataset.norm_params,
                **train_val_dataset_kwargs
            )

        if calib_df is not None:
            if 'final_rul' in train_val_dataset_kwargs:
                train_val_dataset_kwargs.pop('final_rul')
            if not use_max_rul_on_valid and 'max_rul' in train_val_dataset_kwargs:
                train_val_dataset_kwargs.pop('max_rul')
            if use_max_rul_on_valid and max_rul is not None:
                train_val_dataset_kwargs['max_rul'] = max_rul
            if 'only_final' in train_val_dataset_kwargs:
                train_val_dataset_kwargs.pop('only_final')
            calib_dataset = CMAPSSDataset(
                calib_df,
                kmeans_model=train_dataset.kmeans_model,
                norm_params=train_dataset.norm_params,
                **train_val_dataset_kwargs
            )

        return train_dataset, test_dataset, valid_dataset, calib_dataset
