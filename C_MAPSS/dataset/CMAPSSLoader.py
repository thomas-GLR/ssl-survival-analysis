import os

import numpy as np
import pandas as pd

from C_MAPSS.dataset.CMAPSSDataset import CMAPSSDataset


class CMAPSSLoader:

    @staticmethod
    def get_datasets(
            dataset_root,
            sub_dataset='FD001',
            sequence_len=1,
            max_rul=None,
            return_sequence_label=False,
            norm_type=None,
            cluster_operations=False,
            norm_by_operations=False,
            include_cols=None,
            exclude_cols=None,
            return_id=False,
            validation_rate=0.2,
            use_only_final_on_test=True,
            use_max_rul_on_test=False,
            use_max_rul_on_valid=True,
            percent_of_censored_data: float = 0.0,
            percent_of_broken_data: float | None = None
    ) -> tuple[CMAPSSDataset, CMAPSSDataset, CMAPSSDataset]:
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

        :param validation_rate:
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

        """
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

        # split valid set
        # train_df[0] is unit id column
        valid_df = None
        valid_dataset = None
        assert 0 <= validation_rate <= 0.99
        if validation_rate:
            ids = train_df[0].unique()
            max_id = np.max(ids)
            valid_len = int(validation_rate * max_id)
            if valid_len:
                # random chose valid engine id
                valid_ids = np.random.choice(np.arange(1, max_id + 1), valid_len, replace=False)

                isin_df = np.isin(train_df[0].to_numpy(), valid_ids)
                valid_df = train_df.iloc[np.where(isin_df == True)]
                train_df = train_df.iloc[np.where(isin_df == False)]

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
            'return_id': return_id
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

        return train_dataset, test_dataset, valid_dataset
