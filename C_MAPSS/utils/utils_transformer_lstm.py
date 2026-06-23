from datetime import datetime
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader

import constants.results_columns as results_columns
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning.TransformerLstmModule import TransformerLstmModule
from C_MAPSS.models.Simple_LSTM import Simple_LSTM
from C_MAPSS.models.TransformerEncoder_LSTM_1 import TransformerEncoder_LSTM_1
from utils.utils import cmapss_score
from C_MAPSS.utils.utils_cmapss import extract_benchmark_information_from_config, extract_dataset_params_from_config, \
    extract_model_params_from_config

# For PyTorch 2.6+
# We indicate to PyTorch that these classes are "safe" when loading checkpoints
add_safe_globals([Simple_LSTM, TransformerEncoder_LSTM_1])

NECESSARY_TRANSFORMER_KEYS = [
    "sequence_len",
    "transformer_encoder_head_num",
    "hidden_dim",
    "lstm_num_layers",
    "lstm_dropout",
    "fc_layer_dim",
    "fc_dropout",
    "batch_size",
    "lr",
    "patience",
    "max_epochs",
]

NECESSARY_LSTM_KEYS = [
    "sequence_len",
    "hidden_dim",
    "lstm_num_layers",
    "lstm_dropout",
    "fc_layer_dim",
    "fc_dropout",
    "batch_size",
    "lr",
    "patience",
    "max_epochs",
]

NECESSARY_DATASET_KEYS = [
    "seed",
    "max_rul",
    "return_sequence_label",
    "norm_type",
    "cluster_operations",
    "norm_by_operations",
    "include_cols",
    "exclude_cols",
    "return_id",
    "validation_rate",
    "use_only_final_on_test",
    "use_max_rul_on_test",
    "use_max_rul_on_valid",
]


def benchmark_for_transformer_or_lstm(
        config_path: str,
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        model_version: str,
        device: str,
        benchmark_version: str = "default",
) -> None:
    """
    Launch benchmark on cmapss depending on information in config file

    :param config_path: the path for all the config files
    :param checkpoints_path: the path to store the checkpoints
    :param results_path: the path to store results
    :param dataset_root: the path to the dataset folder where all cmapss files are stored
    :param model_version: the version of the model (transformer, lstm)
    :param device: the device where to run the model
    :param benchmark_version: the folder of the version for the benchmark.
        It enables to run different benchmark configuration
    """
    config_path = f"{config_path}/{benchmark_version}"
    config_benchmark_file_path = f"{config_path}/benchmark.json"
    config_model_file_path = f"{config_path}/{model_version}.json"

    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist."
    assert os.path.exists(results_path), f"{results_path} does not exist."
    assert os.path.exists(config_path), f"{config_path} does not exist."
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist."
    assert os.path.exists(config_benchmark_file_path), f"{config_benchmark_file_path} does not exist."
    assert os.path.exists(config_model_file_path), f"{config_model_file_path} does not exist."

    broken_percentages, censored_percentages, cmapss_files = extract_benchmark_information_from_config(
        config_benchmark_file_path
    )

    columns = [
        results_columns.SUB_DATASET,
        results_columns.CENSORED_PERCENTAGE,
        results_columns.BROKEN_PERCENTAGE,
        results_columns.MODEL,
        results_columns.RMSE,
        results_columns.SCORE
    ]
    rows = []

    benchmark_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for sub_dataset in cmapss_files:
        secure_save_for_sub_dataset_rows = []

        for censored_percentage in censored_percentages:
            secure_save_for_censored_percentage_rows = []

            for broken_percentage in broken_percentages:
                print(
                    f"Training model {model_version} for the sub dataset : {sub_dataset}, censored percentage : {censored_percentage} and broken percentage : {broken_percentage}")

                dataset_params = extract_dataset_params_from_config(
                    config_path=config_model_file_path,
                    sub_dataset=sub_dataset,
                    necessary_keys=NECESSARY_DATASET_KEYS,
                )

                necessary_model_keys = NECESSARY_TRANSFORMER_KEYS if model_version == "transformer" else NECESSARY_LSTM_KEYS

                model_params = extract_model_params_from_config(
                    config_model_file_path,
                    sub_dataset=sub_dataset,
                    necessary_keys=necessary_model_keys,
                )

                rmse, score = train_model(
                    checkpoints_path=checkpoints_path,
                    results_path=results_path,
                    model_version=model_version,
                    dataset_root=dataset_root,
                    sub_dataset=sub_dataset,
                    percent_of_broken_data=broken_percentage,
                    percent_of_censored_data=censored_percentage,
                    **dataset_params,
                    **model_params,
                    device=device,
                    datetime_for_folders=benchmark_datetime,
                )

                new_dataframe_row = {
                    results_columns.SUB_DATASET: sub_dataset,
                    results_columns.CENSORED_PERCENTAGE: censored_percentage,
                    results_columns.BROKEN_PERCENTAGE: broken_percentage,
                    results_columns.MODEL: model_version,
                    results_columns.RMSE: rmse,
                    results_columns.SCORE: score,
                }

                rows.append(new_dataframe_row)
                secure_save_for_censored_percentage_rows.append(new_dataframe_row)
                secure_save_for_sub_dataset_rows.append(new_dataframe_row)

            secure_save_for_censored_percentage = pd.DataFrame(secure_save_for_censored_percentage_rows, columns=columns)

            print(f"Saving intermediate result for sub dataset {sub_dataset} and censored percentage : {censored_percentage}...")
            secure_save_for_censored_percentage.to_csv(
                f"{results_path}/secure_{sub_dataset}_censored_{censored_percentage:.2f}_{model_version}_benchmark_{benchmark_version}_results_turbofan.csv",
                index=False)


        secure_save_for_sub_dataset = pd.DataFrame(secure_save_for_sub_dataset_rows, columns=columns)

        print(f"Saving intermediate result for sub dataset {sub_dataset}...")
        secure_save_for_sub_dataset.to_csv(f"{results_path}/secure_{sub_dataset}_{model_version}_benchmark_{benchmark_version}_results_turbofan.csv", index=False)

    df_results = pd.DataFrame(rows, columns=columns)

    print(df_results.head())

    print("Saving results...")

    df_results.to_csv(f"{results_path}/{model_version}_benchmark_{benchmark_version}_results_turbofan.csv", index=False)


def train_model(
        checkpoints_path: str,
        results_path: str,
        model_version: str,
        # Dataset params
        dataset_root: str,
        seed: int | None,
        sub_dataset: str,
        sequence_len: int = 30,
        max_rul: int = 125,
        return_sequence_label: bool = False,
        norm_type: str = 'z-score',
        cluster_operations: bool = True,
        norm_by_operations: bool = True,
        include_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        return_id: bool = False,
        validation_rate=0.2,
        use_only_final_on_test: bool = True,
        use_max_rul_on_test: bool = False,
        use_max_rul_on_valid: bool = True,
        percent_of_broken_data: float | None = None,
        percent_of_censored_data: float = 0.9,
        # Model params
        transformer_encoder_head_num=2,
        lstm_num_layers=3,
        hidden_dim=32,
        lstm_dropout=0.2,
        fc_layer_dim=32,
        fc_dropout=0.2,
        # Training
        device: str | None = None,
        batch_size=256,
        lr=0.001,
        patience=10,
        max_epochs: int = 500,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
) -> tuple[float, float]:
    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist"
    assert os.path.exists(results_path), f"{results_path} does not exist"
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist"

    assert sub_dataset in ['FD001', 'FD002', 'FD003',
                           'FD004'], f"Sub dataset must be one of ['FD001', 'FD002', 'FD003', 'FD004'] and not {sub_dataset}"

    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

    scores = pd.DataFrame(columns=['train_rmse', 'val_rmse', 'test_rmse', 'test_score'])

    broken_percentage = 0. if percent_of_broken_data is None else percent_of_broken_data

    folder_for_current_training = f"/model-{model_version}-turbofan-{sub_dataset}-{datetime_for_folders}/censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"

    checkpoints_path = f'{checkpoints_path}/{folder_for_current_training}'
    results_path = f'{results_path}/{folder_for_current_training}'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    dataset_kwargs = {
        'dataset_root': dataset_root,
        'sub_dataset': sub_dataset,
        'sequence_len': sequence_len,
        'max_rul': max_rul,
        'return_sequence_label': return_sequence_label,
        'norm_type': norm_type,
        'cluster_operations': cluster_operations,
        'norm_by_operations': norm_by_operations,
        'include_cols': include_cols,
        'exclude_cols': exclude_cols,
        'return_id': return_id,
        'validation_rate': validation_rate,
        'use_only_final_on_test': use_only_final_on_test,
        'use_max_rul_on_test': use_max_rul_on_test,
        'use_max_rul_on_valid': use_max_rul_on_valid,
        'percent_of_broken_data': percent_of_broken_data,
        'percent_of_censored_data': percent_of_censored_data,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        sequence_len=sequence_len,
        seed=seed,
        max_rul=max_rul,
        return_sequence_label=return_sequence_label,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
        return_id=return_id,
        validation_rate=validation_rate,
        use_only_final_on_test=use_only_final_on_test,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
    )

    feature_num = len(train_dataset.feature_cols)

    # As the models can only handle supervised learning we need to filtered censored data
    train_loader = train_dataset.get_data_loader_without_censored_data(batch_size=batch_size)
    test_loader = test_dataset.get_data_loader_without_censored_data(batch_size=batch_size)
    valid_loader = valid_dataset.get_data_loader_without_censored_data(
        batch_size=batch_size) if valid_dataset is not None else None

    model_kwargs = {
        'sequence_len': sequence_len,
        'feature_num': feature_num,
        'hidden_dim': hidden_dim,
        'fc_layer_dim': fc_layer_dim,
        'lstm_num_layers': lstm_num_layers,
        'transformer_encoder_head_num': transformer_encoder_head_num,
        'fc_dropout': fc_dropout,
        'lstm_dropout': lstm_dropout,
    }
    print('Training model with the following parameters:')
    print(f"Sequence length : {sequence_len}")
    print(f"Patience : {patience}")
    print(f"Models parameters : {model_kwargs}")

    if model_version == 'transformer':
        model = TransformerEncoder_LSTM_1(**model_kwargs)
    elif model_version == 'lstm':
        model = Simple_LSTM(**model_kwargs)
    else:
        raise ValueError(f"Model version {model_version} is not supported")

    transformer_lstm_module = TransformerLstmModule(
        lr=lr,
        model=model
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        default_root_dir=checkpoints_path,
        accelerator=device,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(transformer_lstm_module, train_loader, val_dataloaders=valid_loader or test_loader)

    callbacks_metrics = trainer.callback_metrics
    train_rmse = callbacks_metrics['train_rmse']
    val_rmse = callbacks_metrics['val_rmse']

    trainer.test(dataloaders=test_loader, ckpt_path='best')

    callbacks_metrics = trainer.callback_metrics

    test_rmse = callbacks_metrics['test_rmse']
    test_score = callbacks_metrics['test_score']

    # Add the results to the dataframe
    scores.loc[0] = [train_rmse, val_rmse, test_rmse, test_score]

    # Save the results
    scores.to_csv(f'{results_path}/{model_version}-turbofan-{sub_dataset}.csv', index=False)

    print(f"Scores from train and test :\n{scores}")

    # Save model predictions
    if model_version == 'transformer':
        model_for_reload = TransformerEncoder_LSTM_1(**model_kwargs)
    elif model_version == 'lstm':
        model_for_reload = Simple_LSTM(**model_kwargs)
    else:
        raise ValueError(f"Model reload version {model_version} is not supported")

    transformer_lstm_module_with_trained_model = TransformerLstmModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model=model_for_reload,
    )

    transformer_lstm_module_with_trained_model = transformer_lstm_module_with_trained_model.to(device)
    transformer_lstm_module_with_trained_model.eval()

    return _generate_and_save_model_prediction(
        loader=test_loader,
        device=device,
        module=transformer_lstm_module_with_trained_model,
        model_version=model_version,
        prediction_type='test',
        results_path=results_path,
        sub_dataset=sub_dataset,
    )


def _generate_and_save_model_prediction(
        loader: DataLoader,
        device: str,
        module: pl.LightningModule,
        model_version: str,
        prediction_type: str,
        results_path: str,
        sub_dataset: str,
) -> tuple[float, float]:
    """
    Generate results for predictions from the data_loader and save them in csv file

    :param loader: the loader to use for predictions.
    :param device: the device to use for predictions.
    :param module: the module containing the model.
    :param model_version: the model version.
    :param prediction_type: the prediction type.
    :param results_path: the path from which to save the results.
    :param sub_dataset: the sub dataset name to use in the results file name.
    """
    predictions = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y_hat = module(x)
        predictions.extend(y_hat.cpu().detach().numpy().flatten())
        targets.extend(y.cpu().detach().numpy().flatten())

    df = pd.DataFrame({
        'targets': targets,
        'predictions': predictions
    })

    csv_path = f'{results_path}/{model_version}_{prediction_type}_results_{sub_dataset}.csv'

    df.to_csv(csv_path, index=False)

    print(f"Results for {prediction_type} are saved under : {csv_path}")

    predictions_tensor = torch.Tensor(predictions)
    targets_tensor = torch.Tensor(targets)

    rmse = torch.sqrt(torch.mean((targets_tensor - predictions_tensor) ** 2))
    score = cmapss_score(np.array(predictions), np.array(targets))

    return rmse.item(), score
