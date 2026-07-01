import json
import os
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from constants import necessary_keys_cmapss
from constants import results_columns
from C_MAPSS.utils import (utils_transformer_lstm,
                           utils_pyclus,
                           utils_coprog,
                           utils_random_survival_forest,
                           utils_self_supervised)
from C_MAPSS.utils.ModelVersion import ModelVersion


def reproduce_result(
        config_path: str,
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        model_version: ModelVersion,
        device: str | None,
        benchmark_version: str = "default",
) -> None:
    """
    Launch benchmark on cmapss depending on information in config file

    :param config_path: the path for all the config files
    :param checkpoints_path: the path to store the checkpoints
    :param results_path: the path to store results
    :param dataset_root: the path to the dataset folder where all cmapss files are stored
    :param model_version: the version of the model
    :param device: the device where to run the model
    :param benchmark_version: the folder of the version for the benchmark.
        It enables to run different benchmark configuration
    """
    config_path = f"{config_path}/{benchmark_version}"
    config_benchmark_file_path = f"{config_path}/benchmark.json"
    config_model_file_path = f"{config_path}/{model_version.value}.json"

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

    necessary_dataset_keys = _get_necessary_dataset_keys(model_version)
    necessary_model_keys = _get_necessary_model_keys(model_version)

    benchmark_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_model_func = _get_train_model_method(model_version)

    for sub_dataset in cmapss_files:
        secure_save_for_sub_dataset_rows = []

        for censored_percentage in censored_percentages:
            # secure_save_for_censored_percentage_rows = []

            # We don't wan't to iterate for no reason when censored == 0.0
            broken_percentages_tmp = [0.0] if censored_percentage == 0.0 else broken_percentages

            for broken_percentage in broken_percentages_tmp:
                print(
                    f"Training model {model_version.value} for the sub dataset : {sub_dataset}, censored percentage : {censored_percentage} and broken percentage : {broken_percentage}")

                dataset_params = extract_dataset_params_from_config(
                    config_path=config_model_file_path,
                    sub_dataset=sub_dataset,
                    necessary_keys=necessary_dataset_keys,
                )

                model_params = extract_model_params_from_config(
                    config_model_file_path,
                    sub_dataset=sub_dataset,
                    necessary_keys=necessary_model_keys,
                )

                rmse, score = train_model_func(
                    checkpoints_path=checkpoints_path,
                    results_path=results_path,
                    model_version=model_version.value,
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
                    results_columns.MODEL: model_version.value,
                    results_columns.RMSE: rmse,
                    results_columns.SCORE: score,
                }

                rows.append(new_dataframe_row)
                # secure_save_for_censored_percentage_rows.append(new_dataframe_row)
                secure_save_for_sub_dataset_rows.append(new_dataframe_row)

            # secure_save_for_censored_percentage = pd.DataFrame(secure_save_for_censored_percentage_rows,
            #                                                    columns=columns)

            # print(
            #     f"Saving intermediate result for sub dataset {sub_dataset} and censored percentage : {censored_percentage}...")
            # secure_save_for_censored_percentage.to_csv(
            #     f"{results_path}/secure_{sub_dataset}_censored_{censored_percentage:.2f}_{model_version.value}_benchmark_{benchmark_version}_{benchmark_datetime}_results_turbofan.csv",
            #     index=False)

        secure_save_for_sub_dataset = pd.DataFrame(secure_save_for_sub_dataset_rows, columns=columns)

        print(f"Saving intermediate result for sub dataset {sub_dataset}...")
        secure_save_for_sub_dataset.to_csv(
            f"{results_path}/secure_{sub_dataset}_{model_version.value}_benchmark_{benchmark_version}_{benchmark_datetime}_results_turbofan.csv",
            index=False)

    df_results = pd.DataFrame(rows, columns=columns)

    print(df_results.head())

    print("Saving results...")

    df_results.to_csv(f"{results_path}/{model_version.value}_benchmark_{benchmark_version}_{benchmark_datetime}_results_turbofan.csv",
                      index=False)


def extract_benchmark_information_from_config(config_benchmark_file_path: str) -> tuple[
    list[float], list[float], list[str]]:
    """
    Extract benchmark information from config file

    :param config_benchmark_file_path: th path to the config benchmark gile
    :return:
    """
    config = extract_data_from_config(config_benchmark_file_path)

    broken_percentage: list[float] = config["broken_percentage"]
    censored_percentage: list[float] = config["censored_percentage"]
    cmapss_files: list[str] = config["cmapss_files"]

    return broken_percentage, censored_percentage, cmapss_files


def extract_data_from_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, mode="rt") as f:
        config = json.load(f)

    return config


def extract_dataset_params_from_config(config_path: str, sub_dataset: str, necessary_keys: list[str]) -> dict:
    params_key = "dataset_params"

    return _extract_params_from_config(params_key, config_path, sub_dataset, necessary_keys)


def extract_model_params_from_config(config_path: str, sub_dataset: str, necessary_keys: list[str]) -> dict:
    params_key = "model_params"

    return _extract_params_from_config(params_key, config_path, sub_dataset, necessary_keys)


def _extract_params_from_config(params_key: str, config_path: str, sub_dataset: str, necessary_keys: list[str]) -> dict:
    config = extract_data_from_config(config_path)

    if sub_dataset not in config:
        raise KeyError(f"{sub_dataset} not found in config {config_path}")

    if params_key not in config[sub_dataset]:
        raise KeyError(f"{params_key} not found in config {config_path} for sub dataset {sub_dataset}")

    model_params = config[sub_dataset][params_key]

    _assert_params_contains_all_key(model_params, necessary_keys, params_key)

    return model_params


def _get_train_model_method(model_version: ModelVersion) -> Callable:
    match model_version:
        case ModelVersion.TRANSFORMER:
            return utils_transformer_lstm.train_model
        case ModelVersion.LSTM:
            return utils_transformer_lstm.train_model
        case ModelVersion.AUTOENCODER:
            return utils_self_supervised.train_self_supervised
        case ModelVersion.METRIC:
            return utils_self_supervised.train_self_supervised
        case ModelVersion.RSF:
            return utils_random_survival_forest.train_model
        case ModelVersion.PYCLUS:
            return utils_pyclus.train_model
        case ModelVersion.COPROG:
            return utils_coprog.train_model
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def _get_necessary_dataset_keys(model_version: ModelVersion) -> list[str]:
    match model_version:
        case ModelVersion.TRANSFORMER:
            return necessary_keys_cmapss.NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
        case ModelVersion.LSTM:
            return necessary_keys_cmapss.NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
        case ModelVersion.AUTOENCODER:
            return necessary_keys_cmapss.NECESSARY_DATASET_SELF_SUPERVISED_KEYS
        case ModelVersion.METRIC:
            return necessary_keys_cmapss.NECESSARY_DATASET_SELF_SUPERVISED_KEYS
        case ModelVersion.RSF:
            return necessary_keys_cmapss.NECESSARY_DATASET_RSF_KEYS
        case ModelVersion.PYCLUS:
            return necessary_keys_cmapss.NECESSARY_DATASET_PYCLUS_KEYS
        case ModelVersion.COPROG:
            return necessary_keys_cmapss.NECESSARY_DATASET_COPROG_KEYS
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def _get_necessary_model_keys(model_version: ModelVersion) -> list[str]:
    match model_version:
        case ModelVersion.TRANSFORMER:
            return necessary_keys_cmapss.NECESSARY_TRANSFORMER_KEYS
        case ModelVersion.LSTM:
            return necessary_keys_cmapss.NECESSARY_LSTM_KEYS
        case ModelVersion.AUTOENCODER:
            return necessary_keys_cmapss.NECESSARY_SELF_SUPERVISED_KEYS
        case ModelVersion.METRIC:
            return necessary_keys_cmapss.NECESSARY_SELF_SUPERVISED_KEYS
        case ModelVersion.RSF:
            return necessary_keys_cmapss.NECESSARY_RSF_KEYS
        case ModelVersion.PYCLUS:
            return necessary_keys_cmapss.NECESSARY_PYCLUS_KEYS
        case ModelVersion.COPROG:
            return necessary_keys_cmapss.NECESSARY_COPROG_KEYS
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def _assert_params_contains_all_key(params: dict, necessary_keys: list[str], params_name: str):
    no_existing_keys = []

    for key in necessary_keys:
        if key not in params:
            no_existing_keys.append(key)

    if len(no_existing_keys) > 0:
        raise KeyError(f"The following keys are needed in {params_name} : {no_existing_keys}")


def assert_data_is_valid(
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        sub_dataset: str,
):
    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist"
    assert os.path.exists(results_path), f"{results_path} does not exist"
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist"

    assert sub_dataset in ['FD001', 'FD002', 'FD003',
                           'FD004'], f"Sub dataset must be one of ['FD001', 'FD002', 'FD003', 'FD004'] and not {sub_dataset}"


def create_and_get_checkpoints_results_path(
        percent_of_censored_data: float,
        percent_of_broken_data: float | None,
        model_version: str,
        sub_dataset: str,
        datetime_for_folders: str,
        checkpoints_path: str,
        results_path: str,
) -> tuple[str, str]:
    broken_percentage = percent_of_broken_data if percent_of_broken_data is not None else 0.0

    folder_for_current_training = (
        f"model-{model_version}-turbofan-{sub_dataset}-{datetime_for_folders}/"
        f"censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"
    )

    final_checkpoints_path = os.path.join(checkpoints_path, folder_for_current_training)
    os.makedirs(final_checkpoints_path, exist_ok=True)

    final_results_path = os.path.join(results_path, folder_for_current_training)
    os.makedirs(final_results_path, exist_ok=True)

    return final_checkpoints_path, final_results_path


def cmapss_score(predict: np.ndarray, label: np.ndarray) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)
