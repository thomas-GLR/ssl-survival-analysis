import json
import os
from typing import Callable

import numpy as np

from C_MAPSS.utils import (utils_transformer_lstm,
                           utils_pyclus,
                           utils_coprog,
                           utils_random_survival_forest,
                           utils_self_supervised,
                           utils_cotraining_ensemble)
from C_MAPSS.utils.ModelVersion import ModelVersion
from constants import necessary_keys_cmapss


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


def get_train_model_method(model_version: ModelVersion) -> Callable:
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
        case ModelVersion.CNN:
            return utils_transformer_lstm.train_model
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return utils_cotraining_ensemble.train_model
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return utils_cotraining_ensemble.train_model_v2
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def get_necessary_dataset_keys(model_version: ModelVersion) -> list[str]:
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
        case ModelVersion.CNN:
            return necessary_keys_cmapss.NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return necessary_keys_cmapss.NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return necessary_keys_cmapss.NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_KEYS
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def get_necessary_model_keys(model_version: ModelVersion) -> list[str]:
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
        case ModelVersion.CNN:
            return necessary_keys_cmapss.NECESSARY_CNN_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return necessary_keys_cmapss.NECESSARY_CO_TRAINING_ENSEMBLE_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return necessary_keys_cmapss.NECESSARY_CO_TRAINING_ENSEMBLE_V2_KEYS
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
