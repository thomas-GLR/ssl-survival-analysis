from typing import Callable

import numpy as np

from C_MAPSS.utils import (
    utils_transformer_lstm,
    utils_pyclus,
    utils_random_survival_forest,
    utils_self_supervised,
    utils_cotraining_ensemble
)
from constants import necessary_keys_cmapss
from shared.enums import ModelVersion
from shared.utils import utils_coprog
from shared.utils import necessary_keys as shared_necessary_keys
from shared.utils.config import extract_data_from_config


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


def get_train_model_method(model_version: ModelVersion) -> Callable:
    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
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
    return shared_necessary_keys.get_necessary_dataset_keys(model_version, necessary_keys_cmapss)


def get_necessary_model_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_model_keys(model_version, necessary_keys_cmapss)


def get_necessary_training_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_training_keys(model_version, necessary_keys_cmapss)
