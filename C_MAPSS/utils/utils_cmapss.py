import os
import json


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
    config = extract_data_from_config(config_path)

    params_key = "dataset_params"

    if sub_dataset not in config:
        raise KeyError(f"{sub_dataset} not found in config {config_path}")

    if params_key not in config[sub_dataset]:
        raise KeyError(f"{params_key} not found in config {config_path} for sub dataset {sub_dataset}")

    dataset_params = config[sub_dataset][params_key]

    _assert_params_contains_all_key(dataset_params, necessary_keys, params_key)

    return dataset_params


def extract_model_params_from_config(config_path: str, sub_dataset: str, necessary_keys: list[str]) -> dict:
    config = extract_data_from_config(config_path)

    params_key = "model_params"

    if sub_dataset not in config:
        raise KeyError(f"{sub_dataset} not found in config {config_path}")

    if params_key not in config[sub_dataset]:
        raise KeyError(f"{params_key} not found in config {config_path} for sub dataset {sub_dataset}")

    model_params = config[sub_dataset][params_key]

    _assert_params_contains_all_key(model_params, necessary_keys, params_key)

    return model_params


def _assert_params_contains_all_key(params: dict, necessary_keys: list[str], params_name: str):
    no_existing_keys = []

    for key in necessary_keys:
        if key not in params:
            no_existing_keys.append(key)

    if len(no_existing_keys) > 0:
        raise KeyError(f"The following keys are needed in {params_name} : {no_existing_keys}")

