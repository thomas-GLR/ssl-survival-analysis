import json
import os


def extract_data_from_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, mode="rt") as f:
        config = json.load(f)

    return config


def assert_params_contains_all_key(params: dict, necessary_keys: list[str], params_name: str) -> None:
    no_existing_keys = []

    for key in necessary_keys:
        if key not in params:
            no_existing_keys.append(key)

    if len(no_existing_keys) > 0:
        raise KeyError(f"The following keys are needed in {params_name} : {no_existing_keys}")


def _extract_params_from_config(params_key: str, config_path: str, necessary_keys: list[str], sub_dataset: str | None=None) -> dict:
    config = extract_data_from_config(config_path)

    if sub_dataset is not None and sub_dataset not in config:
        raise KeyError(f"{sub_dataset} not found in config {config_path}")

    if sub_dataset is not None:
        config = config[sub_dataset]

    if params_key not in config:
        sub_dataset_information = ""

        if sub_dataset is not None:
            sub_dataset_information = f" for sub dataset {sub_dataset}"

        raise KeyError(f"{params_key} not found in config {config_path}{sub_dataset_information}")

    model_params = config[params_key]

    assert_params_contains_all_key(model_params, necessary_keys, params_key)

    return model_params


def extract_dataset_params_from_config(config_path: str, necessary_keys: list[str], sub_dataset: str | None=None) -> dict:
    params_key = "dataset_params"

    return _extract_params_from_config(
        params_key=params_key,
        config_path=config_path,
        necessary_keys=necessary_keys,
        sub_dataset=sub_dataset,
    )


def extract_model_params_from_config(config_path: str, necessary_keys: list[str], sub_dataset: str | None=None) -> dict:
    params_key = "model_params"

    return _extract_params_from_config(
        params_key=params_key,
        config_path=config_path,
        necessary_keys=necessary_keys,
        sub_dataset=sub_dataset,
    )


def extract_training_params_from_config(config_path: str, necessary_keys: list[str], sub_dataset: str | None=None) -> dict:
    params_key = "training_params"

    return _extract_params_from_config(
        params_key=params_key,
        config_path=config_path,
        necessary_keys=necessary_keys,
        sub_dataset=sub_dataset,
    )

