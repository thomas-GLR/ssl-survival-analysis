import json
import os


def extract_data_from_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, mode="rt") as f:
        config = json.load(f)

    return config


def assert_params_contains_all_key(params: dict, necessary_keys: list[str], params_name: str):
    no_existing_keys = []

    for key in necessary_keys:
        if key not in params:
            no_existing_keys.append(key)

    if len(no_existing_keys) > 0:
        raise KeyError(f"The following keys are needed in {params_name} : {no_existing_keys}")