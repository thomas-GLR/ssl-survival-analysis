import os
import json

import numpy as np
import pandas as pd
import torch

from constants import necessary_keys_scania
from shared.utils import ModelVersion
from shared.utils import necessary_keys as shared_necessary_keys
from shared.utils.config import extract_data_from_config, assert_params_contains_all_key


RUN_PARAMETERS = "run_parameters.json"


def assert_data_is_valid(
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
):
    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist"
    assert os.path.exists(results_path), f"{results_path} does not exist"
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist"


def create_and_get_checkpoints_results_path(
        model_version: str,
        datetime_for_folders: str,
        checkpoints_path: str,
        results_path: str,
) -> tuple[str, str]:

    folder_for_current_training = f"model-{model_version}-scania-{datetime_for_folders}"

    final_checkpoints_path = os.path.join(checkpoints_path, folder_for_current_training)
    os.makedirs(final_checkpoints_path, exist_ok=True)

    final_results_path = os.path.join(results_path, folder_for_current_training)
    os.makedirs(final_results_path, exist_ok=True)

    return final_checkpoints_path, final_results_path


def get_necessary_dataset_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_dataset_keys(model_version, necessary_keys_scania)


def get_necessary_model_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_model_keys(model_version, necessary_keys_scania)


def get_necessary_training_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_training_keys(model_version, necessary_keys_scania)


def extract_dataset_params_from_config(config_path: str, necessary_keys: list[str]) -> dict:
    params_key = "dataset_params"

    return _extract_params_from_config(params_key, config_path, necessary_keys)


def extract_model_params_from_config(config_path: str, necessary_keys: list[str]) -> dict:
    params_key = "model_params"

    return _extract_params_from_config(params_key, config_path, necessary_keys)


def extract_training_params_from_config(config_path: str, necessary_keys: list[str]) -> dict:
    params_key = "training_params"

    return _extract_params_from_config(params_key, config_path, necessary_keys)


def _extract_params_from_config(params_key: str, config_path: str, necessary_keys: list[str]) -> dict:
    config = extract_data_from_config(config_path)

    if params_key not in config:
        raise KeyError(f"{params_key} not found in config {config_path}")

    model_params = config[params_key]

    assert_params_contains_all_key(model_params, necessary_keys, params_key)

    return model_params


def save_train_parameters(
        results_path: str,
        dataset_parameters: dict,
        training_parameters: dict,
        model_parameters: dict,
) -> None:
    run_parameters = {
        "dataset_parameters": dataset_parameters,
        "training_parameters": training_parameters,
        "model_parameters": model_parameters,
    }

    with open(os.path.join(results_path, RUN_PARAMETERS), "w") as f:
        json.dump(run_parameters, f, indent=2)
    print(f"Parameters written to {results_path}")


def generate_and_save_model_prediction(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_version: str,
        prediction_type: str,
        results_path: str,
) -> tuple[float, float]:
    """
    Generate results for predictions and save them in a csv file.

    Args:
        predictions:
        targets:
        model_version:
        prediction_type:
        results_path:

    Returns:

    """
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    df = pd.DataFrame({
        'targets': targets,
        'predictions': predictions,
    })

    csv_path = f'{results_path}/predictions_{model_version}_{prediction_type}_scania.csv'

    df.to_csv(csv_path, index=False)

    print(f"Results for {prediction_type} are saved under : {csv_path}")

    rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
    score = _scania_score(predictions, targets)

    return rmse, score


def _scania_score(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    a1 = 13
    a2 = 10
    error = predictions - targets
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)