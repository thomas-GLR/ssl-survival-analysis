import json
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics.functional import mean_squared_error

from scania.lightning_module import BasicLightningModule
from models import CNN1D, Simple_LSTM, TransformerFeatures
from shared.enums import ModelVersion, DatasetVersion

from scania.metrics import scania_score
from C_MAPSS.metrics import cmapss_score

RUN_PARAMETERS = "run_parameters.json"


def assert_data_is_valid(
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        sub_dataset: str | None=None,
):
    """
    Check if all paths exists

    Args:
        checkpoints_path: the path of the checkpoints
        results_path: the path of the results
        dataset_root: the path of the dataset
        sub_dataset: the path of the sub dataset for cmapss if given. If not given the value is not checked

    """
    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist"
    assert os.path.exists(results_path), f"{results_path} does not exist"
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist"

    if sub_dataset is not None:
        assert sub_dataset in ['FD001', 'FD002', 'FD003',
                               'FD004'], f"Sub dataset must be one of ['FD001', 'FD002', 'FD003', 'FD004'] and not {sub_dataset}"


def save_train_parameters(
        results_path: str,
        dataset_parameters: dict,
        training_parameters: dict,
        model_parameters: dict,
) -> None:
    """Save dataset, training, and model parameters into a JSON file.

    Args:
        results_path: Directory where run_parameters.json will be saved.
        dataset_parameters: Dictionary of dataset parameters.
        training_parameters: Dictionary of training parameters.
        model_parameters: Dictionary of model parameters.
    """
    run_parameters = {
        "dataset_parameters": dataset_parameters,
        "training_parameters": training_parameters,
        "model_parameters": model_parameters,
    }

    with open(os.path.join(results_path, RUN_PARAMETERS), "w") as f:
        json.dump(run_parameters, f, indent=2)
    print(f"Parameters written to {results_path}")


def create_and_get_checkpoints_results_path(
    model_version: ModelVersion,
    dataset_version: DatasetVersion,
    datetime_for_folders: str,
    checkpoints_path: str,
    results_path: str,
    sub_dataset: str | None=None,
    percent_of_censored_data: float | None=None,
    percent_of_broken_data: float | None=None,
) -> tuple[str, str]:
    match dataset_version:
        case DatasetVersion.CMAPSS:
            if percent_of_censored_data is None:
                raise ValueError("percent_of_censored_data cannot be None for CMAPSS dataset")

            broken_percentage = percent_of_broken_data if percent_of_broken_data is not None else 0.0

            folder_for_current_training = (
                f"model-{model_version.value}-turbofan-{sub_dataset}-{datetime_for_folders}/"
                f"censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"
            )
        case DatasetVersion.SCANIA:
            folder_for_current_training = f"model-{model_version.value}-scania-{datetime_for_folders}"
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")

    final_checkpoints_path = os.path.join(checkpoints_path, folder_for_current_training)
    os.makedirs(final_checkpoints_path, exist_ok=True)

    final_results_path = os.path.join(results_path, folder_for_current_training)
    os.makedirs(final_results_path, exist_ok=True)

    return final_checkpoints_path, final_results_path


def generate_and_save_model_prediction(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_version: ModelVersion,
        prediction_type: str,
        results_path: str,
        score_callback: Callable[[DatasetVersion, torch.Tensor, torch.Tensor], float],
        dataset_version: DatasetVersion,
        sub_dataset: str | None=None,
) -> tuple[float, float]:
    """
    Generate results for predictions and save them in a csv file.

    Args:
        predictions: the predictions
        targets: the targets
        model_version: the model version for the folder name of output
        prediction_type: the type of prediction for the folder name of output
        results_path: the path of the folder where the results will be saved
        score_callback: the score callback to get the right one depending on the dataset version
        dataset_version: the dataset version (CMAPSS, SCANIA)
        sub_dataset: the sub dataset for cmapss for the fodler name

    Returns:
        the RMSE and the score value
    """
    np_predictions = predictions.cpu().numpy().flatten()
    np_targets = targets.cpu().numpy().flatten()

    df = pd.DataFrame({
        'targets': np_targets,
        'predictions': np_predictions,
    })

    if sub_dataset is not None:
        dataset_information = f"turbofan_{sub_dataset}"
    else:
        dataset_information = "scania"

    csv_path = f'{results_path}/predictions_{model_version.value}_{prediction_type}_{dataset_information}.csv'

    df.to_csv(csv_path, index=False)

    print(f"Results for {prediction_type} are saved under : {csv_path}")

    rmse = criteria_rmse_callback(predictions, targets)

    score = score_callback(dataset_version, predictions, targets)

    return rmse, score


def creating_model(
        model_params: dict,
        model_version: ModelVersion,
        num_features: int,
        sequence_len: int,
) -> nn.Module:
    """
    Instantiate a neural network model.

    Args:
        model_params: Architecture parameter dictionary.
        model_version: ModelVersion enum.
        num_features: Number of input sensor features.
        sequence_len: Input sequence length.

    Returns:
        A nn.Module instance.

    """
    match model_version:
        case ModelVersion.CNN:
            return CNN1D(num_features=num_features)
        case ModelVersion.LSTM:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
            })
            return Simple_LSTM(**model_params)
        case ModelVersion.TRANSFORMER_FEATURES:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
            })
            return TransformerFeatures(**model_params)
        case _:
            raise ValueError(f"{model_version.value} is not a valid model version for COPROG")


def build_scania_module(
    model_params: dict,
    model_version_value: str,
    feature_num: int,
    sequence_len: int,
    lr: float,
    target_mean: float,
    target_std: float,
) -> LightningModule:
    """
    Build a fresh ``BasicLightningModule`` wrapping a fresh Scania model.

    This is a **module-level** function (not a closure) so that, wrapped in
    ``functools.partial``, it is picklable and can be sent to per-GPU worker processes for
    parallel COPROG training. It rebuilds the underlying ``nn.Module`` from its config via
    :func:`_creating_model` and wraps it in :class:`BasicLightningModule`.

    Args:
        model_params: The model's constructor params (inner config dict). Copied before
        use because :func:`_creating_model` mutates it via ``.update``.
        model_version_value: The model version string (e.g. ``"cnn"``), converted back to
        a :class:`ModelVersion` (an enum is passed as its value to keep the partial picklable).
        feature_num: Number of input features.
        sequence_len: Input sequence length.
        lr: Learning rate for the wrapping module.
        target_mean: RUL target mean for standardization (0.0 disables it).
        target_std: RUL target std for standardization (1.0 disables it).

    Returns:
        A fresh ``BasicLightningModule`` ready to train.
    """
    model = creating_model(dict(model_params), ModelVersion(model_version_value), feature_num, sequence_len)

    return BasicLightningModule(
        lr=lr,
        model=model,
        target_mean=target_mean,
        target_std=target_std,
    )

def criteria_rmse_callback(
        preds: torch.Tensor,
        targets: torch.Tensor
) -> float:
    rmse = mean_squared_error(preds, targets, squared=False)

    return rmse.item()


def criteria_score_callback(
        dataset_version: DatasetVersion,
        preds: torch.Tensor,
        targets: torch.Tensor
) -> float:
    """
    Function cost for a prediction/target tensor pair depending on the dataset version (CMAPSS, SCANIA).

    Args:
        dataset_version: the version of the dataset (cmapss and scania have different cost functions)
        preds: predicted RUL, shape (N,).
        targets: True RUL, shape (N,).

    Returns:
        The function cost as a Python float.
    """
    np_predictions = preds.detach().cpu().numpy().flatten()
    np_targets = targets.detach().cpu().numpy().flatten()

    match dataset_version:
        case DatasetVersion.CMAPSS:
            return cmapss_score(
                np_predictions,
                np_targets,
            )
        case DatasetVersion.SCANIA:
            return scania_score(
                np_predictions,
                np_targets,
            )
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")
