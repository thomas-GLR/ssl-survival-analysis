"""
run_train_cmapss.py
──────────
Docker / CLI entry point for training models.

Usage:
    python run_train_cmapss.py transformer --subset FD001 --device cuda --config-path ../config
    --checkpoints-path ../checkpoints --results-path ../results --dataset-root ../data --benchmark-version default
"""

from __future__ import annotations

import os
import traceback
from typing import Callable

from shared.co_training import train_model_coprog
from shared.enums import DatasetVersion
from C_MAPSS.utils import utils_transformer_lstm, utils_self_supervised, utils_random_survival_forest, utils_pyclus, \
    utils_cotraining_ensemble
from shared.co_training import train_model_conformal_cotraining_ensemble

from constants import necessary_shared_keys as shared_keys

# Cap CPU/BLAS thread pools *before* numpy/torch/sklearn initialize their
# native backends. GPU pods (e.g. RunPod) often expose many vCPUs; left
# uncapped, every small CPU-side op (dataset windowing, KMeans, DataLoader
# collate) spawns a thread per visible core and spends more time on
# contention than on work, pegging the CPU without speeding anything up.
# Respect whatever the container/orchestrator already set (e.g. .env.train)
# and only fall back to a default here.
_NUM_THREADS = os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NUM_THREADS)

# This environment variable avoid for missing cuBLAS when using deterministic algorithm.
# This happend because cuBLAS is used by every nn.Linear/matmul, so the LSTM, Transformer, and CNN's dense layers, and the backward pass
# cuBLAS has its own determinism switch that's independent of torch.use_deterministic_algorithms — it needs an environment variable set before CUDA initializes.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import logging
from datetime import datetime

import pandas as pd
import torch

torch.set_num_threads(int(_NUM_THREADS))
torch.set_num_interop_threads(int(_NUM_THREADS))

from constants import results_columns
from shared.utils import (
    extract_dataset_params_from_config,
    extract_model_params_from_config,
    extract_training_params_from_config,
)

from shared.enums import ModelVersion

from C_MAPSS.utils import (
    get_necessary_dataset_keys,
    get_necessary_model_keys,
    get_necessary_training_keys,
    extract_benchmark_information_from_config,
)

logger = logging.getLogger(__name__)


def reproduce_result(
        config_path: str,
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        model_version: ModelVersion,
        benchmark_version: str = "default",
        run_name: str = "",
        gpu_ids: list[int] | None = None,
        pretrained_model_dir: str | None = None,
) -> None:
    """
    Reproduce results for CMAPSS dataset

    Args:
        config_path: the path for all the config files
        checkpoints_path: the path to store the checkpoints
        results_path: the path to store results
        dataset_root: the path to the dataset folder where all cmapss files are stored
        model_version: the version of the model
        benchmark_version: the folder of the version for the benchmark.
            It enables to run different benchmark configuration
        run_name: The name of the run
        gpu_ids: GPU id(s) to train on (COPROG and co-training ensembles only)
        pretrained_model_dir: Directory containing pretrained initial models
    """
    config_path = f"{config_path}/{benchmark_version}"
    config_benchmark_file_path = f"{config_path}/benchmark.json"
    config_model_file_path = f"{config_path}/{model_version.value}.json"

    is_coprog_or_co_training_ensemble_v2 = (
        model_version == ModelVersion.COPROG
        or model_version == ModelVersion.CO_TRAINING_ENSEMBLE
        or model_version == ModelVersion.CO_TRAINING_ENSEMBLE_V2
    )

    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist."
    assert os.path.exists(results_path), f"{results_path} does not exist."
    assert os.path.exists(config_path), f"{config_path} does not exist."
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist."
    assert os.path.exists(config_benchmark_file_path), f"{config_benchmark_file_path} does not exist."
    assert os.path.exists(config_model_file_path), f"{config_model_file_path} does not exist."
    if pretrained_model_dir is not None and is_coprog_or_co_training_ensemble_v2:
        assert os.path.exists(pretrained_model_dir), f"{pretrained_model_dir} does not exist."

    if run_name != "":
        results_path = os.path.join(results_path, run_name)
        os.makedirs(results_path, exist_ok=True)

        checkpoints_path = os.path.join(checkpoints_path, run_name)
        os.makedirs(checkpoints_path, exist_ok=True)

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

    necessary_dataset_keys = get_necessary_dataset_keys(model_version)
    necessary_model_keys = get_necessary_model_keys(model_version)
    train_model_func = get_train_model_method(model_version)

    has_training_params = False
    with open(config_model_file_path, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
        if any("training_params" in _cfg.get(sub, {}) for sub in cmapss_files):
            has_training_params = True

    necessary_training_keys = get_necessary_training_keys(model_version) if has_training_params else []

    gpu_aware_versions = {
        ModelVersion.COPROG,
        ModelVersion.CO_TRAINING_ENSEMBLE,
        ModelVersion.CO_TRAINING_ENSEMBLE_V2,
    }

    benchmark_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file_path = os.path.join(results_path, f"log_model_{model_version.value}_{benchmark_datetime}.txt")

    for sub_dataset in cmapss_files:
        secure_save_for_sub_dataset_rows = []

        for censored_percentage in censored_percentages:
            # secure_save_for_censored_percentage_rows = []

            # We don't wan't to iterate for no reason when censored == 0.0
            broken_percentages_tmp = [0.0] if censored_percentage == 0.0 else broken_percentages

            for broken_percentage in broken_percentages_tmp:
                extra_params = {}
                if model_version in gpu_aware_versions:
                    extra_params["gpu_ids"] = gpu_ids

                if pretrained_model_dir is not None and is_coprog_or_co_training_ensemble_v2:
                    pretrained_model_dir_for_current_run = pretrained_model_dir

                    sub_dataseet_censored_broken_run_name = f"{sub_dataset}-censored_{censored_percentage}-broken_{broken_percentage}"

                    pretrained_model_dir_for_current_run = os.path.join(pretrained_model_dir_for_current_run, sub_dataseet_censored_broken_run_name)
                    os.makedirs(pretrained_model_dir_for_current_run, exist_ok=True)

                    extra_params["pretrained_model_dir"] = pretrained_model_dir_for_current_run

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

                if has_training_params:
                    training_params = extract_training_params_from_config(
                        config_path=config_model_file_path,
                        sub_dataset=sub_dataset,
                        necessary_keys=necessary_training_keys,
                    )
                else:
                    training_params = {}

                try:
                    rmse, score = train_model_func(
                        checkpoints_path=checkpoints_path,
                        results_path=results_path,
                        model_version=model_version,
                        dataset_root=dataset_root,
                        sub_dataset=sub_dataset,
                        percent_of_broken_data=broken_percentage,
                        percent_of_censored_data=censored_percentage,
                        dataset_params=dataset_params,
                        dataset_version=DatasetVersion.CMAPSS,
                        training_params=training_params,
                        models_params=model_params,
                        datetime_for_folders=benchmark_datetime,
                        force_load_from_cache=False, # No cache for CMAPSS
                        **extra_params,
                    )
                except Exception as e:
                    rmse = None
                    score = None

                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"=== {model_version.value} run ===\n")
                        f.write(f"Datetime: {benchmark_datetime}\n")
                        f.write(f"Sub-dataset: {sub_dataset}\n")
                        f.write(f"Percent censored: {censored_percentage}\n")
                        f.write(f"Percent broken: {broken_percentage}\n")
                        f.write(f"Error: {e}\n")
                        traceback.print_exc(file=f)
                        f.write("================================\n")

                    print(f"An error occured for subdataset: {sub_dataset}, censored: {censored_percentage}, broken: {broken_percentage} :")
                    print(f"Error: {e}")
                    traceback.print_exc()

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
            return train_model_coprog
        case ModelVersion.CNN:
            return utils_transformer_lstm.train_model
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return utils_cotraining_ensemble.train_model
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return train_model_conformal_cotraining_ensemble
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all semi-supervised experiments."
    )
    parser.add_argument(
        "--model-version",
        required=True,
        choices=[
            "transformer_lstm",
            "lstm",
            "autoencoder",
            "metric",
            "rsf",
            "pyclus",
            "coprog",
            "cnn",
            "co_training_ensemble",
            "co_training_ensemble_v2"
        ],
        help="The model to train",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="path to config",
    )
    parser.add_argument(
        "--checkpoints-path",
        required=True,
        help="path to checkpoints",
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="path to results",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of CMAPSS data files",
    )
    parser.add_argument(
        "--benchmark-version",
        default="default",
        help="The benchmark version",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="The name of the run",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "GPU id(s) to train on (COPROG and co-training ensembles only). Omit to use a "
            "single GPU (auto). Give one id (e.g. --gpu-ids 0) to pin to that GPU, or several "
            "(e.g. --gpu-ids 0 1) to train the models in parallel across those GPUs."
        ),
    )
    parser.add_argument(
        "--pretrained-model-dir",
        default=None,
        help="Path to the directory containing pretrained models for CoTrainingEnsemble_v2 and Coprog",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)

    logger.info(
        "Training started — model=%s  benchmark_version=%s",
        args.model_version, args.benchmark_version,
    )

    model_version = ModelVersion(args.model_version)

    reproduce_result(
        config_path=args.config_path,
        checkpoints_path=args.checkpoints_path,
        results_path=args.results_path,
        dataset_root=args.dataset_root,
        model_version=model_version,
        benchmark_version=args.benchmark_version,
        run_name=args.run_name,
        gpu_ids=args.gpu_ids,
        pretrained_model_dir=args.pretrained_model_dir,
    )

if __name__ == "__main__":
    main()
