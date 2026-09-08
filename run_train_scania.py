import argparse
import logging
import os
import traceback
from datetime import datetime
from typing import Callable

from shared.utils import ModelVersion
from scania.utils import (
    get_necessary_dataset_keys,
    get_necessary_model_keys,
    extract_dataset_params_from_config,
    extract_model_params_from_config,
    extract_training_params_from_config,
    get_necessary_training_keys,
    override_dataset_params_from_cache_manifest,
)
from scania.utils import (
    train_model_lightning,
    train_model_random_survival,
    train_model_coprog,
    train_model_cotraining_ensemble,
    train_model_cotraining_ensemble_v2,
    train_model_dynamic_deephit,
)

logger = logging.getLogger(__name__)


def reproduce_result(
        config_path: str,
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        dataset_cache_dir: str,
        model_version: ModelVersion,
        benchmark_version: str = "default",
        run_name: str = "",
        gpu_ids: list[int] | None = None,
        force_load_from_cache: bool = False,
):
    config_path = f"{config_path}/{benchmark_version}"
    config_model_file_path = f"{config_path}/{model_version.value}.json"

    assert os.path.exists(checkpoints_path), f"{checkpoints_path} does not exist."
    assert os.path.exists(results_path), f"{results_path} does not exist."
    assert os.path.exists(config_path), f"{config_path} does not exist."
    assert os.path.exists(dataset_root), f"{dataset_root} does not exist."
    assert os.path.exists(config_model_file_path), f"{config_model_file_path} does not exist."

    if run_name != "":
        results_path = os.path.join(results_path, run_name)
        os.makedirs(results_path, exist_ok=True)

        checkpoints_path = os.path.join(checkpoints_path, run_name)
        os.makedirs(checkpoints_path, exist_ok=True)

    necessary_dataset_keys = get_necessary_dataset_keys(model_version)
    necessary_model_keys = get_necessary_model_keys(model_version)
    necessary_training_keys = get_necessary_training_keys(model_version)

    dataset_params = extract_dataset_params_from_config(
        config_path=config_model_file_path,
        necessary_keys=necessary_dataset_keys,
    )

    model_params = extract_model_params_from_config(
        config_model_file_path,
        necessary_keys=necessary_model_keys,
    )

    training_params = extract_training_params_from_config(
        config_model_file_path,
        necessary_keys=necessary_training_keys,
    )

    # Pinned to an existing cache: the split-defining params come from its manifest instead of
    # this model's config, so every benchmarked model sees the same vehicles. Done before the
    # try/except below on purpose -- a missing or incomplete cache must fail loudly rather than
    # be swallowed into the run log with a zero exit code.
    if force_load_from_cache:
        assert os.path.exists(dataset_cache_dir), f"{dataset_cache_dir} does not exist."
        dataset_params = override_dataset_params_from_cache_manifest(
            dataset_params=dataset_params,
            cache_dir=dataset_cache_dir,
            dataset_root=dataset_root,
        )

    train_model = _get_train_model_method(model_version)

    benchmark_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file_path = os.path.join(results_path, f"log_model_{model_version.value}_{benchmark_datetime}.txt")

    # GPU selection is only meaningful for the multi-model parallel trainers (COPROG and the
    # co-training ensembles). The other train_model functions don't accept it, so forward it
    # only for those to avoid a TypeError.
    gpu_aware_versions = {
        ModelVersion.COPROG,
        ModelVersion.CO_TRAINING_ENSEMBLE,
        ModelVersion.CO_TRAINING_ENSEMBLE_V2,
    }
    extra_params = {"gpu_ids": gpu_ids} if model_version in gpu_aware_versions else {}

    try:
        rmse, score = train_model(
            checkpoints_path=checkpoints_path,
            results_path=results_path,
            model_version=model_version,
            cache_dir=dataset_cache_dir,
            force_load_from_cache=force_load_from_cache,
            dataset_root=dataset_root,
            datetime_for_folders=benchmark_datetime,
            **dataset_params,
            **training_params,
            **model_params,
            **extra_params,
        )
    except Exception as e:
        rmse = None
        score = None

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== {model_version.value} run ===\n")
            f.write(f"Datetime: {benchmark_datetime}\n")
            f.write(f"Error: {e}\n")
            traceback.print_exc(file=f)
            f.write("================================\n")

        print(f"Error: {e}")
        traceback.print_exc()


def _get_train_model_method(model_version: ModelVersion) -> Callable:
    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
            return train_model_lightning
        case ModelVersion.LSTM:
            return train_model_lightning
        case ModelVersion.AUTOENCODER:
            raise NotImplementedError("Autoencoder model training is not implemented yet")
        case ModelVersion.METRIC:
            raise NotImplementedError("Metric model training is not implemented yet")
        case ModelVersion.RSF:
            return train_model_random_survival
        case ModelVersion.PYCLUS:
            raise NotImplementedError("PYCLUS model training is not implemented yet")
        case ModelVersion.COPROG:
            return train_model_coprog
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return train_model_cotraining_ensemble
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return train_model_cotraining_ensemble_v2
        case ModelVersion.CNN:
            return train_model_lightning
        case ModelVersion.TRANSFORMER_FEATURES:
            return train_model_lightning
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            return train_model_lightning
        case ModelVersion.DYNAMIC_DEEPHIT:
            return train_model_dynamic_deephit
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


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
            "co_training_ensemble",
            "co_training_ensemble_v2",
            "cnn",
            "transformer_features",
            "transformer_time_sequence",
            "dynamic_deephit",
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
        help="Root directory of Scania data files",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        required=True,
        help="Root directory of the cache to build Scania dataset",
    )
    parser.add_argument(
        "--force-load-from-cache",
        action="store_true",
        help=(
            "Pin the run to the dataset cache in --dataset-cache-dir: load its splits as-is, "
            "never re-preprocess and never overwrite it. Every split-defining dataset param "
            "(sequence_len, seed, data_fraction, val/test/calib rates, stratify, norm_type, "
            "counter_mode, histograms) is taken from the cache's manifest.json instead of the "
            "model config; only num_workers, pin_memory, batch_size, shuffle_loader and "
            "return_sequence_label still come from the config. Use it to benchmark several "
            "models on one identical set of vehicles. Errors out if the cache is missing or "
            "incomplete."
        ),
    )
    parser.add_argument(
        "--benchmark-version",
        default="default",
        help="The benchmark version",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="The name of the RUN"
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

    return parser.parse_args()


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
        dataset_cache_dir=args.dataset_cache_dir,
        model_version=model_version,
        benchmark_version=args.benchmark_version,
        run_name=args.run_name,
        gpu_ids=args.gpu_ids,
        force_load_from_cache=args.force_load_from_cache,
    )

if __name__ == "__main__":
    main()