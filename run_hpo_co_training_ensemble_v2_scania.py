"""CLI entry point for the CoTrainingEnsemble_v2 hyperparameter benchmark on Scania.

Same argument shape as ``run_train_scania.py``, plus the sweep-specific flags. The benchmark
config is resolved as
``<config-path>/<benchmark-version>/hyper_parameter_optimization_co_training_ensemble_v2.json``.

Usage:
    python run_hpo_co_training_ensemble_v2_scania.py --config-path scania/config \
        --checkpoints-path ./checkpoints --results-path ./outputs \
        --dataset-root ./data/Scania_component_X \
        --dataset-cache-dir ./data/Scania_component_X/scania_cache \
        --benchmark-version default
"""

import argparse
import logging
import os

from scania.utils import HPO_CONFIG_FILE, run_hyper_parameter_optimization

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Grid hyperparameter benchmark for CoTrainingEnsemble_v2 on Scania. Trains the base "
            "models once, then replays the co-training loop for every hyperparameter combination."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-path", required=True, help="path to config")
    parser.add_argument("--checkpoints-path", required=True, help="path to checkpoints")
    parser.add_argument("--results-path", required=True, help="path to results")
    parser.add_argument("--dataset-root", required=True, help="Root directory of Scania data files")
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Cache directory of the Scania dataset (defaults to <dataset-root>/scania_cache)",
    )
    parser.add_argument(
        "--force-load-from-cache",
        action="store_true",
        help=(
            "Pin the sweep to the dataset cache in --dataset-cache-dir: load its splits as-is, "
            "never re-preprocess and never overwrite it. Every split-defining dataset param is "
            "taken from the cache's manifest.json instead of the benchmark config; only "
            "num_workers, pin_memory, batch_size, shuffle_loader and return_sequence_label still "
            "come from the config. Errors out if the cache is missing or incomplete."
        ),
    )
    parser.add_argument("--benchmark-version", default="default", help="The benchmark version")
    parser.add_argument("--run-name", default="", help="The name of the RUN")
    parser.add_argument(
        "--start-config",
        type=int,
        default=1,
        help="1-based index of the first configuration to run",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to run from --start-config (default: all)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_FOLDER",
        help=(
            "Name of an existing run folder under --results-path to continue, e.g. "
            "'hyper_parameter_opti_co_training_ensemble_v2_2026_07_31'. Configurations already "
            "completed there are skipped. Required to continue a sweep on a later day, since a "
            "new run is named after the current date. Errors out if the folder does not exist."
        ),
    )
    parser.add_argument(
        "--pretrained-models-dir",
        default=None,
        help=(
            "Directory holding a previously saved set of shared initial models to reuse. Defaults "
            "to 'initial_models' inside the run folder (trained there on the first run)."
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help=(
            "Single GPU id to train on. Omit for auto. The sweep is sequential by design: every "
            "opt-in lever of CoTrainingEnsemble_v2 is sequential-only."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Resolve the benchmark config and run the sweep."""
    args = _parse_args()

    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)

    config_file_path = os.path.join(args.config_path, args.benchmark_version, HPO_CONFIG_FILE)
    assert os.path.exists(config_file_path), f"{config_file_path} does not exist."

    logger.info(
        "Hyperparameter optimization started — benchmark_version=%s  config=%s",
        args.benchmark_version, config_file_path,
    )

    run_hyper_parameter_optimization(
        config_file_path=config_file_path,
        checkpoints_path=args.checkpoints_path,
        results_path=args.results_path,
        dataset_root=args.dataset_root,
        cache_dir=args.dataset_cache_dir,
        run_name=args.run_name,
        start_config=args.start_config,
        max_configs=args.max_configs,
        resume=args.resume,
        pretrained_models_dir=args.pretrained_models_dir,
        gpu_id=args.gpu_id,
        force_load_from_cache=args.force_load_from_cache,
    )


if __name__ == "__main__":
    main()
