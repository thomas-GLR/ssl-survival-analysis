"""
run_hpo_dynamic_deephit_scania.py
──────────────────────────────────
CLI entry point for Optuna hyperparameter optimisation of Dynamic DeepHit (TensorFlow) on the
Scania Component X dataset.

A dedicated script rather than a `run_hpo_scania.py` model choice: Dynamic DeepHit's manual TF
training loop (`scania/utils/utils_dynamic_deephit.py`) has nothing in common with the PyTorch
Lightning `Trainer.fit` pattern every model registered in `scania/hpo/optuna_search.py` uses, and it
optimises a different objective (validation C-index, maximize) with a config-driven search space
instead of hard-coded `trial.suggest_*` calls.

Usage:
    python run_hpo_dynamic_deephit_scania.py \\
        --config-path scania/config --benchmark-version default \\
        --dataset-root ./data/Scania_component_X --dataset-cache-dir ./data/Scania_component_X/scania_cache_ddh \\
        --n-trials 50 --output-dir ./outputs/hpo_dynamic_deephit
"""

from __future__ import annotations

import argparse
import logging
import os

from scania.hpo.optuna_search_dynamic_deephit import (
    best_params,
    run_search,
    save_study_results,
    summary_table,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Dynamic DeepHit Scania HPO — Optuna search maximizing validation C-index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        default="scania/config",
        help="Root config directory; the base config and the Optuna search space are resolved as "
             "<config-path>/<benchmark-version>/dynamic_deephit.json and "
             "<config-path>/<benchmark-version>/dynamic_deephit_search_space.json",
    )
    parser.add_argument(
        "--benchmark-version",
        default="default",
        help="Benchmark version folder under --config-path",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of the Scania data files",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Dataset cache directory (defaults to <dataset-root>/scania_cache)",
    )
    parser.add_argument(
        "--force-load-from-cache",
        action="store_true",
        help="Pin the run to the dataset cache in --dataset-cache-dir instead of preprocessing",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=int(os.getenv("N_TRIALS", "50")),
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "./outputs/hpo_dynamic_deephit"),
        help="Directory for the Optuna DB and result files",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL. Defaults to sqlite:///<output-dir>/optuna.db. "
            "For parallel runs use postgresql://user:pass@host/db"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the Dynamic DeepHit Scania Optuna search and persist its results."""
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_config_path = os.path.join(args.config_path, args.benchmark_version)
    base_config_path = os.path.join(benchmark_config_path, "dynamic_deephit.json")
    search_space_path = os.path.join(benchmark_config_path, "dynamic_deephit_search_space.json")

    # Absolute path + POSIX-slash normalization: unambiguous on both Windows and POSIX, matching
    # run_hpo_scania.py's sqlite URL construction.
    db_path = os.path.abspath(os.path.join(args.output_dir, "optuna.db"))
    storage = args.storage or f"sqlite:///{db_path.replace(os.sep, '/')}"

    logger.info(
        "HPO started — model=dynamic_deephit  config=%s  search_space=%s  trials=%d  storage=%s",
        base_config_path, search_space_path, args.n_trials, storage,
    )

    study = run_search(
        base_config_path=base_config_path,
        search_space_path=search_space_path,
        dataset_root=args.dataset_root,
        n_trials=args.n_trials,
        cache_dir=args.dataset_cache_dir,
        force_load_from_cache=args.force_load_from_cache,
        storage=storage,
    )

    summary_table(study)
    logger.info("Best params: %s", best_params(study))
    save_study_results(study, args.output_dir)


if __name__ == "__main__":
    main()
