"""
run_hpo_scania.py
─────────────────
CLI entry point for hyperparameter optimisation on the Scania Component X
dataset.

Imports the HPO engine BEFORE argparse so that every @register_model decorator
has already run when list_models() is called.

Single-objective only: optimises val_rmse (minimize) with TPE + Hyperband
pruning. Scania has no sub-datasets, so there is a single study per model.

Usage:
    python run_hpo_scania.py cnn --n_trials 50 \
        --data_dir ./data/Scania_component_X --output_dir ./outputs/hpo_scania
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

torch.set_float32_matmul_precision("high")


from scania.hpo.optuna_search import (
    best_params,
    list_models,
    run_search,
    save_study_results,
    summary_table,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scania HPO — wraps optuna_search with the model registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        choices=list_models(),
        help="Model to optimise (registered via @register_model)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=int(os.getenv("N_TRIALS", "50")),
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=int(os.getenv("MAX_EPOCHS", "100")),
        help="Max epochs per trial",
    )
    parser.add_argument(
        "--data_dir",
        default=os.getenv("DATA_DIR", "./data/Scania_component_X"),
        help="Root directory of Scania data files",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.getenv("CACHE_DIR", None),
        help="Base cache directory (defaults to <data_dir>/scania_cache)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.getenv("OUTPUT_DIR", "./outputs"),
        help="Directory for Optuna DB and result files",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL. Defaults to sqlite:////<output_dir>/optuna.db. "
            "For parallel runs use postgresql://user:pass@host/db"
        ),
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build storage URL if not provided explicitly. Use an absolute filesystem
    # path so the URL is unambiguous on both Windows and POSIX (relative paths +
    # the four-slash form resolve to a non-existent root directory otherwise).
    # SQLAlchemy's sqlite URL for an absolute path is "sqlite:///" + abspath.
    db_path = os.path.abspath(os.path.join(args.output_dir, "optuna.db"))
    storage = args.storage or f"sqlite:///{db_path.replace(os.sep, '/')}"

    logger.info(
        "HPO started — model=%s  trials=%d  storage=%s",
        args.model,
        args.n_trials,
        storage,
    )

    study = run_search(
        args.model,
        n_trials=args.n_trials,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_epochs=args.max_epochs,
        storage=storage,
    )

    summary_table(study)
    logger.info("Best params: %s", best_params(study))
    save_study_results(study, args.model, args.output_dir)


if __name__ == "__main__":
    main()
