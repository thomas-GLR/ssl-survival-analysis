"""
run_hpo.py
──────────
Docker / CLI entry point for hyperparameter optimisation.

Imports model_builders BEFORE argparse so that every @register_model
decorator has already run when list_models() is called.

Usage:
    python run_hpo.py transformer --subset FD001 --n_trials 50
    python run_hpo.py lstm        --subset all   --n_trials 100
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

# This environment variable avoid for missing cuBLAS when using deterministic algorithm.
# This happend because cuBLAS is used by every nn.Linear/matmul, so the LSTM, Transformer, and CNN's dense layers, and the backward pass
# cuBLAS has its own determinism switch that's independent of torch.use_deterministic_algorithms — it needs an environment variable set before CUDA initializes.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import optuna
import torch

from enums import DatasetVersion

torch.set_float32_matmul_precision('high')

from shared.hpo.hpo_common import (
    save_all_studies_results,
)

from shared.hpo.hpo_common import (
    list_models,
    run_search,
    best_params,
    save_study_results,
    summary_table,
)

logger = logging.getLogger(__name__)

CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CMAPSS HPO — wraps optuna_search with full model registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        choices=list_models(),
        help="Model to optimise (registered via @register_model)",
    )
    parser.add_argument(
        "--subset",
        default="all",
        help="FD001 | FD002 | FD003 | FD004 | all",
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
        default=os.getenv("DATA_DIR", "/data/CMAPSSData"),
        help="Root directory of CMAPSS data files",
    )
    parser.add_argument(
        "--output_dir",
        default=os.getenv("OUTPUT_DIR", "/outputs"),
        help="Directory for Optuna DB and result files",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL. "
            "Defaults to sqlite:////<output_dir>/optuna.db. "
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

    # Build storage URL if not provided explicitly
    # storage = args.storage or f"sqlite:////{args.output_dir}/optuna.db"

    # Build storage URL if not provided explicitly. Use an absolute filesystem
    # path so the URL is unambiguous on both Windows and POSIX (relative paths +
    # the four-slash form resolve to a non-existent root directory otherwise).
    # SQLAlchemy's sqlite URL for an absolute path is "sqlite:///" + abspath.
    db_path = os.path.abspath(os.path.join(args.output_dir, "optuna.db"))
    storage = args.storage or f"sqlite:///{db_path.replace(os.sep, '/')}"

    logger.info(
        "HPO started — model=%s  subset=%s  trials=%d  storage=%s",
        args.model, args.subset, args.n_trials, storage,
    )

    if args.subset.lower() == "all":

        studies: Dict[str, optuna.Study] = {}

        for sub_dataset in CMAPSS_SUBSETS:
            logger.info("=" * 60)
            logger.info("Subset %s", sub_dataset)

            studies[sub_dataset] = run_search(
                model_name=args.model,
                dataset_version=DatasetVersion.CMAPSS,
                data_dir=args.data_dir,
                sub_dataset=sub_dataset,
                n_trials=args.n_trials,
                max_epochs=args.max_epochs,
                storage=storage,
            )

        summary_table(studies)

        save_all_studies_results(
            studies=studies,
            model_name=args.model,
            dataset_version=DatasetVersion.CMAPSS,
            output_dir=args.output_dir
        )
    else:
        study = run_search(
            model_name=args.model,
            dataset_version=DatasetVersion.CMAPSS,
            data_dir=args.data_dir,
            sub_dataset=args.subset,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            storage=storage,
        )

        logger.info("Best params: %s", best_params(study))

        save_study_results(
            study=study,
            model_name=args.model,
            dataset_version=DatasetVersion.CMAPSS,
            sub_dataset=args.subset,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
