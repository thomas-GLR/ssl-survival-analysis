"""
run_hpo.py
──────────
Docker / CLI entry point for hyperparameter optimisation.

Imports model_builders BEFORE argparse so that every @register_model
decorator has already run when list_models() is called.

Usage:
    python run_hpo.py transformer --subset FD001 --n_trials 50
    python run_hpo.py lstm        --subset all   --n_trials 100 --single_objective
"""

from __future__ import annotations

import argparse
import logging
import os


from C_MAPSS.hpo.optuna_search import (
    best_params,
    list_models,
    run_all_subsets,
    run_search,
    save_all_studies_results,
    save_study_results,
    summary_table,
)

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--single_objective",
        action="store_true",
        default=os.getenv("SINGLE_OBJECTIVE", "false").lower() == "true",
        help="Optimise val_RMSE only (+ Hyperband pruning) instead of Pareto front",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build storage URL if not provided explicitly
    storage = args.storage or f"sqlite:////{args.output_dir}/optuna.db"

    search_kwargs = dict(
        n_trials=args.n_trials,
        data_dir=args.data_dir,
        max_epochs=args.max_epochs,
        multi_objective=not args.single_objective,
        storage=storage,
    )

    logger.info(
        "HPO started — model=%s  subset=%s  trials=%d  storage=%s",
        args.model, args.subset, args.n_trials, storage,
    )

    if args.subset.lower() == "all":
        studies = run_all_subsets(args.model, **search_kwargs)
        summary_table(studies)
        save_all_studies_results(studies, args.model, args.output_dir)
    else:
        study = run_search(args.model, args.subset, **search_kwargs)
        logger.info("Best params: %s", best_params(study))
        save_study_results(study, args.model, args.subset, args.output_dir)


if __name__ == "__main__":
    main()
