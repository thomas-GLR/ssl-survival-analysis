"""
run_train_cmapss.py
──────────
Docker / CLI entry point for training models.

Usage:
    python run_train_cmapss.py transformer --subset FD001 --device cuda --config-path ../config
    --checkpoints-path ../checkpoints --results-path ../results --dataset-root ../data --benchmark-version default
"""

from __future__ import annotations

import argparse
import logging
import os

from utils import utils_cmapss

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all semi-supervised experiments."
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="device type",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        choices=[
            "transformer",
            "lstm",
            "autoencoder",
            "metric",
            "rsf",
            "pyclus",
            "coprog",
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

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)

    logger.info(
        "Training started — model=%s  subset=%s  device=%s  benchmark_version=%s",
        args.model, args.subset, args.device, args.benchmark_version,
    )

    utils_cmapss.reproduce_result(
        config_path=args.config_path,
        checkpoints_path=args.checkpoints_path,
        results_path=args.results_path,
        dataset_root=args.dataset_root,
        model_version=args.model_version,
        device=args.device,
        benchmark_version=args.benchmark_version,
    )

if __name__ == "__main__":
    main()
