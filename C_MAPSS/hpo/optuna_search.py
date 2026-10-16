"""
optuna_search.py
────────────────
Optuna hyperparameter search for CMAPSS + PyTorch Lightning.

Features:
  - Model registry: add your own model with @register_model("name")
  - Optimize both architecture and training hyperparameters per trial
  - Multi-objective mode : Pareto front RMSE + CMAPSS Score (NSGA-II, no pruning)
  - Single-objective mode: val_RMSE + Hyperband pruning (faster)
  - Run on a single subset or all four FD001-FD004

Usage:
    # Single subset
    study = run_search("transformer", "FD001", n_trials=50)
    print(best_params(study))

    # All subsets
    studies = run_all_subsets("lstm", n_trials=100, storage="sqlite:///optuna.db")

    # CLI
    python optuna_search.py transformer --subset FD001 --n_trials 50
    python optuna_search.py lstm --subset all --n_trials 100 --single_objective

Dependencies:
    pip install optuna optuna-integration pytorch-lightning torchmetrics
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

# optuna-integration >= 3.0 ships as a separate package
# try:
#     from optuna.integration import PyTorchLightningPruningCallback
# except ImportError:
from torch.utils.data import DataLoader


@lru_cache(maxsize=None)
def _build_datasets_cached(
        sub_dataset: str,
        sequence_len: int,
        data_dir: str,
        seed: int,
):
    """
    Build (train, test, valid) CMAPSSDataset objects for a (subset, sequence_len)
    pair, cached across trials.

    Everything these datasets depend on (seed, normalization, clustering, RUL,
    windowing) is fixed except `subset` and `sequence_len` — batch_size only
    affects the DataLoader built on top, not the dataset itself. Without this
    cache, an HPO run rebuilds the full dataset (parsing + KMeans + normalization
    + windowing) from scratch on every single trial, which dominates wall-clock
    time for FD002/FD004 (measured ~5.6s/build vs ~1.3s for FD001).
    """
    from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader

    return CMAPSSLoader.get_datasets(
        dataset_root=data_dir,
        sub_dataset=sub_dataset,
        sequence_len=sequence_len,
        seed=seed,
        max_rul=125,
        return_sequence_label=False,
        norm_type="z-score",
        cluster_operations=True,
        norm_by_operations=True,
        include_cols=[],
        exclude_cols=[],
        return_id=False,
        val_rate=0.2,
        use_only_final_on_test=True,
        use_max_rul_on_test=True,
        use_max_rul_on_valid=True,
        percent_of_broken_data=None,
        percent_of_censored_data=0.,
    )


def get_dataloaders_cmapss(
        sub_dataset: str,
        batch_size: int,
        sequence_len: int,
        data_dir: str,
        seed: int,
) -> Tuple[DataLoader, DataLoader | None, DataLoader]:
    """
    Build train / val / test DataLoaders for a CMAPSS subset.
    *** Replace this function body with your own dataset logic. ***
    """
    train_dataset, test_dataset, valid_dataset, _ = _build_datasets_cached(
        sub_dataset,
        sequence_len,
        data_dir,
        seed
    )

    # num_workers=0: the underlying dataset is a TensorDataset already fully
    # materialised in memory, so worker subprocesses add IPC overhead without
    # any parallelism benefit. Creating one DataLoader with num_workers>0 per
    # trial (x3, x n_trials) also leaks worker Process objects across fork()
    # boundaries, which eventually crashes with "AssertionError: can only
    # test a child process" once the garbage collector reaps a stale
    # DataLoader iterator inside a forked worker from a *later* trial.
    return (
        train_dataset.get_data_loader_without_censored_data(batch_size=batch_size, num_workers=0, pin_memory=True),
        valid_dataset.get_data_loader_without_censored_data(batch_size=batch_size, num_workers=0,
                                                            pin_memory=True) if valid_dataset is not None else None,
        test_dataset.get_data_loader_without_censored_data(batch_size=batch_size, num_workers=0, pin_memory=True),
    )