"""
optuna_search.py
────────────────
Optuna hyperparameter search for the Scania Component X dataset + PyTorch
Lightning.

Design (mirrors ``C_MAPSS/hpo/optuna_search.py`` but adapted to Scania):
  - Model registry: add a model with ``@register_model("name", counter_mode=...)``.
    Each model carries its own ``ModelSpec`` (feature ``counter_mode``,
    ``sequence_len`` search bounds, whether to standardize the RUL target, and a
    builder that suggests any architecture hyperparameters and returns an
    ``nn.Module``).
  - Single-objective only: minimise ``val_rmse`` with a TPE sampler and Hyperband
    pruning. ``val_score`` / ``test_rmse`` / ``test_score`` are recorded as trial
    user-attrs for inspection but never optimised.
  - Scania has no sub-datasets (unlike CMAPSS FD001-FD004), so there is a single
    study per model; ``input_size`` (feature count) is read from the data module
    at runtime rather than hard-coded.

Usage:
    study = run_search("cnn", n_trials=50, data_dir="data/Scania_component_X")
    print(best_params(study))

Dependencies:
    pip install optuna optuna-integration pytorch-lightning torchmetrics
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch

import optuna
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

import torch.nn as nn
from torch.utils.data import DataLoader

from scania.dataset import ScaniaDataModule
from scania.lightning_module.BasicLightningModule import BasicLightningModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

# A builder receives (trial, feature_num, sequence_len) and returns an nn.Module.
# feature_num is known only after the data module is built (it depends on
# counter_mode), and sequence_len is a searched hyperparameter, so both are
# passed in rather than suggested inside the builder.
ModelBuilder = Callable[[optuna.Trial, int, int], nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    """Per-model HPO configuration.

    :param counter_mode: fixed ScaniaDataModule counter feature mode
        ("delta" | "cumulative" | "both"). Not a searched hyperparameter.
    :param seq_len_range: inclusive (min, max) bounds for the searched
        ``sequence_len`` hyperparameter.
    :param rul_target_standardization: whether to standardize the RUL target
        (train targets) and de-normalize predictions in BasicLightningModule.
    :param build: builder returning the nn.Module for a trial.
    """

    counter_mode: str
    seq_len_range: Tuple[int, int]
    rul_target_standardization: bool
    build: ModelBuilder


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(
    name: str,
    *,
    counter_mode: str = "both",
    seq_len_range: Tuple[int, int] = (30, 50),
    rul_target_standardization: bool = True,
) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator registering a model builder together with its ``ModelSpec``.

    The builder receives ``(trial, feature_num, sequence_len)`` and returns an
    ``nn.Module``. Any architecture hyperparameters are suggested inside the
    builder via ``trial.suggest_*``.

    :param name: unique registry key (align with ModelVersion values, e.g. "cnn").
    :param counter_mode: fixed feature mode for this model.
    :param seq_len_range: inclusive bounds for the ``sequence_len`` search.
    :param rul_target_standardization: whether to standardize the RUL target.
    :return: the decorator that stores the builder under ``name``.
    """

    def decorator(fn: ModelBuilder) -> ModelBuilder:
        if name in _MODEL_REGISTRY:
            logger.warning("Overwriting existing builder for '%s'.", name)
        _MODEL_REGISTRY[name] = ModelSpec(
            counter_mode=counter_mode,
            seq_len_range=seq_len_range,
            rul_target_standardization=rul_target_standardization,
            build=fn,
        )
        return fn

    return decorator


def list_models() -> list[str]:
    """Return the list of registered model names."""
    return list(_MODEL_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────


@register_model("cnn", counter_mode="both", seq_len_range=(30, 50))
def _build_cnn(trial: optuna.Trial, feature_num: int, sequence_len: int) -> nn.Module:
    """1D CNN model.

    CNN1D has no tunable architecture hyperparameters and is sequence-length
    agnostic (AdaptiveAvgPool1d), so only ``feature_num`` matters here.

    :param trial: the Optuna trial (unused — no architecture params to suggest).
    :param feature_num: number of input features (fixed by counter_mode).
    :param sequence_len: window length (unused by CNN1D at construction).
    :return: an initialised CNN1D module.
    """
    from models import CNN1D

    return CNN1D(num_features=feature_num, output_dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Scania data
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _build_datamodule_cached(
    sequence_len: int,
    counter_mode: str,
    standardize_target: bool,
    data_dir: str,
    cache_dir: str,
) -> Tuple[ScaniaDataModule, float, float]:
    """Build and ``setup()`` a ScaniaDataModule for a (sequence_len, counter_mode)
    pair, cached across trials.

    Everything the datasets depend on (splits, windowing, normalization) is fixed
    except ``sequence_len`` and ``counter_mode``; ``batch_size`` only affects the
    DataLoader built on top, so it is passed later. Without this cache an HPO run
    would rebuild (split + windowing + normalization) on every trial. Each
    (counter_mode, sequence_len) also gets its own on-disk cache sub-directory so
    the many ``sequence_len`` values do not clobber each other's cache.

    When ``standardize_target`` is True, the RUL target mean/std are computed from
    the training (uncensored) window labels and returned so BasicLightningModule
    can train in normalized target space and de-normalize predictions.

    :param sequence_len: window length.
    :param counter_mode: ScaniaDataModule counter feature mode.
    :param standardize_target: whether to compute target mean/std.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory (a per-config sub-dir is created).
    :return: (setup data module, target_mean, target_std).
    """
    sub_cache_dir = os.path.join(cache_dir, f"cm={counter_mode}_sl={sequence_len}")

    # batch_size=None: the DataLoaders are built per-trial from the underlying
    # datasets with the trial's batch_size, so the module's own batch_size is
    # never used. num_workers=0 is deliberate: the datasets are fully in-memory
    # TensorDatasets, and worker subprocesses leak across trials (they eventually
    # crash with "can only test a child process").
    dm = ScaniaDataModule(
        data_dir=data_dir,
        batch_size=None,
        sequence_len=sequence_len,
        counter_mode=counter_mode,
        cache_dir=sub_cache_dir,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    target_mean, target_std = 0.0, 1.0
    if standardize_target:
        train_targets = torch.cat(
            [y for _, y in dm.train_set.get_data_loader_without_censored_data(batch_size=4096)]
        )
        target_mean = float(train_targets.mean())
        target_std = float(train_targets.std())
        if target_std < 1e-6:
            target_std = 1.0
        logger.info(
            "RUL target standardization : mean=%.4f std=%.4f", target_mean, target_std
        )

    return dm, target_mean, target_std


def get_dataloaders(
    sequence_len: int,
    counter_mode: str,
    standardize_target: bool,
    batch_size: int,
    data_dir: str,
    cache_dir: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, float, float]:
    """Build train / val / test DataLoaders for a Scania configuration.

    :param sequence_len: window length.
    :param counter_mode: ScaniaDataModule counter feature mode.
    :param standardize_target: whether to standardize the RUL target.
    :param batch_size: DataLoader batch size for this trial.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory.
    :return: (train_dl, val_dl, test_dl, feature_num, target_mean, target_std).
    """
    dm, target_mean, target_std = _build_datamodule_cached(
        sequence_len, counter_mode, standardize_target, data_dir, cache_dir
    )

    train_dl = dm.train_set.get_data_loader_without_censored_data(
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_dl = dm.val_set.get_data_loader_without_censored_data(
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_dl = dm.test_set.get_data_loader_without_censored_data(
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_dl, val_dl, test_dl, dm.feature_num, target_mean, target_std


# ──────────────────────────────────────────────────────────────────────────────
# Objective function (closure)
# ──────────────────────────────────────────────────────────────────────────────


def _metric_from_trainer(trainer: Trainer, key: str) -> float:
    """Safely extract a scalar metric logged by the Lightning module.

    :param trainer: the Lightning Trainer after fit/test.
    :param key: the metric name to read from ``trainer.callback_metrics``.
    :return: the metric as a float (``inf`` if missing).
    """
    val = trainer.callback_metrics.get(key, float("inf"))
    return val.item() if hasattr(val, "item") else float(val)


def _make_objective(
    model_name: str,
    data_dir: str,
    cache_dir: str,
    max_epochs: int,
) -> Callable[[optuna.Trial], float]:
    """Build the single-objective Optuna closure for a given model.

    Optimises ``val_rmse`` (minimize). ``val_score`` and the test metrics are
    stored as trial user-attrs for inspection only.

    :param model_name: registered model name.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory.
    :param max_epochs: max epochs per trial.
    :return: the objective callable returning ``val_rmse``.
    """
    spec = _MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        # ── Training / data hyperparameters ────────────────────────────────
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        sequence_len = trial.suggest_int("sequence_len", *spec.seq_len_range)

        # ── Data ────────────────────────────────────────────────────────────
        train_dl, val_dl, test_dl, feature_num, target_mean, target_std = get_dataloaders(
            sequence_len=sequence_len,
            counter_mode=spec.counter_mode,
            standardize_target=spec.rul_target_standardization,
            batch_size=batch_size,
            data_dir=data_dir,
            cache_dir=cache_dir,
        )

        # ── Model ─────────────────────────────────────────────────────────
        net = spec.build(trial, feature_num, sequence_len)
        module = BasicLightningModule(
            lr=lr, model=net, target_mean=target_mean, target_std=target_std
        )

        # ── Callbacks ─────────────────────────────────────────────────────
        callbacks = [
            EarlyStopping(monitor="val_rmse", patience=20, mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="val_rmse"),
        ]

        # ── Trainer ───────────────────────────────────────────────────────
        # enable_checkpointing=False: HPO optimises on the logged val_rmse and
        # never reloads a checkpoint, so writing one per trial only litters the
        # project's ./checkpoints directory.
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            callbacks=callbacks,
        )

        # ── Training ──────────────────────────────────────────────────────
        trainer.fit(module, train_dl, val_dl)

        # ── Validation metrics — the optimisation objective ────────────────
        val_rmse = _metric_from_trainer(trainer, "val_rmse")
        val_score = _metric_from_trainer(trainer, "val_score")

        # ── Test metrics — stored for post-hoc analysis only ──────────────
        results = trainer.test(module, test_dl, verbose=False)[0]
        test_rmse = results.get("test_rmse", float("inf"))
        test_score = results.get("test_score", float("inf"))

        trial.set_user_attr("val_rmse", val_rmse)
        trial.set_user_attr("val_score", val_score)
        trial.set_user_attr("test_rmse", test_rmse)
        trial.set_user_attr("test_score", test_score)

        return val_rmse

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def run_search(
    model_name: str,
    *,
    n_trials: int = 100,
    data_dir: str = "data/Scania_component_X",
    cache_dir: Optional[str] = None,
    max_epochs: int = 100,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = 1,
) -> optuna.Study:
    """Run single-objective hyperparameter search for one Scania model.

    Optimises ``val_rmse`` (minimize) with a TPE sampler and Hyperband pruning.

    :param model_name: key registered via ``@register_model`` (e.g. "cnn").
    :param n_trials: number of Optuna trials.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory (defaults to ``<data_dir>/scania_cache``).
    :param max_epochs: upper bound for epochs per trial (also Hyperband max_resource).
    :param study_name: custom study name (auto-generated if None).
    :param storage: Optuna storage URL (e.g. "sqlite:///optuna.db") for resuming.
    :param n_jobs: parallel trials (keep 1 to avoid GPU-memory conflicts).
    :return: the completed ``optuna.Study``.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered. Available models: {list_models()}"
        )

    if cache_dir is None:
        cache_dir = os.path.join(data_dir, "scania_cache")

    study_name = study_name or f"{model_name}_scania"

    # TPE + Hyperband: efficient for single-objective search on a limited budget.
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10,
        max_resource=max_epochs,
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,  # resume an existing study with the same name + storage
    )

    objective = _make_objective(model_name, data_dir, cache_dir, max_epochs)

    logger.info(
        "Starting search: model=%s  trials=%d  mode=single-objective (val_rmse)",
        model_name,
        n_trials,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    _log_results(study, study_name)
    return study


def best_params(study: optuna.Study) -> dict:
    """Return the best hyperparameters from a completed single-objective study.

    :param study: a completed Optuna study.
    :return: dict of hyperparameters for the best trial.
    """
    t = study.best_trial
    logger.info(
        "Best trial #%d | val_rmse=%.4f | params=%s", t.number, t.value, t.params
    )
    return t.params


def summary_table(study: optuna.Study) -> None:
    """Log a one-line summary of the best trial of a study.

    :param study: a completed Optuna study.
    """
    header = f"{'val_RMSE':>10} {'val_Score':>12} {'test_RMSE':>10} {'test_Score':>12} {'Trial':>7}"
    logger.info("\n" + "=" * len(header))
    logger.info("SUMMARY  (val = HPO objective  |  test = post-hoc user-attr)")
    logger.info(header)
    logger.info("-" * len(header))
    t = study.best_trial
    val_rmse = t.user_attrs.get("val_rmse", t.value)
    val_score = t.user_attrs.get("val_score", float("nan"))
    test_rmse = t.user_attrs.get("test_rmse", float("nan"))
    test_score = t.user_attrs.get("test_score", float("nan"))
    logger.info(
        f"{val_rmse:>10.4f} {val_score:>12.4f} {test_rmse:>10.4f}"
        f" {test_score:>12.4f} {t.number:>7d}"
    )
    logger.info("=" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────────────────────


def save_study_results(study: optuna.Study, model_name: str, output_dir: str) -> None:
    """Save all completed trials and the best params of a study to ``output_dir``.

    Files written:
      ``<output_dir>/<model>_trials.csv``       — all completed trials
      ``<output_dir>/<model>_best_params.json`` — best trial params

    :param study: a completed Optuna study.
    :param model_name: registered model name (used in the filenames).
    :param output_dir: directory to write the result files to.
    """
    os.makedirs(output_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials for %s — skipping save.", model_name)
        return

    rows = []
    for t in completed:
        row = {"trial": t.number}
        row.update(t.params)
        row["val_rmse"] = t.user_attrs.get("val_rmse", t.value)
        row["val_score"] = t.user_attrs.get("val_score", float("nan"))
        row["test_rmse"] = t.user_attrs.get("test_rmse", float("nan"))
        row["test_score"] = t.user_attrs.get("test_score", float("nan"))
        rows.append(row)

    trials_path = os.path.join(output_dir, f"{model_name}_trials.csv")
    pd.DataFrame(rows).to_csv(trials_path, index=False)
    logger.info("Trials saved → %s", trials_path)

    best_path = os.path.join(output_dir, f"{model_name}_best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params(study), f, indent=2)
    logger.info("Best params saved → %s", best_path)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _log_results(study: optuna.Study, name: str) -> None:
    """Log the best trial of a finished study.

    :param study: the finished Optuna study.
    :param name: the study name (for the log line).
    """
    logger.info("=" * 60)
    logger.info("Study: %s | finished trials: %d", name, len(study.trials))
    bt = study.best_trial
    logger.info("Best trial #%d  val_rmse=%.4f  params=%s", bt.number, bt.value, bt.params)
