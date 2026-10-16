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
    user-attrs for inspection but never optimised. The ``*_score`` attrs hold the
    class-based Scania cost (:mod:`scania.metrics`), not an RMSE-like quantity.
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

import gc
import logging
import os
from collections import OrderedDict
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from scania.dataset import ScaniaDataModule

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Scania data
# ──────────────────────────────────────────────────────────────────────────────

def _target_stats(dm: ScaniaDataModule) -> Tuple[float, float]:
    ds = dm.train_set
    y = ds.label_array[ds.is_censored_array == 0].astype("float64")  # 1-D unless return_sequence_label
    mean, std = float(y.mean()), float(y.std(ddof=1))  # ddof=1 matches torch.std
    return mean, (std if std >= 1e-6 else 1.0)

_DM_CACHE: "OrderedDict[tuple, ScaniaDataModule]" = OrderedDict()
_DM_CACHE_SIZE = 2   # raise if you have RAM to spare after the ScaniaDataset fixes


# @lru_cache(maxsize=None)
def _build_datamodule_cached(
    sequence_len: int,
    counter_mode: str,
    histogram_mode: str,
    include_histograms: bool,
    standardize_target: bool,
    data_dir: str,
    cache_dir: str,
    seed: int,
) -> ScaniaDataModule | tuple[ScaniaDataModule, float, float]:
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
    :param histogram_mode: ScaniaDataModule histogram feature mode.
    :param include_histograms: whether to include histogram features.
    :param standardize_target: whether to compute target mean/std.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory (a per-config sub-dir is created).
    :param seed: random seed for the vehicle split and dataset processing.
    :return: (setup data module, target_mean, target_std).
    """
    key = (sequence_len, counter_mode, histogram_mode, include_histograms, data_dir, cache_dir, seed)
    if key in _DM_CACHE:
        _DM_CACHE.move_to_end(key)
        return _DM_CACHE[key]

    while len(_DM_CACHE) >= _DM_CACHE_SIZE:   # evict BEFORE building
        _DM_CACHE.popitem(last=False)
    gc.collect()

    sub_cache_dir = os.path.join(cache_dir, f"cm={counter_mode}_sl={sequence_len}_seed={seed}")

    # batch_size=None: the DataLoaders are built per-trial from the underlying
    # datasets with the trial's batch_size, so the module's own batch_size is
    # never used. num_workers=0 is deliberate: the datasets are fully in-memory
    # TensorDatasets, and worker subprocesses leak across trials (they eventually
    # crash with "can only test a child process").
    # seed=seed (not the ScaniaDataModule default of None): without it the vehicle split
    # is drawn from an unseeded RNG and, worse, `seed: null` lands in the cache manifest, so a
    # cache written from a *different* random split still validates on the next run.
    dm = ScaniaDataModule(
        data_dir=data_dir,
        batch_size=None,
        sequence_len=sequence_len,
        counter_mode=counter_mode,
        histogram_mode=histogram_mode,
        include_histograms=include_histograms,
        val_rate=0.15,
        test_rate=0.15,
        cache_dir=sub_cache_dir,
        num_workers=7,
        pin_memory=True,
        seed=seed,
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


def get_dataloaders_scania(
    sequence_len: int,
    counter_mode: str,
    histogram_mode: str,
    include_histograms: bool,
    standardize_target: bool,
    batch_size: int,
    data_dir: str,
    cache_dir: str,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, float, float]:
    """Build train / val / test DataLoaders for a Scania configuration.

    :param sequence_len: window length.
    :param counter_mode: ScaniaDataModule counter feature mode.
    :param histogram_mode: ScaniaDataModule histogram feature mode.
    :param include_histograms: whether to include histogram features.
    :param standardize_target: whether to standardize the RUL target.
    :param batch_size: DataLoader batch size for this trial.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory.
    :param seed: random seed for the dataset split and cache.
    :return: (train_dl, val_dl, test_dl, feature_num, target_mean, target_std).
    """
    dm, target_mean, target_std = _build_datamodule_cached(
        sequence_len,
        counter_mode,
        histogram_mode,
        include_histograms,
        False,
        data_dir,
        cache_dir,
        seed=seed,
    )

    target_mean, target_std = _target_stats(dm) if standardize_target else (0.0, 1.0)

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
