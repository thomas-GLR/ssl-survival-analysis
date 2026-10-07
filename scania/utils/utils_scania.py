import json
import os

import numpy as np
import pandas as pd
import torch

from constants import necessary_keys_scania
from scania.metrics import scania_score
from shared.enums import ModelVersion
from shared.utils import necessary_keys as shared_necessary_keys

RUN_PARAMETERS = "run_parameters.json"

CACHE_MANIFEST_FILE = "manifest.json"

# Dataset params that describe how batches are served rather than what is in them. They are
# absent from ``ScaniaDataModule._cache_config`` (they do not change the cached CSVs), so a
# force-loaded run keeps them under the model config's control.
LOADER_ONLY_DATASET_KEYS = (
    "num_workers",
    "pin_memory",
    "batch_size",
    "shuffle_loader",
    "return_sequence_label",
)

# Manifest config entries that are not ``ScaniaDataModule`` constructor params: a cache-format
# marker and a list derived from counter_mode/include_histograms/histogram_mode.
_NON_PARAM_MANIFEST_CONFIG_KEYS = ("cache_version", "feature_cols")


def get_necessary_dataset_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_dataset_keys(model_version, necessary_keys_scania)


def get_necessary_model_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_model_keys(model_version, necessary_keys_scania)


def get_necessary_training_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_training_keys(model_version, necessary_keys_scania)


def resolve_cache_dir(cache_dir: str | None, dataset_root: str) -> str:
    """Resolve the Scania cache directory the way ``ScaniaDataModule`` does.

    Args:
        cache_dir: Explicit cache directory, or ``None`` to fall back to the default.
        dataset_root: Root directory of the Scania data files.

    Returns:
        The cache directory, defaulting to ``<dataset_root>/scania_cache``.
    """
    return cache_dir or os.path.join(dataset_root, "scania_cache")


def override_dataset_params_from_cache_manifest(
        dataset_params: dict,
        cache_dir: str | None,
        dataset_root: str,
) -> dict:
    """Replace the split-defining dataset params with the ones recorded in a cache manifest.

    ``ScaniaDataModule`` adopts the whole manifest config itself when built with
    ``force_load_from_cache=True``, but the training code also reads ``dataset_params``
    *directly* -- ``sequence_len`` sizes the model heads, ``val_rate`` gates the co-training
    guard, ``seed`` seeds torch -- so the dict handed to ``train_model`` has to agree with the
    cache too, or the models would be built for splits they are not going to see.

    Only keys already present in ``dataset_params`` are overridden. That keeps every
    ``train_model_*`` signature valid: ``calib_rate`` and ``n_quantile_length_strata`` appear in
    the manifest but not in most model configs, and injecting them would raise
    ``TypeError: unexpected keyword argument``. ``ScaniaDataModule`` still applies them
    internally. The keys in :data:`LOADER_ONLY_DATASET_KEYS` are never overridden -- they are
    not part of the manifest config to begin with.

    Args:
        dataset_params: The ``dataset_params`` block read from the model config JSON.
        cache_dir: Cache directory to read ``manifest.json`` from, or ``None`` for the default.
        dataset_root: Root directory of the Scania data files, used to resolve ``cache_dir``.

    Returns:
        A new dict with the cache-defined values applied.

    Raises:
        FileNotFoundError: If the cache directory holds no manifest.
        ValueError: If the manifest has no ``config`` block.
    """
    resolved_cache_dir = resolve_cache_dir(cache_dir, dataset_root)
    manifest_path = os.path.join(resolved_cache_dir, CACHE_MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No {CACHE_MANIFEST_FILE} in {resolved_cache_dir}; cannot force-load the Scania "
            f"dataset from cache. Build the cache first (run without --force-load-from-cache), "
            f"or point --dataset-cache-dir at one that already holds it.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"{manifest_path} has no 'config' block; it was not written by "
            f"ScaniaDataModule._save_cache.")

    merged = dict(dataset_params)
    changes: list[str] = []
    for key, cached_value in config.items():
        if key in _NON_PARAM_MANIFEST_CONFIG_KEYS or key in LOADER_ONLY_DATASET_KEYS:
            continue
        if key not in merged:
            continue
        if merged[key] != cached_value:
            changes.append(f"  {key}: {merged[key]!r} -> {cached_value!r}")
        merged[key] = cached_value

    print(f"[Scania] force-load-from-cache: dataset params pinned to {manifest_path}")
    if changes:
        print("[Scania] overridden by the cache manifest:")
        print("\n".join(changes))
    else:
        print("[Scania] config already matched the cache manifest; nothing overridden.")

    return merged


def generate_and_save_model_prediction(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_version: str,
        prediction_type: str,
        results_path: str,
) -> tuple[float, float]:
    """
    Generate results for predictions and save them in a csv file.

    Args:
        predictions:
        targets:
        model_version:
        prediction_type:
        results_path:

    Returns:

    """
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    df = pd.DataFrame({
        'targets': targets,
        'predictions': predictions,
    })

    csv_path = f'{results_path}/predictions_{model_version}_{prediction_type}_scania.csv'

    df.to_csv(csv_path, index=False)

    print(f"Results for {prediction_type} are saved under : {csv_path}")

    rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
    score = _scania_score(predictions, targets)

    return rmse, score


def _scania_score(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """Total Scania cost of a set of RUL predictions (lower is better).

    Thin alias kept so the existing importers (:mod:`scania.utils.utils_coprog`) and call sites
    stay untouched; the metric itself lives in :mod:`scania.metrics`.

    Args:
        predictions: Predicted RUL in time steps.
        targets: True RUL in time steps.

    Returns:
        The summed Scania cost.
    """
    return scania_score(predictions, targets)