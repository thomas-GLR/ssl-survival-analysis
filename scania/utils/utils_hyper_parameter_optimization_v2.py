"""Grid hyperparameter benchmark for :class:`models.CoTrainingEnsemble_v2` on Scania.

Where :mod:`scania.utils.utils_cotraining_ensemble_v2` trains **one** ensemble from one config,
this module sweeps the ensemble's opt-in levers: it builds the (pruned, deduped) cartesian product
of the ``hyper_parameters`` block and replays the co-training loop once per combination.

Two things are pinned so the configurations are actually comparable:

* the **initial models** are trained once, up front, and every configuration starts from that same
  set (``CoTrainingEnsemble_v2.pretrain_initial_models`` + ``train(pretrained_models=...)``);
* the **censored candidate pool** is drawn from a dedicated, seeded RNG (``train(pool_seed=...)``),
  so iteration ``i`` sees the same units in every configuration while still differing between
  iterations.

The dataset is built once too: a single ``ScaniaDataModule.setup()`` reads the cache, and every
configuration reuses those in-memory tensors.

Everything else (model construction, output saving, the RMSE/score callbacks) is reused from
:mod:`scania.utils.utils_cotraining_common`, :mod:`scania.utils.utils_coprog` and
:mod:`scania.utils.utils_scania`, so a configuration's artifacts are laid out exactly like a
single ``run_train_scania`` run. Every ``*_score`` column of the summary is the class-based
Scania cost (:mod:`scania.metrics`); the ``*_rmse`` columns are unaffected by it.
"""

import copy
import csv
import itertools
import json
import logging
import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any, TextIO

import pandas as pd
import torch
from lightning import LightningModule
from tqdm import tqdm

from constants import necessary_keys_scania
from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2
from scania.dataset import ScaniaDataModule
from scania.utils.utils_coprog import _criteria_callback_for_coprog, _score_callback_for_coprog
from scania.utils.utils_cotraining_common import parse_models_config, save_ensemble_outputs
from scania.utils.utils_scania import (
    assert_data_is_valid,
    override_dataset_params_from_cache_manifest,
    save_train_parameters,
)
from shared.utils import ModelVersion, set_seed
from shared.utils.config import assert_params_contains_all_key, extract_data_from_config

MODEL_VERSION = ModelVersion.CO_TRAINING_ENSEMBLE_V2

#: Name of the benchmark config file inside ``<config_path>/<benchmark_version>/``.
HPO_CONFIG_FILE = "hyper_parameter_optimization_co_training_ensemble_v2.json"

#: Per-configuration artifacts.
SUMMARY_FILE = "hyper_parameter_optimization_summary.csv"
INITIAL_MODELS_DIR = "initial_models"
INITIAL_MODELS_MANIFEST = "manifest.json"
CONFIG_LOG_FILE = "log.txt"
CONFIG_CONSOLE_FILE = "console.txt"
CONFIG_ERROR_FILE = "error.txt"
PER_STAGE_FILE = f"{MODEL_VERSION.value}-per-stage-scania.csv"
SCORES_FILE = f"{MODEL_VERSION.value}-scania.csv"

#: ``fine_tune_patience`` is paired element-wise with ``fine_tune_max_epochs`` instead of being
#: crossed with it (see the spec: it is "the patience for each different epochs").
_PAIRED_FINE_TUNE_KEYS = ("fine_tune_max_epochs", "fine_tune_patience")

#: Swept levers that are arguments of ``CoTrainingEnsemble_v2.train`` rather than of its
#: constructor, so they must not be forwarded to ``__init__``.
_TRAIN_ONLY_HYPER_PARAMETERS = ("suspension_pool_size", "add_ratio")

#: Levers that are read only when their gate is enabled. When the gate is off the ensemble ignores
#: them entirely, so they are collapsed to a single value before deduping — otherwise the sweep
#: would run byte-identical configurations several times over.
#: ``{gate_key: [dependent_key, ...]}``.
_DEPENDENT_HYPER_PARAMETERS: dict[str, list[str]] = {
    "use_fine_tuning": [
        "fine_tune_lr_factor",
        "fine_tune_max_epochs",
        "fine_tune_patience",
        "fine_tune_from_initial_model",
    ],
    "use_monotone_projection": ["monotone_residual_weight"],
    "use_cotraining_ensemble_survival_loss_function": ["cotraining_survival_loss_lambda"],
}


def run_hyper_parameter_optimization(
        config_file_path: str,
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        cache_dir: str | None,
        run_name: str = "",
        start_config: int = 1,
        max_configs: int | None = None,
        resume: str | None = None,
        pretrained_models_dir: str | None = None,
        gpu_id: int | None = None,
        force_load_from_cache: bool = False,
) -> str:
    """Run the full grid benchmark and return the path of the run folder.

    Args:
        config_file_path: Path to the ``hyper_parameter_optimization_co_training_ensemble_v2.json``
            benchmark config.
        checkpoints_path: Root directory for checkpoints; a run folder is created inside it.
        results_path: Root directory for results; a new run creates the run folder
            ``hyper_parameter_opti_co_training_ensemble_v2_YYYY_MM_dd`` inside it.
        dataset_root: Path to the Scania dataset files.
        cache_dir: ``ScaniaDataModule`` cache directory (``None`` → ``<dataset_root>/scania_cache``).
        run_name: Optional extra sub-folder inserted under ``results_path``/``checkpoints_path``.
        start_config: 1-based index of the first configuration to run (earlier ones are skipped).
        max_configs: Maximum number of configurations to run from ``start_config``; ``None`` runs
            all of them.
        resume: Name of an existing run folder to continue, which the run writes into instead of
            creating a dated one — a sweep resumed the day after it started must be pointed at its
            original folder, since a fresh name would restart it from zero. Configurations whose
            scores CSV is already there are skipped. ``None`` starts a new run.
        pretrained_models_dir: Optional directory holding a previously saved set of initial models
            to reuse. ``None`` looks inside the run folder and, failing that, trains them.
        gpu_id: Single GPU id to train on, or ``None`` for auto. The sweep is sequential by design
            (every opt-in lever is sequential-only in ``CoTrainingEnsemble_v2``).
        force_load_from_cache: When ``True``, the data module reads the splits straight out of
            ``cache_dir`` and every split-defining dataset param is taken from its
            ``manifest.json`` instead of the benchmark config; the cache is never rebuilt nor
            overwritten. Pins the sweep to the same vehicles other benchmarked models use.

    Returns:
        The absolute path of the run folder holding every configuration's artifacts and the
        summary CSV.

    Raises:
        FileNotFoundError: If ``resume`` names a run folder that does not exist.
    """
    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
    )

    hyper_parameters, training_params, dataset_params, model_params = _load_config(config_file_path)

    # Applied here, before `seed` and `sequence_len` are read out of dataset_params below: the
    # models must be built for the splits the cache actually holds, not for the config's.
    if force_load_from_cache:
        dataset_params = override_dataset_params_from_cache_manifest(
            dataset_params=dataset_params,
            cache_dir=cache_dir,
            dataset_root=dataset_root,
        )

    if run_name:
        results_path = os.path.join(results_path, run_name)
        checkpoints_path = os.path.join(checkpoints_path, run_name)

    # A new run is stamped with the current date, as the benchmark spec asks. That name moves at
    # midnight, so resuming is explicit rather than derived: the caller names the folder to
    # continue, and a typo fails loudly here instead of silently starting an empty sweep that
    # retrains the shared initial models.
    run_folder_prefix = f"hyper_parameter_opti_{MODEL_VERSION.value}_"
    if resume:
        run_folder = resume
        run_results_path = os.path.join(results_path, run_folder)
        if not os.path.isdir(run_results_path):
            available = sorted(
                name for name in os.listdir(results_path)
                if name.startswith(run_folder_prefix)
                and os.path.isdir(os.path.join(results_path, name))
            ) if os.path.isdir(results_path) else []
            raise FileNotFoundError(
                f"Cannot resume: {run_results_path} does not exist. Run folder(s) available under "
                f"{results_path}: {available if available else 'none'}.")
    else:
        run_folder = f"{run_folder_prefix}{datetime.now().strftime('%Y_%m_%d')}"
        run_results_path = os.path.join(results_path, run_folder)

    run_checkpoints_path = os.path.join(checkpoints_path, run_folder)
    os.makedirs(run_results_path, exist_ok=True)
    os.makedirs(run_checkpoints_path, exist_ok=True)

    run_log_path = os.path.join(run_results_path, CONFIG_LOG_FILE)

    def log(message: str) -> None:
        """Print a run-level milestone and append it to the run log."""
        print(message)
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    log(f"=== Hyperparameter optimization | {MODEL_VERSION.value} | Scania ===")
    log(f"Datetime: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    log(f"Config file: {config_file_path}")
    log(f"Results: {run_results_path}" + (" (resumed)" if resume else ""))

    configurations = build_configurations(hyper_parameters)
    raw_count = _raw_combination_count(hyper_parameters)
    log(f"Hyperparameter grid: {raw_count} raw combination(s) -> {len(configurations)} distinct "
        f"configuration(s) to run (dependent levers pruned and deduplicated).")

    models_cfg = model_params["models"]
    seed = dataset_params.get("seed")
    iterations = training_params["iterations"]
    # train_with_censored_data is not part of the spec's training block; accepted as an optional
    # key because it only affects Initial training, which now happens once outside the grid.
    train_with_censored_data = bool(training_params.get("train_with_censored_data", False))

    # ---------------------------------------------------------------- #
    # Dataset: built once, so the cache is read once and every config
    # reuses the same in-memory tensors.
    # ---------------------------------------------------------------- #
    dataset_kwargs = dict(dataset_params)
    dataset_kwargs.update({
        "data_dir": dataset_root,
        "cache_dir": cache_dir,
        "force_load_from_cache": force_load_from_cache,
    })
    # Only a fallback: under force_load_from_cache the module overrides calib_rate (and every
    # other split-defining param) with the cache manifest's value.
    dataset_kwargs.setdefault("calib_rate", 0.0)

    log("Creating the data module (shared by every configuration)...")
    data_module = ScaniaDataModule(**dataset_kwargs)
    data_module.setup()

    data = _collect_tensors(data_module)

    if dataset_params["val_rate"] <= 0:
        raise ValueError(
            "The co-training ensemble needs a validation set for early stopping / best-model "
            "selection, conformal calibration and the ensemble weights. Set val_rate > 0.")

    nn_modules, module_builders, meta = parse_models_config(
        models_cfg=models_cfg,
        feature_num=len(data_module.feature_cols),
        sequence_len=dataset_params["sequence_len"],
        targets_uncensored=data["train_targets"],
    )
    number_of_models = len(nn_modules)
    batch_size = dataset_params["batch_size"]

    fixed_kwargs = {
        "confidence": training_params["confidence"],
        "inference_batch_size": training_params["inference_batch_size"],
        "bagging_failure_data": bool(training_params["bagging_failure_data"]),
    }

    # ---------------------------------------------------------------- #
    # Initial models: trained (or loaded) once, shared by every config.
    # ---------------------------------------------------------------- #
    initial_models_dir = pretrained_models_dir or os.path.join(run_results_path, INITIAL_MODELS_DIR)
    initial_models, initial_datasets = _get_or_train_initial_models(
        initial_models_dir=initial_models_dir,
        nn_modules=nn_modules,
        module_builders=module_builders,
        meta=meta,
        batch_size=batch_size,
        number_of_models=number_of_models,
        fixed_kwargs=fixed_kwargs,
        data=data,
        seed=seed,
        gpu_id=gpu_id,
        train_with_censored_data=train_with_censored_data,
        log=log,
    )

    # ---------------------------------------------------------------- #
    # The sweep itself.
    # ---------------------------------------------------------------- #
    selected = configurations[start_config - 1:]
    if max_configs is not None:
        selected = selected[:max_configs]
    log(f"Running configuration(s) {start_config}..{start_config + len(selected) - 1} "
        f"of {len(configurations)}.")

    summary_path = os.path.join(run_results_path, SUMMARY_FILE)
    summary_rows: list[dict[str, Any]] = []

    # file=sys.stdout (instead of tqdm's default stderr) so the bar survives the per-configuration
    # stderr redirection that silences Lightning's noise.
    progress = tqdm(selected, desc="configurations", unit="config", file=sys.stdout)
    for offset, configuration in enumerate(progress):
        index = start_config + offset
        config_name = f"configuration_{index}"
        progress.set_postfix_str(config_name)

        config_results_path = os.path.join(run_results_path, config_name)
        config_checkpoints_path = os.path.join(run_checkpoints_path, config_name)
        os.makedirs(config_results_path, exist_ok=True)
        os.makedirs(config_checkpoints_path, exist_ok=True)

        if resume and os.path.exists(os.path.join(config_results_path, SCORES_FILE)):
            row = _summary_row(index, configuration, config_results_path, status="skipped")
            summary_rows.append(row)
            _write_summary(summary_path, summary_rows)
            progress.write(f"[{config_name}] already complete, skipped (resuming {resume}).")
            continue

        status = "ok"
        try:
            _run_one_configuration(
                configuration=configuration,
                fixed_kwargs=fixed_kwargs,
                iterations=iterations,
                dataset_kwargs=dataset_kwargs,
                models_cfg=models_cfg,
                nn_modules=nn_modules,
                module_builders=module_builders,
                meta=meta,
                batch_size=batch_size,
                number_of_models=number_of_models,
                data=data,
                initial_models=initial_models,
                initial_datasets=initial_datasets,
                seed=seed,
                gpu_id=gpu_id,
                results_path=config_results_path,
                checkpoints_path=config_checkpoints_path,
            )
        except Exception as error:  # one bad configuration must not abort the sweep
            status = "failed"
            error_path = os.path.join(config_results_path, CONFIG_ERROR_FILE)
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"=== {config_name} ({MODEL_VERSION.value}) failed ===\n")
                f.write(f"Datetime: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
                f.write(f"Configuration: {json.dumps(configuration, indent=2)}\n")
                f.write(f"Error: {error}\n")
                traceback.print_exc(file=f)
                f.write("=========================================\n")
            progress.write(f"[{config_name}] FAILED: {error} (see {error_path})")

        row = _summary_row(index, configuration, config_results_path, status=status)
        summary_rows.append(row)
        _write_summary(summary_path, summary_rows)

        if row.get("final_test_rmse") is not None:
            progress.set_postfix_str(f"{config_name} rmse={row['final_test_rmse']:.4f}")

    progress.close()

    log(f"Sweep done. Summary written to {summary_path}")
    return run_results_path


# ---------------------------------------------------------------------- #
# Config loading
# ---------------------------------------------------------------------- #


def _load_config(config_file_path: str) -> tuple[dict, dict, dict, dict]:
    """Read and validate the benchmark config's four blocks.

    Args:
        config_file_path: Path to the benchmark JSON.

    Returns:
        ``(hyper_parameters, training_params, dataset_params, model_params)``.

    Raises:
        KeyError: If a block or one of its required keys is missing.
    """
    config = extract_data_from_config(config_file_path)

    for block in ("hyper_parameters", "training_params", "dataset_params", "model_params"):
        if block not in config:
            raise KeyError(f"{block} not found in config {config_file_path}")

    assert_params_contains_all_key(
        config["hyper_parameters"],
        necessary_keys_scania.NECESSARY_HPO_CO_TRAINING_ENSEMBLE_V2_HYPER_PARAMETER_KEYS,
        "hyper_parameters",
    )
    assert_params_contains_all_key(
        config["training_params"],
        necessary_keys_scania.NECESSARY_HPO_CO_TRAINING_ENSEMBLE_V2_TRAINING_KEYS,
        "training_params",
    )
    assert_params_contains_all_key(
        config["dataset_params"],
        necessary_keys_scania.NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_V2_KEYS,
        "dataset_params",
    )
    assert_params_contains_all_key(
        config["model_params"],
        necessary_keys_scania.NECESSARY_CO_TRAINING_ENSEMBLE_V2_KEYS,
        "model_params",
    )

    return (
        config["hyper_parameters"],
        config["training_params"],
        config["dataset_params"],
        config["model_params"],
    )


# ---------------------------------------------------------------------- #
# Grid construction
# ---------------------------------------------------------------------- #


def build_configurations(hyper_parameters: dict[str, list]) -> list[dict[str, Any]]:
    """Expand the ``hyper_parameters`` block into the list of configurations to run.

    Three rules shape the grid:

    1. ``fine_tune_max_epochs`` and ``fine_tune_patience`` are consumed **element-wise** (the
       patience that goes with each epoch budget), never crossed — they must have the same length.
    2. A lever whose gate is off is collapsed to its first listed value (see
       ``_DEPENDENT_HYPER_PARAMETERS``): the ensemble ignores it, so keeping several values would
       only produce duplicate runs.
    3. ``isotonic_time_weighting=True`` requires ``use_monotone_projection=True``
       (``CoTrainingEnsemble_v2.train`` raises otherwise), so those combinations are dropped.

    What survives is then deduplicated, preserving order.

    Args:
        hyper_parameters: The config block; each key maps to a list of candidate values.

    Returns:
        One dict of concrete lever values per configuration, in run order.

    Raises:
        ValueError: If a key does not map to a non-empty list, or if the two paired fine-tuning
            keys have different lengths.
    """
    for key, values in hyper_parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"hyper_parameters['{key}'] must be a non-empty list of candidate values, "
                f"got {values!r}.")

    epochs_key, patience_key = _PAIRED_FINE_TUNE_KEYS
    if len(hyper_parameters[epochs_key]) != len(hyper_parameters[patience_key]):
        raise ValueError(
            f"hyper_parameters['{patience_key}'] gives the patience for each entry of "
            f"'{epochs_key}', so both must have the same length (got "
            f"{len(hyper_parameters[patience_key])} and {len(hyper_parameters[epochs_key])}).")

    # The paired keys travel as one pseudo-key, so the product never crosses them.
    paired_values = list(zip(hyper_parameters[epochs_key], hyper_parameters[patience_key]))
    plain_keys = [k for k in hyper_parameters if k not in _PAIRED_FINE_TUNE_KEYS]
    axes = [hyper_parameters[k] for k in plain_keys] + [paired_values]

    configurations: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for combination in itertools.product(*axes):
        configuration = dict(zip(plain_keys, combination[:-1]))
        configuration[epochs_key], configuration[patience_key] = combination[-1]

        if configuration["isotonic_time_weighting"] and not configuration["use_monotone_projection"]:
            # Would raise in train(): the isotonic weighting only exists inside the projection.
            continue

        _canonicalize(configuration, hyper_parameters)

        key = tuple(sorted((k, _hashable(v)) for k, v in configuration.items()))
        if key in seen:
            continue
        seen.add(key)
        configurations.append(configuration)

    return configurations


def _canonicalize(configuration: dict[str, Any], hyper_parameters: dict[str, list]) -> None:
    """Force every disabled lever's dependents to their first candidate value, in place.

    ``fine_tune_from_initial_model`` gets a stricter treatment: ``train()`` raises when it is
    ``True`` without ``use_fine_tuning``, so it is forced to ``False`` rather than to the first
    listed value.

    Args:
        configuration: One concrete combination; mutated in place.
        hyper_parameters: The config block, used for the first (fallback) value of each key.
    """
    for gate, dependents in _DEPENDENT_HYPER_PARAMETERS.items():
        if configuration[gate]:
            continue
        for dependent in dependents:
            configuration[dependent] = hyper_parameters[dependent][0]

    if not configuration["use_fine_tuning"]:
        configuration["fine_tune_from_initial_model"] = False


def _raw_combination_count(hyper_parameters: dict[str, list]) -> int:
    """Number of combinations before pruning/deduping, for the "N raw -> M distinct" log line.

    Args:
        hyper_parameters: The config block.

    Returns:
        The size of the cartesian product, counting the two paired fine-tuning keys as one axis.
    """
    count = len(hyper_parameters[_PAIRED_FINE_TUNE_KEYS[0]])
    for key, values in hyper_parameters.items():
        if key not in _PAIRED_FINE_TUNE_KEYS:
            count *= len(values)
    return count


def _hashable(value: Any) -> Any:
    """Make a config value usable inside the dedup key (lists become tuples).

    Args:
        value: A candidate lever value.

    Returns:
        The value itself, or a tuple when it is a list.
    """
    return tuple(value) if isinstance(value, list) else value


# ---------------------------------------------------------------------- #
# Shared data + initial models
# ---------------------------------------------------------------------- #


def _collect_tensors(data_module: ScaniaDataModule) -> dict[str, torch.Tensor | None]:
    """Pull every tensor the sweep needs out of the (already set up) data module.

    ``suspension_time_steps`` is fetched unconditionally because some configurations enable
    ``isotonic_time_weighting``; the calibration split is fetched only when one was configured.

    Args:
        data_module: An already ``setup()``-ed ``ScaniaDataModule``.

    Returns:
        A dict of tensors keyed by role (``train_features``, ``suspension_ids``, ``val_features``,
        ``calib_features`` — ``None`` when there is no calibration split — …).
    """
    train_features, train_targets, suspension_data, suspension_ids = \
        data_module.get_cotraining_tensors("train")
    _, _, suspension_lower_bounds = data_module.get_censored_lower_bounds("train")
    _, _, suspension_time_steps = data_module.get_censored_time_steps("train")

    val_features, val_targets, _, _ = data_module.get_cotraining_tensors("val")
    test_features, test_targets, _, _ = data_module.get_cotraining_tensors("test")

    calib_features, calib_targets = None, None
    if data_module.calib_rate > 0:
        calib_features, calib_targets, _, _ = data_module.get_cotraining_tensors("calib")

    return {
        "train_features": train_features,
        "train_targets": train_targets,
        "suspension_data": suspension_data,
        "suspension_ids": suspension_ids,
        "suspension_lower_bounds": suspension_lower_bounds,
        "suspension_time_steps": suspension_time_steps,
        "val_features": val_features,
        "val_targets": val_targets,
        "test_features": test_features,
        "test_targets": test_targets,
        "calib_features": calib_features,
        "calib_targets": calib_targets,
    }


def _build_ensemble(
        configuration: dict[str, Any] | None,
        fixed_kwargs: dict[str, Any],
        nn_modules: list,
        module_builders: list,
        meta: dict,
        batch_size: int,
        seed: int | None,
        gpu_id: int | None,
        verbose: int,
) -> CoTrainingEnsemble_v2:
    """Instantiate and configure one ensemble.

    ``suspension_pool_size`` / ``add_ratio`` are swept too but are ``train`` arguments, so they are
    filtered out of the constructor kwargs here and passed at ``train`` time instead.

    The global RNG is re-seeded right before ``setup_training_builder`` because that call snapshots
    each model's fresh initial weights — seeding makes every configuration's from-scratch retrains
    start from the identical initialization.

    Args:
        configuration: The configuration's lever values, or ``None`` for the pre-training ensemble
            (which only needs the fixed params).
        fixed_kwargs: Levers held constant across the sweep (``confidence``,
            ``inference_batch_size``, ``bagging_failure_data``).
        nn_modules: The per-model ``nn.Module`` list (the ensemble uses it for its model count;
            the builder path rebuilds its own modules for training).
        module_builders: Picklable per-model module builders from ``parse_models_config``.
        meta: The ``parse_models_config`` metadata (``max_epochs``, ``patiences``, …).
        batch_size: Training batch size, shared by every model.
        seed: Seed applied before the initial-weight snapshot; ``None`` leaves the RNG alone.
        gpu_id: Single GPU id, or ``None`` for auto. Never a multi-GPU list — the sweep is
            sequential.
        verbose: ``CoTrainingEnsemble_v2`` verbosity (``0`` keeps the console clean during the
            sweep; messages still reach the configuration's log file).

    Returns:
        The configured ensemble, ready for ``train``.
    """
    constructor_kwargs = {
        key: value
        for key, value in (configuration or {}).items()
        if key not in _TRAIN_ONLY_HYPER_PARAMETERS
    }

    number_of_models = len(nn_modules)
    ensemble = CoTrainingEnsemble_v2(
        models=nn_modules,
        verbose=verbose,
        **fixed_kwargs,
        **constructor_kwargs,
    )

    set_seed(seed)

    ensemble.setup_training_builder(
        module_builders=module_builders,
        max_epochs=meta["max_epochs"],
        patiences=meta["patiences"],
        batchs_size=[batch_size] * number_of_models,
        shuffle_dataloaders=[True] * number_of_models,
        gpu_ids=[gpu_id] if gpu_id is not None else None,
    )
    return ensemble


def _get_or_train_initial_models(
        initial_models_dir: str,
        nn_modules: list,
        module_builders: list,
        meta: dict,
        batch_size: int,
        number_of_models: int,
        fixed_kwargs: dict[str, Any],
        data: dict[str, torch.Tensor | None],
        seed: int | None,
        gpu_id: int | None,
        train_with_censored_data: bool,
        log,
) -> tuple[list[LightningModule], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Load the shared initial models from disk, or train them once and save them.

    Every configuration of the sweep starts from these exact models, so a difference in the
    results is attributable to the hyperparameters rather than to a different initialization.

    Args:
        initial_models_dir: Directory the models are loaded from / saved to.
        nn_modules: The freshly built ``nn.Module`` list (only its length matters here).
        module_builders: Picklable per-model module builders.
        meta: ``parse_models_config`` metadata.
        batch_size: Training batch size.
        number_of_models: Number of models in the ensemble.
        fixed_kwargs: Levers held constant across the sweep.
        data: The tensors from :func:`_collect_tensors`.
        seed: Seed applied before the initial-weight snapshot.
        gpu_id: Single GPU id, or ``None``.
        train_with_censored_data: Whether Initial training also uses the censored data under
            ``survival_loss_function``.
        log: Run-level logging callable.

    Returns:
        ``(initial_models, initial_datasets)`` — the trained module per model and each model's own
        ``(x, y)`` training split.
    """
    manifest_path = os.path.join(initial_models_dir, INITIAL_MODELS_MANIFEST)
    if os.path.exists(manifest_path):
        log(f"Loading the shared initial models from {initial_models_dir}...")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["number_of_models"] != number_of_models:
            raise ValueError(
                f"{manifest_path} holds {manifest['number_of_models']} model(s) but the config "
                f"declares {number_of_models}.")
        initial_models = [
            torch.load(os.path.join(initial_models_dir, name), weights_only=False)
            for name in manifest["model_files"]
        ]
        initial_datasets = torch.load(
            os.path.join(initial_models_dir, manifest["datasets_file"]), weights_only=False)
        return initial_models, initial_datasets

    log(f"Training the shared initial models ({number_of_models} model(s), reused by every "
        f"configuration)...")
    pretrain_ensemble = _build_ensemble(
        configuration=None,
        fixed_kwargs=fixed_kwargs,
        nn_modules=nn_modules,
        module_builders=module_builders,
        meta=meta,
        batch_size=batch_size,
        seed=seed,
        gpu_id=gpu_id,
        verbose=1,
    )
    initial_models, initial_datasets = pretrain_ensemble.pretrain_initial_models(
        failure_data=data["train_features"],
        failure_label=data["train_targets"],
        val_data=data["val_features"],
        val_label=data["val_targets"],
        train_with_censored_data=train_with_censored_data,
        suspension_data=data["suspension_data"],
        suspension_lower_bounds=data["suspension_lower_bounds"],
    )

    os.makedirs(initial_models_dir, exist_ok=True)
    model_files = []
    for i, (module, version_str) in enumerate(zip(initial_models, meta["version_strs"])):
        name = f"initial_model_{i}_{version_str}.pth"
        torch.save(module, os.path.join(initial_models_dir, name))
        model_files.append(name)
    datasets_file = "initial_models_datasets.pt"
    torch.save(initial_datasets, os.path.join(initial_models_dir, datasets_file))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "number_of_models": number_of_models,
                "version_strs": meta["version_strs"],
                "model_files": model_files,
                "datasets_file": datasets_file,
                "train_with_censored_data": train_with_censored_data,
                "bagging_failure_data": fixed_kwargs["bagging_failure_data"],
                "seed": seed,
            },
            f,
            indent=2,
        )
    log(f"Shared initial models saved to {initial_models_dir}")

    return initial_models, initial_datasets


# ---------------------------------------------------------------------- #
# One configuration
# ---------------------------------------------------------------------- #


@contextmanager
def _capture_logging_handlers(stream: TextIO) -> Iterator[None]:
    """Temporarily point every console logging handler at ``stream``.

    ``redirect_stderr`` only rebinds ``sys.stderr``; a ``StreamHandler`` built at import time keeps
    its own reference to the original stream. Lightning installs exactly such a handler on its own
    logger, so without this its "GPU available: ..." / "`Trainer.fit` stopped" messages would still
    reach the terminal and break the progress bar apart.

    Args:
        stream: File object the patched handlers write to for the duration of the block.

    Yields:
        ``None``. The original handler streams are restored on exit, including on error.
    """
    real_streams = {id(sys.__stdout__), id(sys.__stderr__)}
    loggers = [logging.getLogger()] + [
        obj for obj in list(logging.root.manager.loggerDict.values())
        if isinstance(obj, logging.Logger)
    ]

    patched: list[tuple[logging.StreamHandler, TextIO]] = []
    for logger_obj in loggers:
        for handler in logger_obj.handlers:
            # FileHandler is a StreamHandler subclass, but it already writes to its own file.
            if (isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)
                    and id(handler.stream) in real_streams):
                patched.append((handler, handler.stream))
                handler.setStream(stream)

    try:
        yield
    finally:
        for handler, original_stream in patched:
            handler.setStream(original_stream)


def _run_one_configuration(
        configuration: dict[str, Any],
        fixed_kwargs: dict[str, Any],
        iterations: int,
        dataset_kwargs: dict[str, Any],
        models_cfg: list[dict],
        nn_modules: list,
        module_builders: list,
        meta: dict,
        batch_size: int,
        number_of_models: int,
        data: dict[str, torch.Tensor | None],
        initial_models: list[LightningModule],
        initial_datasets: list[tuple[torch.Tensor, torch.Tensor]],
        seed: int | None,
        gpu_id: int | None,
        results_path: str,
        checkpoints_path: str,
) -> None:
    """Train and evaluate one configuration, writing all of its artifacts.

    The configuration's ``run_parameters.json``, ``log.txt``, per-stage metrics CSV, prediction
    CSVs and scores CSV land in ``results_path``, laid out exactly like a single
    ``run_train_scania`` run. Everything the reused helpers print goes to ``console.txt`` so the
    sweep's progress bar stays readable; the ensemble's own messages go to ``log.txt`` (the
    ensemble runs with ``verbose=0``, and ``_log`` writes to its log file regardless of verbosity).

    Args:
        configuration: The configuration's lever values.
        fixed_kwargs: Levers held constant across the sweep.
        iterations: Number of co-training iterations.
        dataset_kwargs: The dataset params, recorded into ``run_parameters.json``.
        models_cfg: The config ``models`` list, recorded into ``run_parameters.json``.
        nn_modules: The per-model ``nn.Module`` list (used for the ensemble's model count).
        module_builders: Picklable per-model module builders.
        meta: ``parse_models_config`` metadata.
        batch_size: Training batch size.
        number_of_models: Number of models in the ensemble.
        data: The tensors from :func:`_collect_tensors`.
        initial_models: The shared initial models (deep-copied before use).
        initial_datasets: The shared per-model datasets (deep-copied before use).
        seed: Seed for the initial-weight snapshot and for the censored-pool RNG.
        gpu_id: Single GPU id, or ``None``.
        results_path: This configuration's result folder.
        checkpoints_path: This configuration's checkpoint folder.
    """
    ensemble = _build_ensemble(
        configuration=configuration,
        fixed_kwargs=fixed_kwargs,
        nn_modules=nn_modules,
        module_builders=module_builders,
        meta=meta,
        batch_size=batch_size,
        seed=seed,
        gpu_id=gpu_id,
        verbose=0,
    )

    training_parameters = {"iterations": iterations, **fixed_kwargs, **configuration}

    log_file_path = os.path.join(results_path, CONFIG_LOG_FILE)
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"=== Co-Training Ensemble (v2) | {os.path.basename(results_path)} ===\n")
        f.write(f"Datetime: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
        f.write(f"Models ({number_of_models}): {meta['version_strs']}\n")
        f.write(f"Configuration: {json.dumps(training_parameters, indent=2)}\n")
        f.write("====================================================\n")

    # Everything the configuration emits is captured: the reused helpers print to stdout, warnings
    # go to stderr and Lightning logs through its own handler (hence _capture_logging_handlers).
    # tqdm holds its own reference to the real stdout (see the ``file=sys.stdout`` at construction),
    # so the progress bar is unaffected.
    console_path = os.path.join(results_path, CONFIG_CONSOLE_FILE)
    with open(console_path, "w", encoding="utf-8") as console, \
            redirect_stdout(console), redirect_stderr(console), \
            _capture_logging_handlers(console):
        save_train_parameters(
            results_path=results_path,
            dataset_parameters=dataset_kwargs,
            training_parameters=training_parameters,
            model_parameters={"models": models_cfg},
        )

        # Deep copies: a configuration must not mutate the starting point of the next one.
        ensemble.train(
            train_with_censored_data=False,
            failure_data=data["train_features"],
            failure_label=data["train_targets"],
            suspension_data=data["suspension_data"],
            suspension_ids=data["suspension_ids"],
            suspension_lower_bounds=data["suspension_lower_bounds"],
            suspension_time_steps=data["suspension_time_steps"],
            iterations=iterations,
            suspension_pool_size=configuration["suspension_pool_size"],
            add_ratio=configuration["add_ratio"],
            val_data=data["val_features"],
            val_label=data["val_targets"],
            calib_data=data["calib_features"],
            calib_label=data["calib_targets"],
            test_data=data["test_features"],
            test_label=data["test_targets"],
            score_callback=_score_callback_for_coprog,
            weight_callback=_criteria_callback_for_coprog,
            weight_mode="min",
            metrics_file=os.path.join(results_path, PER_STAGE_FILE),
            log_file=log_file_path,
            pretrained_models=copy.deepcopy(initial_models),
            pretrained_models_datasets=copy.deepcopy(initial_datasets),
            pool_seed=seed,
        )

        ensemble.calculate_weights(
            x_test=data["val_features"],
            target=data["val_targets"],
            criteria_callback=_criteria_callback_for_coprog,
            mode="min",
            calib_data=data["calib_features"],
            calib_label=data["calib_targets"],
        )

        save_ensemble_outputs(
            ensemble=ensemble,
            model_version=MODEL_VERSION,
            checkpoints_path=checkpoints_path,
            results_path=results_path,
            test_features=data["test_features"],
            test_targets=data["test_targets"],
            version_strs=meta["version_strs"],
            # Initial training is not part of this: every configuration reuses the shared
            # pretrained models, so the duration covers the co-training iterations only.
            training_time_seconds=ensemble.training_duration_seconds,
            avg_iteration_time_seconds=ensemble.average_iteration_duration_seconds,
        )


# ---------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------- #


def _summary_row(
        index: int,
        configuration: dict[str, Any],
        config_results_path: str,
        status: str,
) -> dict[str, Any]:
    """Build one summary row from a configuration's per-stage metrics file.

    The per-stage CSV written by ``CoTrainingEnsemble_v2.train`` has one row per stage
    (``initial``, ``iteration_k``, ``final``), so "first" is the ``initial`` row, "final" the
    ``final`` row, and min/max are taken across every stage. Both the weighted-ensemble metrics and
    the per-model averages are reported, on the validation set (``*_val_*``) and on the test set
    (``*_test_*``).

    Args:
        index: 1-based configuration number.
        configuration: The configuration's lever values (each becomes its own column).
        config_results_path: The configuration's result folder.
        status: ``"ok"``, ``"failed"`` or ``"skipped"``.

    Returns:
        The summary row; the metric fields are ``None`` when the per-stage file is missing or empty
        (e.g. a configuration that failed before the first stage).
    """
    row: dict[str, Any] = {"configuration": index, "status": status}
    row.update(configuration)

    metric_fields = [
        "first_val_rmse", "first_val_score", "final_val_rmse", "final_val_score",
        "min_val_rmse", "max_val_rmse",
        "first_avg_val_rmse", "first_avg_val_score", "final_avg_val_rmse",
        "final_avg_val_score", "min_avg_val_rmse", "max_avg_val_rmse",
        "first_test_rmse", "first_test_score", "final_test_rmse", "final_test_score",
        "min_test_rmse", "max_test_rmse",
        "first_avg_test_rmse", "first_avg_test_score", "final_avg_test_rmse",
        "final_avg_test_score", "min_avg_test_rmse", "max_avg_test_rmse",
    ]
    row.update({field: None for field in metric_fields})

    per_stage_path = os.path.join(config_results_path, PER_STAGE_FILE)
    if not os.path.exists(per_stage_path):
        return row

    stages = pd.read_csv(per_stage_path)
    if stages.empty:
        return row

    first = stages[stages["stage"] == "initial"]
    first = first.iloc[0] if not first.empty else stages.iloc[0]
    final = stages[stages["stage"] == "final"]
    final = final.iloc[0] if not final.empty else stages.iloc[-1]

    # The validation columns were added to the per-stage file after the test ones, so a file
    # written by an earlier run only has the test columns: fill in a split only when present,
    # leaving the other fields at None.
    for split in ("val", "test"):
        weighted_rmse = f"weighted_{split}_rmse"
        weighted_score = f"weighted_{split}_score"
        avg_rmse = f"avg_{split}_rmse"
        avg_score = f"avg_{split}_score"
        if weighted_rmse not in stages.columns:
            continue

        row.update({
            f"first_{split}_rmse": float(first[weighted_rmse]),
            f"first_{split}_score": float(first[weighted_score]),
            f"final_{split}_rmse": float(final[weighted_rmse]),
            f"final_{split}_score": float(final[weighted_score]),
            f"min_{split}_rmse": float(stages[weighted_rmse].min()),
            f"max_{split}_rmse": float(stages[weighted_rmse].max()),
            f"first_avg_{split}_rmse": float(first[avg_rmse]),
            f"first_avg_{split}_score": float(first[avg_score]),
            f"final_avg_{split}_rmse": float(final[avg_rmse]),
            f"final_avg_{split}_score": float(final[avg_score]),
            f"min_avg_{split}_rmse": float(stages[avg_rmse].min()),
            f"max_avg_{split}_rmse": float(stages[avg_rmse].max()),
        })
    return row


def _write_summary(summary_path: str, rows: list[dict[str, Any]]) -> None:
    """Rewrite the summary CSV after every configuration, so a crash never loses what ran.

    Args:
        summary_path: Destination CSV.
        rows: Every summary row produced so far.
    """
    if not rows:
        return
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
