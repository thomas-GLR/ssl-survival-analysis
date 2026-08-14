"""
optuna_search_dynamic_deephit.py
─────────────────────────────────
Optuna hyperparameter search for Dynamic DeepHit (TensorFlow) on the Scania Component X dataset,
optimizing the validation C-index.

Design:
  - Config-driven search space: every hyperparameter's Optuna range comes from a JSON file
    (``{"<name>": {"type": "float"|"int"|"categorical", "low":, "high":, "log": (optional),
    "choices": (categorical only), "step": (optional)}}``), not hard-coded in Python -- unlike every
    other HPO path in this repo (``C_MAPSS/hpo/optuna_search.py``, ``scania/hpo/optuna_search.py``).
  - Single-objective: maximize the validation C-index
    (``scania.utils.utils_dynamic_deephit._validation_c_index``, the same metric ``train_model`` uses
    internally for checkpoint selection but never reports), via TPE + a step-based median pruner fed
    at every ``eval_every`` checkpoint.
  - None of the 18 searchable hyperparameters (``mb_size, iteration_burn_in, iteration, keep_prob,
    lr_train, alpha, beta, gamma, reg_W, reg_W_out, FC_active_fn, RNN_active_fn, h_dim_RNN, h_dim_FC,
    num_layers_RNN, num_layers_ATT, num_layers_CS, RNN_type``) affect dataset construction, so the
    ``ScaniaDataModule``, per-split arrays, time-bin edges and training masks (a
    ``utils_dynamic_deephit.RunContext``) are built ONCE for the whole study and reused, unchanged,
    by every trial -- only a fresh ``Model_Longitudinal_Attention`` is built per trial.
  - GPU visibility (``_configure_gpu``) is configured once, before ``study.optimize``, never per
    trial: TensorFlow raises once a GPU has already been initialized in the process.
  - Every trial skips disk checkpointing entirely (``_fit(..., checkpoint=None)``): only the scalar
    best C-index is needed, so there is nothing to persist per trial.

Usage:
    study = run_search(
        base_config_path="scania/config/default/dynamic_deephit.json",
        search_space_path="scania/config/default/dynamic_deephit_search_space.json",
        dataset_root="./data/Scania_component_X",
        n_trials=50,
    )
    save_study_results(study, "./outputs/hpo_dynamic_deephit")
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Any, Callable

import optuna
import pandas as pd

from constants import necessary_keys_scania
from scania.utils.utils_dynamic_deephit import (
    RunContext,
    _build_network_settings,
    _build_run_context,
    _configure_gpu,
    _fit,
    import_dynamic_deephit,
)
from shared.utils.config import extract_data_from_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Seed for the Optuna TPE sampler, so the sequence of suggested trials is reproducible across HPO
# runs. Distinct from the base config's own ``seed`` (which governs the vehicle split, TF's RNG and
# the C-index validation subsample, exactly as it does for a plain `train_model` run).
HPO_SEED = 42

# The 18 hyperparameters this search can tune, split by which part of train_model()'s signature they
# belong to (see constants/necessary_keys_scania.py). Every other key in these two blocks -- plus all
# of NECESSARY_DATASET_DYNAMIC_DEEPHIT_KEYS -- is fixed, taken from the base config, because none of
# them are dataset params: the RunContext built once from the base config's dataset_params stays
# valid for every trial regardless of what gets searched.
_SEARCHABLE_MODEL_KEYS = frozenset(necessary_keys_scania.NECESSARY_DYNAMIC_DEEPHIT_KEYS)
_SEARCHABLE_TRAINING_KEYS = frozenset(necessary_keys_scania.NECESSARY_TRAINING_DYNAMIC_DEEPHIT_KEYS) - {
    "burn_in_mode", "eval_every", "num_category_bins", "use_gpu", "c_index_time_quantiles",
    "c_index_max_vehicles", "inference_batch_size",
}
_SEARCHABLE_KEYS = _SEARCHABLE_MODEL_KEYS | _SEARCHABLE_TRAINING_KEYS

# The training_params subset _fit() actually accepts (num_category_bins/use_gpu are consumed
# elsewhere -- context construction and one-shot GPU config, respectively -- and are not _fit kwargs).
_FIT_TRAINING_KEYS = (
    "mb_size", "burn_in_mode", "iteration_burn_in", "iteration", "keep_prob", "lr_train",
    "alpha", "beta", "gamma", "eval_every", "c_index_time_quantiles", "c_index_max_vehicles",
    "inference_batch_size",
)


def _load_search_space(search_space_path: str) -> dict[str, dict[str, Any]]:
    """Load and validate the Optuna search-space config.

    Args:
        search_space_path: Path to a JSON file shaped as ``{"<param_name>": {"type":
            "float"|"int"|"categorical", "low":, "high":, "log": (optional, float/int only),
            "choices": (categorical only), "step": (optional)}}``.

    Returns:
        The parsed search space.

    Raises:
        ValueError: If a key is not one of the 18 hyperparameters this module can search, or a spec
            is missing the fields its ``type`` requires.
    """
    search_space = extract_data_from_config(search_space_path)

    unknown_keys = set(search_space) - _SEARCHABLE_KEYS
    if unknown_keys:
        raise ValueError(
            f"{search_space_path} searches unknown hyperparameter(s) {sorted(unknown_keys)}; "
            f"Dynamic DeepHit HPO only supports {sorted(_SEARCHABLE_KEYS)}."
        )

    for name, spec in search_space.items():
        param_type = spec.get("type")
        if param_type in ("float", "int"):
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"{search_space_path}: '{name}' ({param_type}) needs 'low'/'high'.")
        elif param_type == "categorical":
            if "choices" not in spec:
                raise ValueError(f"{search_space_path}: '{name}' (categorical) needs 'choices'.")
        else:
            raise ValueError(
                f"{search_space_path}: '{name}' has unsupported type {param_type!r}; "
                f"expected 'float', 'int' or 'categorical'."
            )

    return search_space


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Dispatch one search-space entry to the matching ``trial.suggest_*`` call.

    Args:
        trial: The current Optuna trial.
        name: The hyperparameter name (becomes the Optuna param name).
        spec: One entry of the search space (see :func:`_load_search_space`).

    Returns:
        The suggested value.
    """
    param_type = spec["type"]
    if param_type == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False), step=spec.get("step"))
    if param_type == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
    return trial.suggest_categorical(name, spec["choices"])


def _load_base_params(base_config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load Dynamic DeepHit's ``dataset_params``/``training_params``/``model_params`` from a config
    JSON, validated the same way ``run_train_scania.py`` validates them.

    Args:
        base_config_path: Path to a ``dynamic_deephit.json``-shaped config. Every value not present
            in the search space stays fixed at what this config says; the searched subset of
            ``training_params``/``model_params`` is overridden per trial.

    Returns:
        ``(dataset_params, training_params, model_params)``.
    """
    from scania.utils.utils_scania import (
        extract_dataset_params_from_config,
        extract_model_params_from_config,
        extract_training_params_from_config,
    )

    dataset_params = extract_dataset_params_from_config(
        base_config_path, necessary_keys_scania.NECESSARY_DATASET_DYNAMIC_DEEPHIT_KEYS)
    training_params = extract_training_params_from_config(
        base_config_path, necessary_keys_scania.NECESSARY_TRAINING_DYNAMIC_DEEPHIT_KEYS)
    model_params = extract_model_params_from_config(
        base_config_path, necessary_keys_scania.NECESSARY_DYNAMIC_DEEPHIT_KEYS)
    return dataset_params, training_params, model_params


def _make_objective(
        model_class: type,
        context: RunContext,
        base_training_params: dict[str, Any],
        base_model_params: dict[str, Any],
        search_space: dict[str, dict[str, Any]],
        seed: int | None,
        utils_helper_module: Any,
        utils_eval_module: Any,
        tf_module: Any,
) -> Callable[[optuna.Trial], float]:
    """Build the single-objective Optuna closure: maximize the validation C-index.

    Args:
        model_class: ``Model_Longitudinal_Attention`` (from :func:`import_dynamic_deephit`).
        context: The shared :class:`RunContext`, built once for the whole study.
        base_training_params: Fixed ``training_params`` from the base config; the keys present in
            ``search_space`` are overridden per trial before being handed to ``_fit``.
        base_model_params: Fixed ``model_params`` from the base config; same override rule.
        search_space: The loaded search space (see :func:`_load_search_space`).
        seed: Seed for the C-index validation subsample (the base config's ``dataset_params.seed``).
        utils_helper_module: The vendored ``dynamic_deephit.utils_helper`` module.
        utils_eval_module: The vendored ``dynamic_deephit.utils_eval`` module.
        tf_module: The imported ``tensorflow`` module.

    Returns:
        The objective callable returning the trial's best validation C-index.
    """

    def objective(trial: optuna.Trial) -> float:
        training_params = dict(base_training_params)
        model_params = dict(base_model_params)
        for name, spec in search_space.items():
            value = _suggest(trial, name, spec)
            if name in model_params:
                model_params[name] = value
            else:
                training_params[name] = value

        network_settings = _build_network_settings(tf_module, **model_params)
        model = model_class(
            f"dynamic_deephit_scania_trial_{trial.number}", context.input_dims, network_settings)

        def on_eval(step: int, c_index: float) -> None:
            trial.report(c_index, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fit_kwargs = {key: training_params[key] for key in _FIT_TRAINING_KEYS}
        try:
            best_c_index, _ = _fit(
                model=model,
                checkpoint=None,
                checkpoint_prefix=None,
                context=context,
                seed=seed,
                utils_helper_module=utils_helper_module,
                utils_eval_module=utils_eval_module,
                on_eval=on_eval,
                **fit_kwargs,
            )
            return best_c_index
        finally:
            # TF2 never releases GPU/variable memory back between trials on its own (no
            # tf.compat.v1.reset_default_graph() applies here, this is eager-mode Keras) -- without
            # this, a many-trial study accumulates every trial's variables and optimizer slots for
            # the life of the process. Runs on every exit path (return, TrialPruned, any exception).
            del model
            tf_module.keras.backend.clear_session()
            gc.collect()

    return objective


def run_search(
        base_config_path: str,
        search_space_path: str,
        dataset_root: str,
        n_trials: int = 50,
        cache_dir: str | None = None,
        force_load_from_cache: bool = False,
        study_name: str | None = None,
        storage: str | None = None,
) -> optuna.Study:
    """Run single-objective Optuna HPO for Dynamic DeepHit on Scania, maximizing validation C-index.

    Builds the (searched-hyperparameter-independent) ``RunContext`` once and configures GPU
    visibility once, before any trial runs -- every trial only builds a fresh model and re-runs the
    training loop against that shared context.

    Args:
        base_config_path: Path to a ``dynamic_deephit.json``-shaped config; every hyperparameter not
            present in ``search_space_path`` (all of ``dataset_params``, plus the non-searched
            subset of ``training_params``/``model_params``) comes from here.
        search_space_path: Path to the Optuna search-space JSON (see :func:`_load_search_space`).
        dataset_root: Root directory of the Scania data files.
        n_trials: Number of Optuna trials.
        cache_dir: Dataset cache directory, or ``None`` for ``<dataset_root>/scania_cache``.
        force_load_from_cache: Pin the run to an existing dataset cache.
        study_name: Custom study name (auto-generated if ``None``).
        storage: Optuna storage URL (e.g. ``sqlite:///optuna.db``) for resuming.

    Returns:
        The completed ``optuna.Study``.
    """
    search_space = _load_search_space(search_space_path)
    dataset_params, training_params, model_params = _load_base_params(base_config_path)

    if force_load_from_cache:
        from scania.utils.utils_scania import override_dataset_params_from_cache_manifest
        assert cache_dir is not None and os.path.exists(cache_dir), f"{cache_dir} does not exist."
        dataset_params = override_dataset_params_from_cache_manifest(
            dataset_params=dataset_params, cache_dir=cache_dir, dataset_root=dataset_root)

    model_class, import_data, utils_helper, utils_eval = import_dynamic_deephit()

    import tensorflow as tf
    tf.random.set_seed(dataset_params["seed"])
    _configure_gpu(tf, training_params["use_gpu"])

    dataset_kwargs = {
        "dataset_root": dataset_root,
        "cache_dir": cache_dir,
        "force_load_from_cache": force_load_from_cache,
        **dataset_params,
    }
    context = _build_run_context(dataset_kwargs, training_params["num_category_bins"], import_data)

    objective = _make_objective(
        model_class=model_class,
        context=context,
        base_training_params=training_params,
        base_model_params=model_params,
        search_space=search_space,
        seed=dataset_params["seed"],
        utils_helper_module=utils_helper,
        utils_eval_module=utils_eval,
        tf_module=tf,
    )

    study_name = study_name or "dynamic_deephit_scania"
    sampler = optuna.samplers.TPESampler(seed=HPO_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,  # resume an existing study with the same name + storage
    )

    logger.info(
        "Starting search: model=dynamic_deephit  trials=%d  objective=val C-index (maximize)",
        n_trials,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    _log_results(study, study_name)
    return study


def best_params(study: optuna.Study) -> dict[str, Any]:
    """Return the best hyperparameters from a completed study.

    Args:
        study: A completed Optuna study.

    Returns:
        Dict of hyperparameters for the best trial.
    """
    t = study.best_trial
    logger.info("Best trial #%d | val_c_index=%.4f | params=%s", t.number, t.value, t.params)
    return t.params


def summary_table(study: optuna.Study) -> None:
    """Log a one-line summary of the best trial of a study.

    Args:
        study: A completed Optuna study.
    """
    header = f"{'val_C_index':>12} {'Trial':>7}"
    logger.info("\n" + "=" * len(header))
    logger.info("SUMMARY  (objective = validation C-index, maximize)")
    logger.info(header)
    logger.info("-" * len(header))
    t = study.best_trial
    logger.info(f"{t.value:>12.4f} {t.number:>7d}")
    logger.info("=" * len(header))


def save_study_results(study: optuna.Study, output_dir: str) -> None:
    """Save all completed trials and the best params of a study to ``output_dir``.

    Files written:
      ``<output_dir>/dynamic_deephit_trials.csv``       -- all completed trials
      ``<output_dir>/dynamic_deephit_best_params.json`` -- best trial params

    Args:
        study: A completed Optuna study.
        output_dir: Directory to write the result files to.
    """
    os.makedirs(output_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials for dynamic_deephit — skipping save.")
        return

    rows = []
    for t in completed:
        row = {"trial": t.number}
        row.update(t.params)
        row["val_c_index"] = t.value
        rows.append(row)

    trials_path = os.path.join(output_dir, "dynamic_deephit_trials.csv")
    pd.DataFrame(rows).to_csv(trials_path, index=False)
    logger.info("Trials saved → %s", trials_path)

    best_path = os.path.join(output_dir, "dynamic_deephit_best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params(study), f, indent=2)
    logger.info("Best params saved → %s", best_path)


def _log_results(study: optuna.Study, name: str) -> None:
    """Log the best trial of a finished study.

    Args:
        study: The finished Optuna study.
        name: The study name (for the log line).
    """
    logger.info("=" * 60)
    logger.info("Study: %s | finished trials: %d", name, len(study.trials))
    bt = study.best_trial
    logger.info("Best trial #%d  val_c_index=%.4f  params=%s", bt.number, bt.value, bt.params)
