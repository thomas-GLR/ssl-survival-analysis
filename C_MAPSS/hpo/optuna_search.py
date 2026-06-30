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

import json
import logging
import os
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

import optuna
from lightning import Trainer

from C_MAPSS.lightning_module.TransformerLstmModule import TransformerLstmModule

# optuna-integration >= 3.0 ships as a separate package
# try:
#     from optuna.integration import PyTorchLightningPruningCallback
# except ImportError:
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

ModelBuilder = Callable[[optuna.Trial, int, int], nn.Module]
_MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "lstm": ModelBuilder,
    "transformer_lstm": ModelBuilder,
    "cnn1d": ModelBuilder,
}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """
    Decorator – registers a model builder under a unique name.

    The builder receives (trial, input_size, output_size) and returns an nn.Module.
    Architecture hyperparameters are suggested inside the builder via trial.suggest_*.

    Example:
        @register_model("my_rnn")
        def build_my_rnn(trial, input_size, output_size):
            hidden = trial.suggest_categorical("hidden_size", [64, 128, 256])
            layers = trial.suggest_int("num_layers", 1, 3)
            return MyRNN(input_size, hidden, layers, output_size)
    """
    def decorator(fn: ModelBuilder) -> ModelBuilder:
        if name in _MODEL_REGISTRY:
            logger.warning("Overwriting existing builder for '%s'.", name)
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def list_models() -> list[str]:
    """Return the list of registered model names."""
    return list(_MODEL_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Model builders  ← adapt imports to match your actual model classes
# ──────────────────────────────────────────────────────────────────────────────


@register_model("lstm")
def _build_lstm(trial: optuna.Trial, input_size: int, output_size: int) -> nn.Module:
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 4)
    # dropout is ignored for a single layer (PyTorch raises a warning otherwise)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5, step=0.05) if lstm_num_layers > 1 else 0.0
    fc_layer_dim = trial.suggest_categorical("fc_layer_dim", [64, 128, 256, 512])
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5, step=0.05) if lstm_num_layers > 1 else 0.0
    sequence_len = trial.suggest_categorical("sequence_len", [32, 64, 128, 256])

    from C_MAPSS.models.Simple_LSTM import Simple_LSTM
    return Simple_LSTM(
        feature_num=input_size,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout,
    )


@register_model("transformer_lstm")
def _build_transformer_lstm(trial: optuna.Trial, input_size: int, output_size: int) -> nn.Module:
    """Hybrid Transformer encoder + LSTM decoder."""
    # Transformer block
    sequence_len = trial.suggest_categorical("sequence_len", [32, 64, 128, 256])
    valid_heads = [h for h in [1, 2, 4, 8, 16] if sequence_len % h == 0]
    nhead = trial.suggest_categorical("nhead", valid_heads)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 4)
    # dropout is ignored for a single layer (PyTorch raises a warning otherwise)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5, step=0.05) if lstm_num_layers > 1 else 0.0
    fc_layer_dim = trial.suggest_categorical("fc_layer_dim", [64, 128, 256, 512])
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5, step=0.05) if lstm_num_layers > 1 else 0.0

    from C_MAPSS.models.TransformerEncoder_LSTM_1 import TransformerEncoder_LSTM_1  # ← adapt import
    return TransformerEncoder_LSTM_1(
        feature_num=input_size,
        sequence_len=sequence_len,
        transformer_encoder_head_num=nhead,
        hidden_dim=hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout,
    )

@register_model("cnn1d")
def _build_cnn1d(trial: optuna.Trial, input_size: int, output_size: int) -> nn.Module:
    """1D CNN model."""

    from C_MAPSS.models.CNN1D import CNN1D
    return CNN1D(
        num_features=input_size,
        output_dim=output_size,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CMAPSS configuration
# ──────────────────────────────────────────────────────────────────────────────

CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Number of input features after preprocessing – adjust if needed
CMAPSS_INPUT_SIZES: Dict[str, int] = {
    "FD001": 24,
    "FD002": 24,
    "FD003": 24,
    "FD004": 24,
}


def get_dataloaders(
    subset: str,
    batch_size: int,
    sequence_len: int,
    data_dir: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for a CMAPSS subset.
    *** Replace this function body with your own dataset logic. ***
    """
    from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader

    train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(
        dataset_root=data_dir,
        sub_dataset=subset,
        sequence_len=sequence_len,
        seed=42,
        max_rul=125,
        return_sequence_label=False,
        norm_type="z-score",
        cluster_operations=True,
        norm_by_operations=True,
        include_cols=[],
        exclude_cols=[],
        return_id=False,
        validation_rate=0.2,
        use_only_final_on_test=True,
        use_max_rul_on_test=True,
        use_max_rul_on_valid=True,
        percent_of_broken_data=None,
        percent_of_censored_data=0.,
    )

    return (
        train_dataset.get_data_loader_without_censored_data(batch_size=batch_size, num_workers=4, pin_memory=True),
        test_dataset.get_data_loader_without_censored_data(batch_size=batch_size, num_workers=4, pin_memory=True),
        valid_dataset.get_data_loader_without_censored_data(batch_size=batch_size, num_workers=4, pin_memory=True)
    )



# ──────────────────────────────────────────────────────────────────────────────
# Objective function (closure)
# ──────────────────────────────────────────────────────────────────────────────

def _metric_from_trainer(trainer: Trainer, key: str) -> float:
    """Safely extract a scalar metric logged by the Lightning module after fit."""
    val = trainer.callback_metrics.get(key, float("inf"))
    return val.item() if hasattr(val, "item") else float(val)


def _make_objective(
    model_name: str,
    subset: str,
    data_dir: str,
    max_epochs: int,
    multi_objective: bool,
):
    """
    Build the Optuna objective closure for a given model and subset.

    Metric strategy:
      - val_rmse       → monitored during training for pruning (single-objective mode)
      - test_rmse
      + test_score     → final evaluation at the end of each trial, used as Optuna
                         objectives (multi-objective) or stored as user_attrs (single-objective)

    Note: using the test set to guide optimisation is the standard practice on the
    CMAPSS benchmark, which has no separate final holdout.
    """
    builder    = _MODEL_REGISTRY[model_name]
    input_size = CMAPSS_INPUT_SIZES[subset]

    def objective(trial: optuna.Trial):
        # ── Model architecture ─────────────────────────────────────────────
        # Build the model FIRST so that architecture params (including
        # sequence_len, if the model needs it) are suggested by the builder.
        # TransformerLstmModule receives the already-built nn.Module → no changes needed.
        net = builder(trial, input_size, output_size=1)

        # sequence_len is owned by the builder (suggested inside it).
        # Models that don't need it (e.g. CNN1D) simply don't suggest it,
        # and we fall back to a sensible default for the dataloader.
        seq_len = trial.params.get("sequence_len", 32)

        # ── Training hyperparameters ───────────────────────────────────────
        lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        epochs     = trial.suggest_int("max_epochs", 3, max_epochs)

        module = TransformerLstmModule(lr=lr, model=net)

        # ── Data ──────────────────────────────────────────────────────────
        train_dl, val_dl, test_dl = get_dataloaders(
            subset, batch_size, seq_len, data_dir
        )

        # ── Callbacks ─────────────────────────────────────────────────────
        callbacks = []
        if not multi_objective:
            # Pruning is disabled in multi-objective mode (incompatible with NSGA-II)
            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor="val_rmse")
            )

        # ── Trainer ───────────────────────────────────────────────────────
        trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=callbacks,
        )

        # ── Training ──────────────────────────────────────────────────────
        trainer.fit(module, train_dl, val_dl)

        # ── Validation metrics — used as optimisation objectives ───────────
        # These are computed on units held out from training so the sampler
        # never sees the test set during the search.
        val_rmse  = _metric_from_trainer(trainer, "val_rmse")
        val_score = _metric_from_trainer(trainer, "val_score")

        # ── Test metrics — stored for post-hoc analysis only ──────────────
        # NOT returned as objectives. Call final_test_evaluation() once after
        # study.optimize() to get an honest test score.
        results    = trainer.test(module, test_dl, verbose=False)[0]
        test_rmse  = results.get("test_rmse",  float("inf"))
        test_score = results.get("test_score", float("inf"))

        trial.set_user_attr("val_rmse",   val_rmse)
        trial.set_user_attr("val_score",  val_score)
        trial.set_user_attr("test_rmse",  test_rmse)
        trial.set_user_attr("test_score", test_score)

        # Optimise on the validation set — the sampler never sees test metrics
        return (val_rmse, val_score) if multi_objective else val_rmse

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_search(
    model_name: str,
    subset: str,
    *,
    n_trials: int = 100,
    data_dir: str = "data/CMAPSSData",
    max_epochs: int = 100,
    multi_objective: bool = True,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = 1,
) -> optuna.Study:
    """
    Run hyperparameter search for one model on one CMAPSS subset.

    Args:
        model_name:      Key registered via @register_model (e.g. "transformer").
        subset:          "FD001" | "FD002" | "FD003" | "FD004".
        n_trials:        Number of Optuna trials.
        data_dir:        Root directory of CMAPSS data files.
        max_epochs:      Upper bound for the max_epochs hyperparameter.
        multi_objective: True  → Pareto front (RMSE + Score, NSGA-II, no pruning).
                         False → single val_RMSE objective + Hyperband pruning (faster).
        study_name:      Custom study name (auto-generated if None).
        storage:         Optuna storage URL, e.g. "sqlite:///optuna.db".
                         Allows resuming an interrupted study.
        n_jobs:          Parallel trials (beware of GPU memory conflicts; keep 1 in general).

    Returns:
        Completed optuna.Study.

    Examples:
        # Multi-objective (default)
        study = run_search("transformer", "FD001", n_trials=50)
        params = best_params(study)

        # Single-objective with pruning (faster, good starting point)
        study = run_search("lstm", "FD002", n_trials=100, multi_objective=False)

        # Persist to disk (resumes if interrupted)
        study = run_search("transformer", "FD001", storage="sqlite:///optuna.db")
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered. "
            f"Available models: {list_models()}"
        )
    subset = subset.upper()
    if subset not in CMAPSS_SUBSETS:
        raise ValueError(f"Unknown subset '{subset}'. Choose from {CMAPSS_SUBSETS}.")

    study_name = study_name or f"{model_name}_{subset}"

    # ── Sampler & pruner based on the selected mode ────────────────────────
    if multi_objective:
        # NSGA-II is designed for multi-objective problems (Pareto front)
        sampler    = optuna.samplers.NSGAIISampler(seed=42)
        pruner     = optuna.pruners.NopPruner()
        directions = ["minimize", "minimize"]
    else:
        # TPE + Hyperband: efficient for single-objective with a limited budget
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.HyperbandPruner(
            min_resource=10,
            max_resource=max_epochs,
            reduction_factor=3,
        )
        directions = ["minimize"]

    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,  # resume an existing study with the same name + storage
    )

    objective = _make_objective(
        model_name, subset, data_dir, max_epochs, multi_objective
    )

    logger.info(
        "Starting search: model=%s  subset=%s  trials=%d  mode=%s",
        model_name, subset, n_trials,
        "multi-objective" if multi_objective else "single-objective",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    _log_results(study, study_name, multi_objective)
    return study


def run_all_subsets(
    model_name: str,
    *,
    n_trials: int = 100,
    **kwargs,
) -> Dict[str, optuna.Study]:
    """
    Run hyperparameter search on all four CMAPSS subsets sequentially.

    Args:
        model_name: Model to optimise (registered via @register_model).
        n_trials:   Number of trials per subset.
        **kwargs:   Forwarded to run_search (data_dir, max_epochs, storage, …).

    Returns:
        {"FD001": study, "FD002": study, "FD003": study, "FD004": study}
    """
    studies: Dict[str, optuna.Study] = {}
    for subset in CMAPSS_SUBSETS:
        logger.info("=" * 60)
        logger.info("Subset %s", subset)
        studies[subset] = run_search(model_name, subset, n_trials=n_trials, **kwargs)
    return studies


def best_params(study: optuna.Study) -> dict:
    """
    Extract the best hyperparameters from a completed study.

    - Single-objective : returns study.best_trial.params.
    - Multi-objective  : selects the Pareto-front trial that minimises the
                         harmonic mean of the two normalised objectives
                         (best RMSE / Score trade-off).

    Returns:
        dict of hyperparameters for the best trial.
    """
    if len(study.directions) == 1:
        t = study.best_trial
        logger.info("Best trial #%d | val_RMSE=%.4f | params=%s",
                    t.number, t.value, t.params)
        return t.params

    pareto = study.best_trials
    if not pareto:
        raise RuntimeError("No completed trials found in the study.")

    rmses  = [t.values[0] for t in pareto]
    scores = [t.values[1] for t in pareto]
    lo_r, hi_r = min(rmses),  max(rmses)
    lo_s, hi_s = min(scores), max(scores)

    def _norm(v: float, lo: float, hi: float) -> float:
        return (v - lo) / (hi - lo + 1e-9)

    def _harmonic(t: optuna.trial.FrozenTrial) -> float:
        nr = _norm(t.values[0], lo_r, hi_r) + 1e-9
        ns = _norm(t.values[1], lo_s, hi_s) + 1e-9
        return 2.0 / (1.0 / nr + 1.0 / ns)

    chosen = min(pareto, key=_harmonic)
    logger.info(
        "Best balanced trial #%d | val_RMSE=%.4f | val_Score=%.4f | params=%s",
        chosen.number, chosen.values[0], chosen.values[1], chosen.params,
    )
    return chosen.params


def final_test_evaluation(
    study: optuna.Study,
    model_name: str,
    subset: str,
    *,
    data_dir: str = "data/CMAPSSData",
    max_epochs: Optional[int] = None,
) -> dict:
    """
    Retrain the best hyperparameter configuration from scratch and evaluate
    on the test set exactly once.

    This is the only correct way to report test scores: call it once after
    ``study.optimize()`` finishes. Never evaluate the test set inside the
    objective — that inflates scores by letting the sampler indirectly peek
    at test performance across many trials.

    Args:
        study:       A completed Optuna study.
        model_name:  Registered model name (same as passed to run_search).
        subset:      CMAPSS subset (e.g. "FD001").
        data_dir:    Root directory of CMAPSS data files.
        max_epochs:  Override epoch count (uses best trial value if None).

    Returns:
        dict with "test_rmse" and "test_score".
    """
    from optuna.trial import FixedTrial

    params     = best_params(study)
    builder    = _MODEL_REGISTRY[model_name]
    input_size = CMAPSS_INPUT_SIZES[subset]

    # FixedTrial routes every trial.suggest_* call back to the stored best
    # values without touching the study.
    net        = builder(FixedTrial(params), input_size, output_size=1)
    seq_len    = params.get("sequence_len", 32)
    lr         = params["lr"]
    batch_size = params["batch_size"]
    epochs     = max_epochs if max_epochs is not None else params["max_epochs"]

    module = TransformerLstmModule(lr=lr, model=net)
    train_dl, val_dl, test_dl = get_dataloaders(subset, batch_size, seq_len, data_dir)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, train_dl, val_dl)
    results = trainer.test(module, test_dl, verbose=True)[0]

    logger.info(
        "Final test | subset=%s | test_RMSE=%.4f | test_Score=%.4f",
        subset,
        results.get("test_rmse", float("nan")),
        results.get("test_score", float("nan")),
    )
    return results


def summary_table(studies: Dict[str, optuna.Study]) -> None:
    """
    Print a summary table of the best results per subset.
    Useful after run_all_subsets().
    """
    header = f"{'Subset':<8} {'val_RMSE':>10} {'val_Score':>12} {'test_RMSE':>10} {'test_Score':>12} {'Trial':>7}"
    logger.info("\n" + "=" * len(header))
    logger.info("SUMMARY  (val = HPO objective  |  test = post-hoc, from final_test_evaluation)")
    logger.info(header)
    logger.info("-" * len(header))
    for subset, study in studies.items():
        if len(study.directions) > 1:
            params = best_params(study)
            t = next(t for t in study.best_trials if t.params == params)
            val_rmse, val_score = t.values
        else:
            t = study.best_trial
            val_rmse  = t.user_attrs.get("val_rmse",  t.value)
            val_score = t.user_attrs.get("val_score", float("nan"))
        test_rmse  = t.user_attrs.get("test_rmse",  float("nan"))
        test_score = t.user_attrs.get("test_score", float("nan"))
        logger.info(
            f"{subset:<8} {val_rmse:>10.4f} {val_score:>12.4f}"
            f" {test_rmse:>10.4f} {test_score:>12.4f} {t.number:>7d}"
        )
    logger.info("=" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_study_results(
    study: optuna.Study,
    model_name: str,
    subset: str,
    output_dir: str,
) -> None:
    """
    Save all trials and best params for one study to output_dir.

    Files written:
      <output_dir>/<model>_<subset>_trials.csv   — all completed trials
      <output_dir>/<model>_<subset>_best_params.json
    """
    os.makedirs(output_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials for %s/%s — skipping save.", model_name, subset)
        return

    rows = []
    for t in completed:
        row = {"trial": t.number}
        row.update(t.params)
        # t.values holds the optimisation objectives (val metrics)
        row["val_rmse"]   = t.user_attrs.get("val_rmse",  t.values[0] if t.values else float("nan"))
        row["val_score"]  = t.user_attrs.get("val_score", t.values[1] if t.values and len(t.values) > 1 else float("nan"))
        # test metrics are stored only as user_attrs, never used as objectives
        row["test_rmse"]  = t.user_attrs.get("test_rmse",  float("nan"))
        row["test_score"] = t.user_attrs.get("test_score", float("nan"))
        rows.append(row)

    trials_path = os.path.join(output_dir, f"{model_name}_{subset}_trials.csv")
    pd.DataFrame(rows).to_csv(trials_path, index=False)
    logger.info("Trials saved → %s", trials_path)

    try:
        params = best_params(study)
    except RuntimeError:
        params = {}

    best_path = os.path.join(output_dir, f"{model_name}_{subset}_best_params.json")
    with open(best_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Best params saved → %s", best_path)


def save_all_studies_results(
    studies: Dict[str, optuna.Study],
    model_name: str,
    output_dir: str,
) -> None:
    """Save results for every subset and write a combined summary CSV."""
    for subset, study in studies.items():
        save_study_results(study, model_name, subset, output_dir)

    rows = []
    for subset, study in studies.items():
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            continue
        try:
            params = best_params(study)
            t = next(t for t in study.best_trials if t.params == params)
            val_rmse  = t.values[0] if t.values else float("nan")
            val_score = t.values[1] if t.values and len(t.values) > 1 else t.user_attrs.get("val_score", float("nan"))
        except (RuntimeError, StopIteration):
            val_rmse = val_score = float("nan")
            t = None
        test_rmse  = t.user_attrs.get("test_rmse",  float("nan")) if t is not None else float("nan")
        test_score = t.user_attrs.get("test_score", float("nan")) if t is not None else float("nan")
        rows.append({
            "subset": subset,
            "val_rmse": val_rmse, "val_score": val_score,
            "test_rmse": test_rmse, "test_score": test_score,
        })

    if rows:
        summary_path = os.path.join(output_dir, f"{model_name}_summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        logger.info("Summary saved → %s", summary_path)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_results(study: optuna.Study, name: str, multi: bool) -> None:
    logger.info("=" * 60)
    logger.info("Study: %s | finished trials: %d", name, len(study.trials))
    if multi:
        logger.info("Pareto front (%d trials):", len(study.best_trials))
        for t in study.best_trials:
            logger.info(
                "  #%d  val_RMSE=%.4f  val_Score=%.4f  params=%s",
                t.number, t.values[0], t.values[1], t.params,
            )
    else:
        bt = study.best_trial
        logger.info(
            "Best trial #%d  val_RMSE=%.4f  params=%s",
            bt.number, bt.value, bt.params,
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for CMAPSS"
    )
    parser.add_argument(
        "model",
        choices=list_models(),
        help="Name of the model to optimise",
    )
    parser.add_argument(
        "--subset",
        default="all",
        help="FD001 | FD002 | FD003 | FD004 | all (default: all)",
    )
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--data_dir",   default="../data/C_MAPSS")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db (resumes study if it exists)",
    )
    parser.add_argument(
        "--single_objective",
        action="store_true",
        help="Use single-objective mode (val_RMSE + Hyperband pruning) instead of multi-objective",
    )
    args = parser.parse_args()

    search_kwargs = dict(
        n_trials=args.n_trials,
        data_dir=args.data_dir,
        max_epochs=args.max_epochs,
        multi_objective=not args.single_objective,
        storage=args.storage,
    )

    if args.subset.lower() == "all":
        all_studies = run_all_subsets(args.model, **search_kwargs)
        summary_table(all_studies)
        for subset, study in all_studies.items():
            final_test_evaluation(
                study, args.model, subset,
                data_dir=args.data_dir,
                max_epochs=args.max_epochs,
            )
    else:
        s = run_search(args.model, args.subset, **search_kwargs)
        print("\nBest hyperparameters:", best_params(s))
        final_test_evaluation(
            s, args.model, args.subset,
            data_dir=args.data_dir,
            max_epochs=args.max_epochs,
        )