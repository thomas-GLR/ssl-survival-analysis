import gc
import json
import logging
import os
from typing import Callable, Dict, Any, Optional

import optuna
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from optuna_integration import PyTorchLightningPruningCallback
from torch import nn
from torch.utils.data import DataLoader

from shared.enums import DatasetVersion
from C_MAPSS.hpo.optuna_search import get_dataloaders_cmapss
from scania.hpo.optuna_search import get_dataloaders_scania
from scania.lightning_module import BasicLightningModule
from shared.utils import set_seed

# Seed shared by the Optuna sampler and the vehicle split, so that both the sequence of
# suggested trials and the data every trial sees are reproducible across HPO runs.
HPO_SEED = 42

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def get_logger() -> logging.Logger:
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# CMAPSS configuration
# ──────────────────────────────────────────────────────────────────────────────

# Number of input features after preprocessing
NUMBER_FEATURES_CMAPSS_PER_SUBDATASET: Dict[str, int] = {
    "FD001": 24,
    "FD002": 24,
    "FD003": 24,
    "FD004": 24,
}


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

ModelBuilder = Callable[[optuna.Trial, int, int], nn.Module]
MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


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
        if name in MODEL_REGISTRY:
            logger.warning("Overwriting existing builder for '%s'.", name)
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def list_models() -> list[str]:
    """Return the list of registered model names."""
    return list(MODEL_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────


@register_model("cnn")
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


@register_model("lstm")
def _build_lstm(trial: optuna.Trial, feature_num: int, sequence_len: int) -> nn.Module:
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.4, step=0.1) if lstm_num_layers > 1 else 0.0
    fc_layer_dim = trial.suggest_categorical("fc_layer_dim", [64, 128, 256])
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.4, step=0.1)

    from models.Simple_LSTM import Simple_LSTM
    return Simple_LSTM(
        feature_num=feature_num,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout,
    )


@register_model("transformer_features")
def _build_transformer_features(trial: optuna.Trial, feature_num: int, sequence_len: int) -> nn.Module:
    fc_layer_dim = trial.suggest_categorical("fc_layer_dim", [64, 128, 256])
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.4, step=0.1)
    valid_heads = [h for h in [1, 2, 4, 8] if sequence_len % h == 0]
    transformer_encoder_head_num = trial.suggest_categorical("transformer_encoder_head_num", valid_heads)
    transformer_num_layer = trial.suggest_categorical("transformer_num_layer", [1, 2, 3])

    from models.TransformerFeatures import TransformerFeatures

    return TransformerFeatures(
        feature_num=feature_num,
        sequence_len=sequence_len,
        transformer_encoder_head_num=transformer_encoder_head_num,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout,
        transformer_num_layer=transformer_num_layer,
    )


@register_model("transformer_lstm")
def _build_transformer_lstm(trial: optuna.Trial, feature_num: int, sequence_len: int) -> nn.Module:
    """Hybrid Transformer encoder + LSTM decoder."""
    # Transformer block
    valid_heads = [h for h in [1, 2, 4, 8] if sequence_len % h == 0]
    nhead = trial.suggest_categorical("nhead", valid_heads)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.4, step=0.1) if lstm_num_layers > 1 else 0.0
    fc_layer_dim = trial.suggest_categorical("fc_layer_dim", [64, 128, 256])
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.4, step=0.1)

    from models.TransformerEncoder_LSTM_1 import TransformerEncoder_LSTM_1

    return TransformerEncoder_LSTM_1(
        feature_num=feature_num,
        sequence_len=sequence_len,
        transformer_encoder_head_num=nhead,
        hidden_dim=hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def metric_from_trainer(trainer: Trainer, key: str) -> float:
    """Safely extract a scalar metric logged by the Lightning module.

    Args:
        trainer: the Lightning Trainer after fit/test.
        key: the metric name to read from ``trainer.callback_metrics``.

    Returns:
        the metric as a float (``inf`` if missing).
    """
    val = trainer.callback_metrics.get(key, float("inf"))
    return val.item() if hasattr(val, "item") else float(val)


# ──────────────────────────────────────────────────────────────────────────────
# Objective function (closure)
# ──────────────────────────────────────────────────────────────────────────────

def _make_objective(
        model_name: str,
        dataset_version: DatasetVersion,
        data_dir: str,
        max_epochs: int,
        cache_dir: str | None = None,
        sub_dataset: str | None = None,
        seed: int = HPO_SEED,
) -> Callable[[optuna.Trial], float]:
    """Build the single-objective Optuna closure for a given model.

    Optimises ``val_rmse`` (minimize). ``val_score`` and the test metrics are
    stored as trial user-attrs for inspection only.

    :param model_name: registered model name.
    :param data_dir: root directory of the Scania data files.
    :param cache_dir: base cache directory.
    :param max_epochs: max epochs per trial.
    :param seed: random seed for reproducibility across Python, NumPy, Pandas,
        PyTorch, and PyTorch CUDA.
    :return: the objective callable returning ``val_rmse``.
    """
    model_builder = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        # Seed Python, NumPy, Pandas, PyTorch CPU, and PyTorch CUDA for this trial
        set_seed(seed)

        # ── Training / data hyperparameters ────────────────────────────────
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        sequence_len = trial.suggest_categorical("sequence_len", [64])

        # ── Data ────────────────────────────────────────────────────────────
        train_dl, val_dl, test_dl, feature_num, target_mean, target_std = _get_dataloaders(
            dataset_version=dataset_version,
            sequence_len=sequence_len,
            data_dir=data_dir,
            batch_size=batch_size,
            sub_dataset=sub_dataset,
            cache_dir=cache_dir,
            trial=trial,
            seed=seed,
        )

        try:

            # ── Model ─────────────────────────────────────────────────────────
            net = model_builder(trial, feature_num, sequence_len)
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
            val_rmse = metric_from_trainer(trainer, "val_rmse")
            val_score = metric_from_trainer(trainer, "val_score")

            # ── Test metrics — stored for post-hoc analysis only ──────────────
            results = trainer.test(module, test_dl, verbose=False)[0]
            test_rmse = results.get("test_rmse", float("inf"))
            test_score = results.get("test_score", float("inf"))

            trial.set_user_attr("val_rmse", val_rmse)
            trial.set_user_attr("val_score", val_score)
            trial.set_user_attr("test_rmse", test_rmse)
            trial.set_user_attr("test_score", test_score)

            return val_rmse

        finally:
            del train_dl, val_dl, test_dl
            trainer = module = net = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def _get_dataloaders(
        dataset_version: DatasetVersion,
        sequence_len: int,
        data_dir: str,
        batch_size: int,
        sub_dataset: str | None,
        cache_dir: str | None,
        trial: optuna.Trial,
        seed: int,
) -> tuple[DataLoader, DataLoader | None, DataLoader, int, float, float]:
    match dataset_version:
        case DatasetVersion.SCANIA:
            assert cache_dir is not None, f"cache_dir must not be None for {DatasetVersion.SCANIA.value} dataset"

            counter_mode = trial.suggest_categorical("counter_mode", ["delta", "cumulative", "both"])
            histogram_mode = trial.suggest_categorical("histogram_mode", ["sum", "delta", "cumulative"])
            include_histograms = trial.suggest_categorical("include_histograms", [True])
            standardize_target = trial.suggest_categorical("standardize_target", [True, False])

            # ── Data ────────────────────────────────────────────────────────────
            train_dl, val_dl, test_dl, feature_num, target_mean, target_std = get_dataloaders_scania(
                sequence_len=sequence_len,
                counter_mode=counter_mode,
                histogram_mode=histogram_mode,
                include_histograms=include_histograms,
                standardize_target=standardize_target,
                batch_size=batch_size,
                data_dir=data_dir,
                cache_dir=cache_dir,
                seed=seed,
            )

            return train_dl, val_dl, test_dl, feature_num, target_mean, target_std
        case DatasetVersion.CMAPSS:
            assert sub_dataset is not None, f"sub_dataset must not be None for {DatasetVersion.CMAPSS.value} dataset"
            # ── Data ──────────────────────────────────────────────────────────
            train_dl, val_dl, test_dl = get_dataloaders_cmapss(
                sub_dataset=sub_dataset,
                batch_size=batch_size,
                sequence_len=sequence_len,
                data_dir=data_dir,
                seed=seed,
            )

            feature_num = NUMBER_FEATURES_CMAPSS_PER_SUBDATASET[sub_dataset]

            return train_dl, val_dl, test_dl, feature_num, 0.0, 1.0
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_search(
        model_name: str,
        dataset_version: DatasetVersion,
        data_dir: str,
        sub_dataset: str | None,
        n_trials: int = 100,
        max_epochs: int = 100,
        cache_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_jobs: int = 1,
        seed: int = HPO_SEED,
) -> optuna.Study:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered. Available models: {list_models()}"
        )

    # Set seed globally at search start
    set_seed(seed)

    if cache_dir is None:
        cache_dir = os.path.join(data_dir, "scania_cache")

    match dataset_version:
        case DatasetVersion.SCANIA:
            default_study_name = f"{model_name}_scania"
        case DatasetVersion.CMAPSS:
            assert sub_dataset is not None, "sub_dataset can't be None for CMAPSS"

            default_study_name = f"{model_name}_{sub_dataset}_turbofan"
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")

    study_name = study_name or default_study_name

    # TPE + Hyperband: efficient for single-objective with a limited budget
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

    objective = _make_objective(
        model_name=model_name,
        dataset_version=dataset_version,
        data_dir=data_dir,
        max_epochs=max_epochs,
        cache_dir=cache_dir,
        sub_dataset=sub_dataset,
        seed=seed,
    )

    logger.info(
        "Starting search: model=%s  subset=%s  trials=%d  mode=single-objective (val_rmse)",
        model_name,
        sub_dataset,
        n_trials,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

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


def summary_table(studies_per_name: Dict[str, optuna.Study]) -> None:
    """
    Print a summary table of the best results per subset.
    Useful after run_all_subsets().
    """
    header = f"{'Study':<8} {'val_RMSE':>10} {'val_Score':>12} {'test_RMSE':>10} {'test_Score':>12} {'Trial':>7}"
    logger.info("\n" + "=" * len(header))
    logger.info("SUMMARY  (val = HPO objective  |  test = post-hoc, from final_test_evaluation)")
    logger.info(header)
    logger.info("-" * len(header))
    for study_name, study in studies_per_name.items():
        t = study.best_trial
        val_rmse = t.user_attrs.get("val_rmse", t.value)
        val_score = t.user_attrs.get("val_score", float("nan"))
        test_rmse = t.user_attrs.get("test_rmse", float("nan"))
        test_score = t.user_attrs.get("test_score", float("nan"))
        logger.info(
            f"{study_name:<8} {val_rmse:>10.4f} {val_score:>12.4f}"
            f" {test_rmse:>10.4f} {test_score:>12.4f} {t.number:>7d}"
        )
    logger.info("=" * len(header))


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

    params = best_params(study)
    builder = MODEL_REGISTRY[model_name]
    input_size = NUMBER_FEATURES_CMAPSS_PER_SUBDATASET[subset]

    # FixedTrial routes every trial.suggest_* call back to the stored best
    # values without touching the study.
    net = builder(FixedTrial(params), input_size, output_size=1)
    seq_len = params.get("sequence_len", 32)
    lr = params["lr"]
    batch_size = params["batch_size"]
    epochs = max_epochs if max_epochs is not None else params["max_epochs"]

    module = BasicLightningModule(lr=lr, model=net)
    train_dl, val_dl, test_dl = get_dataloaders_cmapss(subset, batch_size, seq_len, data_dir)

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


# ──────────────────────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_study_results(
        study: optuna.Study,
        model_name: str,
        dataset_version: DatasetVersion,
        sub_dataset: str | None,
        output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials for %s/%s — skipping save.", model_name, sub_dataset)
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

    match dataset_version:
        case DatasetVersion.SCANIA:
            file_name_trials = f"{model_name}_trials.csv"
            file_name_best_params = f"{model_name}_best_params.json"
        case DatasetVersion.CMAPSS:
            file_name_trials = f"{model_name}_{sub_dataset}_trials.csv"
            file_name_best_params = f"{model_name}_{sub_dataset}_best_params.json"
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")

    try:
        params = best_params(study)
    except RuntimeError:
        params = {}


    trials_path = os.path.join(output_dir, file_name_trials)
    pd.DataFrame(rows).to_csv(trials_path, index=False)
    logger.info("Trials saved → %s", trials_path)

    best_path = os.path.join(output_dir, file_name_best_params)
    with open(best_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Best params saved → %s", best_path)


def save_all_studies_results(
        studies: Dict[str, optuna.Study],
        model_name: str,
        dataset_version: DatasetVersion,
        output_dir: str,
) -> None:
    """Save results for every subset and write a combined summary CSV."""
    for sub_dataset, study in studies.items():
        save_study_results(
            study=study,
            model_name=model_name,
            dataset_version=dataset_version,
            sub_dataset=sub_dataset,
            output_dir=output_dir
        )

    rows = []
    for sub_dataset, study in studies.items():
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            continue
        try:
            params = best_params(study)
            t = next(t for t in study.best_trials if t.params == params)
            val_rmse = t.values[0] if t.values else float("nan")
            val_score = t.values[1] if t.values and len(t.values) > 1 else t.user_attrs.get("val_score", float("nan"))
        except (RuntimeError, StopIteration):
            val_rmse = val_score = float("nan")
            t = None
        test_rmse = t.user_attrs.get("test_rmse", float("nan")) if t is not None else float("nan")
        test_score = t.user_attrs.get("test_score", float("nan")) if t is not None else float("nan")
        rows.append({
            "subset": sub_dataset,
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

def _log_results(study: optuna.Study, name: str) -> None:
    """Log the best trial of a finished study.

    :param study: the finished Optuna study.
    :param name: the study name (for the log line).
    """
    logger.info("=" * 60)
    logger.info("Study: %s | finished trials: %d", name, len(study.trials))
    bt = study.best_trial
    logger.info("Best trial #%d  val_rmse=%.4f  params=%s", bt.number, bt.value, bt.params)
