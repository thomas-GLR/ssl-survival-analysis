"""Shared helpers for the Scania co-training ensemble utils (v1 and v2).

Both ``utils_cotraining_ensemble_v1`` and ``utils_cotraining_ensemble_v2`` train a
``CoTrainingEnsemble`` with a **configurable number of models** and, at the end, write one
prediction file per model plus the weighted-ensemble prediction (mirroring COPROG's
``test_h1``/``test_h2`` outputs). This module holds the parts that are identical between the
two versions:

* :func:`parse_models_config` — turn the config ``models`` list (one self-contained entry per
  model) into the ``nn.Module`` list, the picklable ``module_builders`` and the per-model
  training metadata the ensemble's ``setup_training_builder`` needs.
* :func:`save_ensemble_outputs` — save every trained model, write the per-model + weighted
  prediction CSVs and a summary scores CSV, and return the weighted ``(rmse, score)``.

The model-building primitives (:func:`~scania.utils.utils_coprog._creating_model`,
:func:`~scania.utils.utils_coprog._build_scania_module`) are reused from ``utils_coprog`` so
there is a single source of truth for how a Scania model + its ``BasicLightningModule`` are
built.
"""

import functools
from typing import Callable

import pandas as pd
import torch
from lightning import LightningModule
from torch import nn

from constants import necessary_keys_scania
from scania.utils.utils_coprog import _build_scania_module, _creating_model
from scania.utils.utils_scania import generate_and_save_model_prediction
from shared.utils import ModelVersion
from shared.utils.config import assert_params_contains_all_key

# The co-training ensembles only support these four architectures (same set as COPROG).
_ALLOWED_MODEL_VERSIONS = {
    ModelVersion.CNN,
    ModelVersion.LSTM,
    ModelVersion.TRANSFORMER_FEATURES,
    ModelVersion.TRANSFORMER_TIME_SEQUENCE,
}

# Per-model entry fields (besides the architecture ``model_params`` block).
_PER_MODEL_TRAINING_FIELDS = ["model_params", "lr", "max_epochs", "patience", "rul_target_standardization"]


def _necessary_model_keys_for(model_version: ModelVersion) -> list[str]:
    """Return the required architecture-param keys for one model version."""
    match model_version:
        case ModelVersion.CNN:
            return necessary_keys_scania.NECESSARY_CNN_KEYS
        case ModelVersion.LSTM:
            return necessary_keys_scania.NECESSARY_LSTM_KEYS
        case ModelVersion.TRANSFORMER_FEATURES:
            return necessary_keys_scania.NECESSARY_TRANSFORMER_FEATURES_KEYS
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            return necessary_keys_scania.NECESSARY_TRANSFORMER_TIME_SEQUENCE_KEYS
        case _:
            raise ValueError(f"{model_version.value} is not a valid co-training ensemble model version")


def _target_standardization_stats(
        standardize: bool,
        targets_uncensored: torch.Tensor,
) -> tuple[float, float]:
    """Return ``(mean, std)`` for RUL target standardization, or ``(0.0, 1.0)`` when disabled.

    Stats are computed on the uncensored (labelled) training window targets only, to avoid
    leakage; a near-zero std collapses to ``1.0`` so standardization is a safe no-op.
    """
    if not standardize:
        return 0.0, 1.0
    mean = float(targets_uncensored.mean())
    std = float(targets_uncensored.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def parse_models_config(
        models_cfg: list[dict],
        feature_num: int,
        sequence_len: int,
        targets_uncensored: torch.Tensor,
) -> tuple[list[nn.Module], list[Callable[[], LightningModule]], dict]:
    """Parse the config ``models`` list into models, picklable builders and per-model metadata.

    Each entry of ``models_cfg`` is a single-key dict whose key is a model-version string
    (``"cnn"``, ``"lstm"``, ``"transformer_features"`` or ``"transformer_time_sequence"``) and
    whose value carries that model's ``model_params`` plus its ``lr``, ``max_epochs``,
    ``patience`` and ``rul_target_standardization`` (per-model, so adding a model is just adding
    an entry).

    Args:
        models_cfg: The ``model_params["models"]`` list from the config.
        feature_num: Number of input features (from ``ScaniaDataModule.feature_cols``).
        sequence_len: Input sequence length.
        targets_uncensored: Labelled training targets, used to compute standardization stats.

    Returns:
        ``(nn_modules, module_builders, meta)`` where ``nn_modules`` is the list of freshly
        built ``nn.Module`` (passed to the ensemble constructor for its model count),
        ``module_builders`` are picklable ``functools.partial`` callables (for the builder-style
        parallel/inline training), and ``meta`` holds the aligned lists ``version_strs``,
        ``lr``, ``max_epochs``, ``patiences``.

    Raises:
        ValueError: If fewer than two models are given, an entry is malformed, or a model
            version is not one of the four allowed architectures.
        KeyError: If a model's ``model_params`` block is missing a required architecture key.
    """
    if not isinstance(models_cfg, list) or len(models_cfg) < 2:
        raise ValueError(
            f"'models' must be a list of at least 2 models for co-training, got {models_cfg!r}.")

    nn_modules: list[nn.Module] = []
    module_builders: list[Callable[[], LightningModule]] = []
    version_strs: list[str] = []
    lrs: list[float] = []
    max_epochs_list: list[int] = []
    patiences: list[int] = []

    for idx, entry in enumerate(models_cfg):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"models[{idx}] must be a single-key dict {{model_version: {{...}}}}, got {entry!r}.")

        (version_str, inner), = entry.items()
        try:
            model_version = ModelVersion(version_str)
        except ValueError:
            raise ValueError(f"models[{idx}]: '{version_str}' is not a valid model version.")
        if model_version not in _ALLOWED_MODEL_VERSIONS:
            raise ValueError(
                f"models[{idx}]: '{version_str}' is not supported by the co-training ensemble "
                f"(allowed: {sorted(v.value for v in _ALLOWED_MODEL_VERSIONS)}).")

        assert_params_contains_all_key(inner, _PER_MODEL_TRAINING_FIELDS, f"models[{idx}].{version_str}")

        model_params = inner["model_params"]
        assert_params_contains_all_key(
            model_params, _necessary_model_keys_for(model_version), f"models[{idx}].{version_str}.model_params")

        target_mean, target_std = _target_standardization_stats(
            bool(inner["rul_target_standardization"]), targets_uncensored)

        # dict(...) copies: _creating_model mutates its params dict via .update, so keep the
        # extracted params pristine for the (picklable) builder below.
        nn_modules.append(_creating_model(dict(model_params), model_version, feature_num, sequence_len))
        module_builders.append(
            functools.partial(
                _build_scania_module,
                model_params=dict(model_params),
                model_version_value=model_version.value,
                feature_num=feature_num,
                sequence_len=sequence_len,
                lr=inner["lr"],
                target_mean=target_mean,
                target_std=target_std,
            )
        )
        version_strs.append(version_str)
        lrs.append(inner["lr"])
        max_epochs_list.append(inner["max_epochs"])
        patiences.append(inner["patience"])

    meta = {
        "version_strs": version_strs,
        "lr": lrs,
        "max_epochs": max_epochs_list,
        "patiences": patiences,
    }
    return nn_modules, module_builders, meta


def save_ensemble_outputs(
        ensemble,
        model_version: ModelVersion,
        checkpoints_path: str,
        results_path: str,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        version_strs: list[str],
) -> tuple[float, float]:
    """Save trained models, per-model + weighted prediction CSVs, and a summary scores CSV.

    Mirrors the tail of :func:`scania.utils.utils_coprog.train_model` but for an arbitrary
    number of models: one ``.pth`` per model, ``predictions_<mv>_test_h{i}_scania.csv`` per
    model, ``predictions_<mv>_test_weighted_scania.csv`` for the weighted ensemble, and a
    ``<mv>-scania.csv`` scores table.

    Args:
        ensemble: A trained ``CoTrainingEnsemble`` / ``CoTrainingEnsemble_v2`` (``.weights`` set
            via ``calculate_weights``); exposes ``lightning_modules``, ``predict`` and
            ``predict_per_model``.
        model_version: The ensemble's model version (for file naming).
        checkpoints_path: Destination directory for the per-model ``.pth`` files.
        results_path: Destination directory for the prediction/score CSVs.
        test_features: Test features.
        test_targets: Test targets.
        version_strs: Per-model architecture strings (for ``.pth`` file names).

    Returns:
        ``(rmse_weighted, score_weighted)`` for the weighted-ensemble prediction.
    """
    n = len(version_strs)

    print("Saving trained models...")
    for i, version_str in enumerate(version_strs):
        torch.save(ensemble.lightning_modules[i], f"{checkpoints_path}/{model_version.value}_{version_str}_{i}.pth")

    targets_flat = test_targets.detach().cpu().view(-1)

    # Weighted ensemble prediction.
    weighted_pred = ensemble.predict(test_features).detach().cpu().view(-1)
    rmse_weighted, score_weighted = generate_and_save_model_prediction(
        predictions=weighted_pred,
        targets=targets_flat,
        model_version=model_version.value,
        prediction_type="test_weighted",
        results_path=results_path,
    )
    print(f"Weighted ensemble | Test RMSE: {rmse_weighted} | Score: {score_weighted}")

    # Per-model (unweighted) predictions.
    per_model_preds = ensemble.predict_per_model(test_features)
    per_model_rmse: list[float] = []
    per_model_score: list[float] = []
    for i in range(n):
        pred_i = per_model_preds[i].detach().cpu().view(-1)
        rmse_i, score_i = generate_and_save_model_prediction(
            predictions=pred_i,
            targets=targets_flat,
            model_version=model_version.value,
            prediction_type=f"test_h{i}",
            results_path=results_path,
        )
        per_model_rmse.append(rmse_i)
        per_model_score.append(score_i)

    # Summary scores table with dynamic per-model columns.
    columns: list[str] = []
    row: list[float] = []
    for i in range(n):
        columns += [f"test_rmse_h{i}", f"test_score_h{i}"]
        row += [per_model_rmse[i], per_model_score[i]]
    columns += ["test_rmse_weighted", "test_score_weighted"]
    row += [rmse_weighted, score_weighted]
    for i in range(n):
        columns += [f"weight_h{i}"]
        row += [ensemble.weights[i]]

    scores = pd.DataFrame(columns=columns)
    scores.loc[0] = row
    scores.to_csv(f"{results_path}/{model_version.value}-scania.csv", index=False)

    return rmse_weighted, score_weighted
