"""Shared helpers for the Scania **ordinal-classification** co-training ensemble.

The classification counterpart of :mod:`scania.utils.utils_cotraining_common`. It builds the
same four architectures, but from ``models/classification/`` and wrapped in an
:class:`~scania.lightning_module.OrdinalLightningModule` (which in turn wraps them in
``spacecutter``'s ``OrdinalLogisticModel``) instead of a ``BasicLightningModule``.

Two differences from the regression version are worth calling out:

* there is no ``rul_target_standardization`` per model — class indices are categorical
  positions, so there is nothing to standardize;
* the saved outputs are classification metrics (accuracy, macro-F1, MAE and RMSE on the class
  index, and the Scania cost) rather than RMSE + the C-MAPSS score.

The classification architectures reuse the class names of their regression originals
(``CNN1D``, ``Simple_LSTM``, ...), so they are imported under aliases here.
"""

import functools
from typing import Callable

import pandas as pd
import torch
from lightning import LightningModule
from torch import nn

from constants import necessary_keys_scania
from models.classification import ordinal_metrics
from models.classification.CNN1DOrdinalRegression import CNN1D as OrdinalCNN1D
from models.classification.SimpleLstmOrdinalRegression import Simple_LSTM as OrdinalSimpleLstm
from models.classification.TransformerFeaturesOrdinalRegression import (
    TransformerFeatures as OrdinalTransformerFeatures,
)
from models.classification.TransformerTimeSequenceOrdinalRegression import (
    TransformerTimeSequence as OrdinalTransformerTimeSequence,
)
from scania.lightning_module import OrdinalLightningModule
from shared.utils import ModelVersion
from shared.utils.config import assert_params_contains_all_key

# The co-training ensembles only support these four architectures (same set as COPROG).
_ALLOWED_MODEL_VERSIONS = {
    ModelVersion.CNN,
    ModelVersion.LSTM,
    ModelVersion.TRANSFORMER_FEATURES,
    ModelVersion.TRANSFORMER_TIME_SEQUENCE,
}

# Per-model entry fields (besides the architecture ``model_params`` block). Unlike the
# regression ensembles there is no ``rul_target_standardization``.
_PER_MODEL_TRAINING_FIELDS = ["model_params", "lr", "max_epochs", "patience"]


def _necessary_model_keys_for(model_version: ModelVersion) -> list[str]:
    """Return the required architecture-param keys for one model version.

    Args:
        model_version: The architecture to look up.

    Returns:
        The list of required keys inside that model's ``model_params`` block.

    Raises:
        ValueError: If the version is not one of the four supported architectures.
    """
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


def _creating_ordinal_model(
        model_params: dict,
        model_version: ModelVersion,
        num_features: int,
        sequence_len: int,
) -> nn.Module:
    """Build one ordinal *predictor* from its config block.

    The returned module still has the scalar ``(B, 1)`` head every architecture in this repo
    uses — that single number is exactly what ``LogisticCumulativeLink`` turns into a class
    distribution, so no architectural change is needed. The ordinal wrapping happens in
    :func:`_build_ordinal_module`.

    Args:
        model_params: The architecture's constructor params (mutated in place via ``.update``,
            so callers pass a copy).
        model_version: Which architecture to build.
        num_features: Number of input features.
        sequence_len: Input sequence length.

    Returns:
        The freshly built ``nn.Module``.

    Raises:
        ValueError: If the version is not one of the four supported architectures.
    """
    match model_version:
        case ModelVersion.CNN:
            return OrdinalCNN1D(num_features=num_features)
        case ModelVersion.LSTM:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
            })
            return OrdinalSimpleLstm(**model_params)
        case ModelVersion.TRANSFORMER_FEATURES:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
            })
            return OrdinalTransformerFeatures(**model_params)
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
                "d_model": sequence_len,
            })
            # Configs expose the layer count as ``transformer_num_layer`` (like
            # TransformerFeatures), but TransformerTimeSequence's constructor calls it
            # ``num_layers`` — map it so the config schema stays consistent across models.
            if "transformer_num_layer" in model_params:
                model_params["num_layers"] = model_params.pop("transformer_num_layer")
            return OrdinalTransformerTimeSequence(**model_params)
        case _:
            raise ValueError(
                f"{model_version.value} is not a valid model version for the ordinal ensemble")


def _build_ordinal_module(
        model_params: dict,
        model_version_value: str,
        feature_num: int,
        sequence_len: int,
        lr: float,
        num_classes: int,
) -> LightningModule:
    """Build a fresh :class:`OrdinalLightningModule` wrapping a fresh predictor.

    This is a **module-level** function (not a closure) so that, wrapped in
    ``functools.partial``, it is picklable and can be handed to the training job runner.

    Args:
        model_params: The architecture's constructor params. Copied before use because
            :func:`_creating_ordinal_model` mutates it via ``.update``.
        model_version_value: The model version string (e.g. ``"cnn"``), passed as its value
            rather than the enum to keep the partial picklable.
        feature_num: Number of input features.
        sequence_len: Input sequence length.
        lr: Learning rate for the wrapping module.
        num_classes: Number of ordered classes.

    Returns:
        A fresh ``OrdinalLightningModule`` ready to train.
    """
    model = _creating_ordinal_model(
        dict(model_params), ModelVersion(model_version_value), feature_num, sequence_len)
    return OrdinalLightningModule(lr=lr, model=model, num_classes=num_classes)


def parse_ordinal_models_config(
        models_cfg: list[dict],
        feature_num: int,
        sequence_len: int,
        num_classes: int = ordinal_metrics.NUM_CLASSES,
) -> tuple[list[nn.Module], list[Callable[[], LightningModule]], dict]:
    """Parse the config ``models`` list into predictors, picklable builders and metadata.

    Each entry of ``models_cfg`` is a single-key dict whose key is a model-version string
    (``"cnn"``, ``"lstm"``, ``"transformer_features"`` or ``"transformer_time_sequence"``) and
    whose value carries that model's ``model_params`` plus its ``lr``, ``max_epochs`` and
    ``patience`` (per-model, so adding a model is just adding an entry).

    Args:
        models_cfg: The ``model_params["models"]`` list from the config.
        feature_num: Number of input features (from the data module's ``feature_cols``).
        sequence_len: Input sequence length.
        num_classes: Number of ordered classes.

    Returns:
        ``(nn_modules, module_builders, meta)`` where ``nn_modules`` is the list of freshly
        built predictors (passed to the ensemble constructor for its model count),
        ``module_builders`` are picklable ``functools.partial`` callables, and ``meta`` holds
        the aligned lists ``version_strs``, ``lr``, ``max_epochs``, ``patiences``.

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

        # dict(...) copies: _creating_ordinal_model mutates its params dict via .update, so keep
        # the extracted params pristine for the (picklable) builder below.
        nn_modules.append(
            _creating_ordinal_model(dict(model_params), model_version, feature_num, sequence_len))
        module_builders.append(
            functools.partial(
                _build_ordinal_module,
                model_params=dict(model_params),
                model_version_value=model_version.value,
                feature_num=feature_num,
                sequence_len=sequence_len,
                lr=inner["lr"],
                num_classes=num_classes,
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


def _save_ordinal_prediction(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_version: str,
        prediction_type: str,
        results_path: str,
) -> dict[str, float]:
    """Write one prediction CSV and return its classification metrics.

    Args:
        predictions: Predicted class indices.
        targets: True class indices.
        model_version: Model version string, used in the file name.
        prediction_type: ``"test_weighted"`` / ``"test_h{i}"``, used in the file name.
        results_path: Destination directory.

    Returns:
        ``{"accuracy", "f1_macro", "mae", "rmse", "cost"}`` for this prediction.
    """
    frame = pd.DataFrame({
        "prediction": predictions.detach().cpu().reshape(-1).long().numpy(),
        "target": targets.detach().cpu().reshape(-1).long().numpy(),
    })
    frame.to_csv(
        f"{results_path}/predictions_{model_version}_{prediction_type}_scania.csv", index=False)

    return {
        "accuracy": ordinal_metrics.accuracy(predictions, targets),
        "f1_macro": ordinal_metrics.f1_macro(predictions, targets),
        "mae": ordinal_metrics.mae(predictions, targets),
        "rmse": ordinal_metrics.rmse(predictions, targets),
        "cost": ordinal_metrics.scania_cost(predictions, targets),
    }


def save_ordinal_ensemble_outputs(
        ensemble,
        model_version: ModelVersion,
        checkpoints_path: str,
        results_path: str,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        version_strs: list[str],
) -> tuple[float, float]:
    """Save trained models, per-model + weighted prediction CSVs, and a summary scores CSV.

    The classification counterpart of
    :func:`scania.utils.utils_cotraining_common.save_ensemble_outputs`: one ``.pth`` per model,
    ``predictions_<mv>_test_h{i}_scania.csv`` per model,
    ``predictions_<mv>_test_weighted_scania.csv`` for the weighted ensemble, and a
    ``<mv>-scania.csv`` scores table.

    Args:
        ensemble: A trained ``CoTrainingEnsembleOrdinalRegression`` whose ``weights`` have been
            set via ``calculate_weights``.
        model_version: The ensemble's model version (for file naming).
        checkpoints_path: Destination directory for the per-model ``.pth`` files.
        results_path: Destination directory for the prediction/score CSVs.
        test_features: Test features.
        test_targets: Test targets (class indices).
        version_strs: Per-model architecture strings (for the ``.pth`` file names).

    Returns:
        ``(f1_macro_weighted, cost_weighted)`` for the weighted-ensemble prediction — the macro-F1
        to maximize and the Scania cost to minimize.
    """
    n = len(version_strs)

    print("Saving trained models...")
    for i, version_str in enumerate(version_strs):
        torch.save(ensemble.lightning_modules[i], f"{checkpoints_path}/{model_version.value}_{version_str}_{i}.pth")

    targets_flat = test_targets.detach().cpu().reshape(-1)

    weighted_metrics = _save_ordinal_prediction(
        predictions=ensemble.predict(test_features).reshape(-1),
        targets=targets_flat,
        model_version=model_version.value,
        prediction_type="test_weighted",
        results_path=results_path,
    )
    print(f"Weighted ensemble | Test accuracy: {weighted_metrics['accuracy']:.4f} | "
          f"macro-F1: {weighted_metrics['f1_macro']:.4f} | cost: {weighted_metrics['cost']:.1f}")

    per_model_preds = ensemble.predict_per_model(test_features)
    per_model_metrics: list[dict[str, float]] = []
    for i in range(n):
        per_model_metrics.append(
            _save_ordinal_prediction(
                predictions=per_model_preds[i].reshape(-1),
                targets=targets_flat,
                model_version=model_version.value,
                prediction_type=f"test_h{i}",
                results_path=results_path,
            )
        )

    metric_keys = ["accuracy", "f1_macro", "mae", "rmse", "cost"]

    columns: list[str] = []
    row: list[float] = []
    for i in range(n):
        columns += [f"test_{key}_h{i}" for key in metric_keys]
        row += [per_model_metrics[i][key] for key in metric_keys]
    columns += [f"test_{key}_weighted" for key in metric_keys]
    row += [weighted_metrics[key] for key in metric_keys]
    for i in range(n):
        columns += [f"weight_h{i}"]
        row += [ensemble.weights[i]]

    scores = pd.DataFrame(columns=columns)
    scores.loc[0] = row
    scores.to_csv(f"{results_path}/{model_version.value}-scania.csv", index=False)

    return weighted_metrics["f1_macro"], weighted_metrics["cost"]
