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
* :func:`load_ensemble_for_inference` (and the lower-level :func:`load_module_checkpoint` /
  :func:`load_ensemble_modules`) — rebuild a trained ensemble from its saved checkpoints so a
  finished run can be re-tested without retraining.

The model-building primitives (:func:`~scania.utils.utils_coprog._creating_model`,
:func:`~scania.utils.utils_coprog._build_scania_module`) are reused from ``utils_coprog`` so
there is a single source of truth for how a Scania model + its ``BasicLightningModule`` are
built.
"""

import functools
import glob
import os
from typing import Any, Callable

import lightning
import pandas as pd
import torch
from lightning import LightningModule
from torch import nn

from constants import necessary_keys_scania
from scania.lightning_module import BasicLightningModule
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

# Top-level checkpoint key holding the extra Scania metadata (architecture spec, position in the
# ensemble, ensemble weights). Lightning's loader only reads the keys it knows about and ignores
# everything else, so this rides along inside a checkpoint that ``load_from_checkpoint`` accepts.
CHECKPOINT_SPEC_KEY = "scania_cotraining"


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
        ``lr``, ``max_epochs``, ``patiences`` and ``model_specs`` (the per-model architecture
        recipe embedded into each saved checkpoint by :func:`save_ensemble_outputs`, so a
        trained model can be rebuilt from its checkpoint alone).

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
    model_specs: list[dict] = []
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
        # The same four arguments ``_creating_model`` needs, kept as plain JSON-ish values so the
        # spec can be embedded in a checkpoint and reloaded with ``weights_only=True``.
        model_specs.append({
            "model_version": model_version.value,
            "model_params": dict(model_params),
            "feature_num": feature_num,
            "sequence_len": sequence_len,
        })
        version_strs.append(version_str)
        lrs.append(inner["lr"])
        max_epochs_list.append(inner["max_epochs"])
        patiences.append(inner["patience"])

    meta = {
        "version_strs": version_strs,
        "lr": lrs,
        "max_epochs": max_epochs_list,
        "patiences": patiences,
        "model_specs": model_specs,
    }
    return nn_modules, module_builders, meta


def save_module_checkpoint(
        module: LightningModule,
        path: str,
        spec: dict | None = None,
) -> None:
    """Save one ``BasicLightningModule`` as a real Lightning checkpoint.

    Writes the three keys Lightning's loader reads — ``pytorch-lightning_version`` (gates the
    checkpoint-migration pass), ``hyper_parameters`` (replayed as ``cls(**hparams)``) and
    ``state_dict`` — plus, under :data:`CHECKPOINT_SPEC_KEY`, the Scania metadata needed to
    rebuild the wrapped ``nn.Module``.

    This replaces pickling the live module object (``torch.save(module, path)``), which produced
    a file that :meth:`~lightning.pytorch.LightningModule.load_from_checkpoint` rejects (it is a
    module, not a checkpoint dict) and that only unpickles when every architecture class is
    importable and allow-listed under PyTorch 2.6's ``weights_only=True`` default. Everything
    written here is a tensor or a primitive, so the file loads either way.

    Args:
        module: The trained module to save.
        path: Destination ``.pth`` path.
        spec: Optional metadata stored under :data:`CHECKPOINT_SPEC_KEY`. When it carries the
            ``model_version``/``model_params``/``feature_num``/``sequence_len`` keys produced by
            :func:`parse_models_config`, the checkpoint can rebuild its own architecture on load.
    """
    hyper_parameters = dict(module.hparams)
    # The two survival-loss settings are flipped on the *instance* after construction by the
    # co-training jobs (see BasicLightningModule.__init__), which never updates ``hparams``.
    # Snapshot the live values so a reloaded module is configured exactly like the saved one.
    if isinstance(module, BasicLightningModule):
        hyper_parameters["use_cotraining_ensemble_survival_loss_function"] = (
            module.use_cotraining_ensemble_survival_loss_function)
        hyper_parameters["cotraining_survival_loss_lambda"] = module.cotraining_survival_loss_lambda

    checkpoint: dict[str, Any] = {
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": hyper_parameters,
        "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    }
    if spec is not None:
        checkpoint[CHECKPOINT_SPEC_KEY] = spec

    torch.save(checkpoint, path)


def _model_from_spec(spec: dict) -> nn.Module:
    """Rebuild the raw ``nn.Module`` described by a checkpoint's spec block.

    Args:
        spec: The :data:`CHECKPOINT_SPEC_KEY` block, holding ``model_version``, ``model_params``,
            ``feature_num`` and ``sequence_len``.

    Returns:
        A freshly built (untrained) ``nn.Module`` of the right architecture and shape.

    Raises:
        KeyError: If the spec is missing one of the four architecture keys.
    """
    missing = [key for key in ("model_version", "model_params", "feature_num", "sequence_len")
               if key not in spec]
    if missing:
        raise KeyError(
            f"The checkpoint's '{CHECKPOINT_SPEC_KEY}' block cannot rebuild the architecture: "
            f"missing {missing}. Pass the model explicitly via the 'model' argument instead.")
    # dict(...): _creating_model mutates its params dict via .update.
    return _creating_model(
        dict(spec["model_params"]),
        ModelVersion(spec["model_version"]),
        spec["feature_num"],
        spec["sequence_len"],
    )


def load_module_checkpoint(
        path: str,
        model: nn.Module | None = None,
        map_location: str | torch.device = "cpu",
) -> LightningModule:
    """Load one saved co-training model back into a ``BasicLightningModule``.

    Handles both on-disk formats:

    * the Lightning checkpoint written by :func:`save_module_checkpoint` — rebuilt through
      ``BasicLightningModule.load_from_checkpoint``, so ``lr``, ``target_mean`` and
      ``target_std`` are restored from the saved ``hyper_parameters`` (de-normalized predictions
      depend on the last two);
    * the **legacy** whole-module pickle written by older runs (``torch.save(module, path)``) —
      returned as-is, since the pickled object already carries its weights and hyperparameters.

    Args:
        path: Path to the ``.pth`` file.
        model: The raw architecture to wrap. ``None`` (default) rebuilds it from the checkpoint's
            spec block; required for a checkpoint saved without one.
        map_location: Device to map the stored tensors onto. Defaults to CPU.

    Returns:
        The restored module, in ``eval`` mode.

    Raises:
        ValueError: If the file is neither a Lightning checkpoint nor a pickled module.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, LightningModule):
        # Legacy format: the module itself was pickled, so nothing needs rebuilding.
        module = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        if model is None:
            spec = checkpoint.get(CHECKPOINT_SPEC_KEY)
            if spec is None:
                raise ValueError(
                    f"{path} has no '{CHECKPOINT_SPEC_KEY}' block, so its architecture cannot be "
                    f"rebuilt automatically. Pass the model explicitly via the 'model' argument.")
            model = _model_from_spec(spec)
        module = BasicLightningModule.load_from_checkpoint(
            checkpoint_path=path, model=model, map_location=map_location)
    else:
        raise ValueError(
            f"{path} is neither a Lightning checkpoint (a dict with a 'state_dict' key) nor a "
            f"pickled LightningModule; got {type(checkpoint).__name__}.")

    module.eval()
    return module


def read_checkpoint_spec(path: str) -> dict | None:
    """Return a checkpoint's :data:`CHECKPOINT_SPEC_KEY` block, or ``None`` if it has none.

    Args:
        path: Path to the ``.pth`` file.

    Returns:
        The spec dict, or ``None`` for a legacy checkpoint saved before specs existed.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint.get(CHECKPOINT_SPEC_KEY)
    return None


def _checkpoint_index(path: str, model_version_value: str) -> int | None:
    """Extract the ensemble position ``i`` from a ``<mv>_<version_str>_<i>.pth`` file name.

    Args:
        path: Path to a candidate checkpoint file.
        model_version_value: The ensemble's model-version string (the file-name prefix).

    Returns:
        The index, or ``None`` when the name does not match the pattern — which is also how a
        ``co_training_ensemble_v2_*`` file is rejected while scanning for ``co_training_ensemble``
        (v1), whose prefix is a prefix of v2's.
    """
    stem = os.path.basename(path)[: -len(".pth")]
    prefix = f"{model_version_value}_"
    if not stem.startswith(prefix):
        return None
    version_str, _, index_str = stem[len(prefix):].rpartition("_")
    if not index_str.isdigit():
        return None
    try:
        if ModelVersion(version_str) not in _ALLOWED_MODEL_VERSIONS:
            return None
    except ValueError:
        return None
    return int(index_str)


def load_ensemble_modules(
        checkpoints_path: str,
        model_version: ModelVersion = ModelVersion.CO_TRAINING_ENSEMBLE_V2,
        map_location: str | torch.device = "cpu",
) -> tuple[list[LightningModule], list[float] | None]:
    """Load every per-model checkpoint of one ensemble run, in ensemble order.

    Discovers the ``<model_version>_<version_str>_<i>.pth`` files written by
    :func:`save_ensemble_outputs` and orders them by ``i``, so model ``i`` lines up with the
    ``weight_h{i}`` / ``test_rmse_h{i}`` columns of the run's CSVs.

    Args:
        checkpoints_path: The run's checkpoint directory.
        model_version: Which ensemble's files to look for. Defaults to
            ``ModelVersion.CO_TRAINING_ENSEMBLE_V2``.
        map_location: Device to map the stored tensors onto. Defaults to CPU.

    Returns:
        ``(modules, weights)``, where ``weights`` are the ensemble weights recorded in the
        checkpoints, or ``None`` for legacy checkpoints that predate them.

    Raises:
        FileNotFoundError: If the directory holds no matching checkpoint.
        ValueError: If the indices do not form the complete range ``0..n-1``.
    """
    candidates = glob.glob(os.path.join(checkpoints_path, f"{model_version.value}_*.pth"))
    indexed = [(index, path) for path in sorted(candidates)
               if (index := _checkpoint_index(path, model_version.value)) is not None]
    if not indexed:
        raise FileNotFoundError(
            f"No '{model_version.value}_<version>_<i>.pth' checkpoint found in {checkpoints_path}.")

    indexed.sort()
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"Expected checkpoints indexed 0..{len(indices) - 1} in {checkpoints_path}, got {indices}.")

    modules: list[LightningModule] = []
    weights: list[float] | None = None
    for _, path in indexed:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        spec = checkpoint.get(CHECKPOINT_SPEC_KEY) if isinstance(checkpoint, dict) else None
        if spec is not None and spec.get("ensemble_weights") is not None:
            weights = [float(w) for w in spec["ensemble_weights"]]
        modules.append(load_module_checkpoint(path, map_location=map_location))

    return modules, weights


def read_ensemble_weights_from_results(results_csv_path: str, number_of_models: int) -> list[float]:
    """Read the ensemble weights back out of a run's ``<model_version>-scania.csv``.

    The fallback for legacy checkpoints, which do not carry their weights.

    Args:
        results_csv_path: Path to the run's ``<model_version>-scania.csv``.
        number_of_models: How many ``weight_h{i}`` columns to read.

    Returns:
        The weights, ordered by model index.

    Raises:
        KeyError: If a ``weight_h{i}`` column is missing.
    """
    scores = pd.read_csv(results_csv_path)
    missing = [f"weight_h{i}" for i in range(number_of_models) if f"weight_h{i}" not in scores.columns]
    if missing:
        raise KeyError(f"{results_csv_path} has no {missing} column(s).")
    return [float(scores.loc[0, f"weight_h{i}"]) for i in range(number_of_models)]


def load_ensemble_for_inference(
        checkpoints_path: str,
        model_version: ModelVersion = ModelVersion.CO_TRAINING_ENSEMBLE_V2,
        weights: list[float] | None = None,
        results_csv_path: str | None = None,
        map_location: str | torch.device = "cpu",
        inference_batch_size: int | None = None,
):
    """Rebuild a trained ``CoTrainingEnsemble_v2`` from its checkpoints, ready to predict.

    The returned ensemble is **inference-only**: its models and weights are restored, so
    ``predict`` / ``predict_per_model`` reproduce the run's saved predictions, but nothing needed
    to resume training (builders, trainers, accumulated datasets) is set.

    Args:
        checkpoints_path: The run's checkpoint directory.
        model_version: Which ensemble's files to load. Defaults to
            ``ModelVersion.CO_TRAINING_ENSEMBLE_V2``.
        weights: Ensemble weights, overriding whatever the checkpoints carry. Needed only for
            legacy checkpoints saved without them (or to re-weight deliberately).
        results_csv_path: Fallback source for the weights — the run's
            ``<model_version>-scania.csv``, whose ``weight_h{i}`` columns are read when the
            checkpoints carry none and ``weights`` is not given.
        map_location: Device to map the stored tensors onto. Defaults to CPU.
        inference_batch_size: Chunk size for ``predict``'s forward passes. ``None`` (default)
            feeds the whole tensor at once; set it to bound peak memory on a large test set.

    Returns:
        A ``CoTrainingEnsemble_v2`` with ``lightning_modules`` and ``weights`` populated.

    Raises:
        ValueError: If the weights can be found neither in the checkpoints, nor in ``weights``,
            nor in ``results_csv_path``.
    """
    # Local import: models.CoTrainingEnsemble_v2 pulls in crepes, which the v1 path that also
    # imports this module does not need.
    from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2

    modules, stored_weights = load_ensemble_modules(
        checkpoints_path=checkpoints_path, model_version=model_version, map_location=map_location)

    if weights is None:
        weights = stored_weights
    if weights is None and results_csv_path is not None:
        weights = read_ensemble_weights_from_results(results_csv_path, len(modules))
    if weights is None:
        raise ValueError(
            f"The checkpoints in {checkpoints_path} carry no ensemble weights (they predate the "
            f"Lightning checkpoint format). Pass 'weights', or point 'results_csv_path' at the "
            f"run's {model_version.value}-scania.csv to read its weight_h* columns.")
    if len(weights) != len(modules):
        raise ValueError(f"Got {len(weights)} weights for {len(modules)} models.")

    ensemble = CoTrainingEnsemble_v2(
        models=[module.net for module in modules],
        weights=weights,
        inference_batch_size=inference_batch_size,
    )
    ensemble.lightning_modules = modules
    return ensemble


def save_ensemble_outputs(
        ensemble,
        model_version: ModelVersion,
        checkpoints_path: str,
        results_path: str,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        version_strs: list[str],
        model_specs: list[dict] | None = None,
        training_time_seconds: float | None = None,
        avg_iteration_time_seconds: float | None = None,
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
        model_specs: Per-model architecture recipes from ``parse_models_config``'s
            ``meta["model_specs"]``, embedded in each checkpoint so
            :func:`load_ensemble_for_inference` can rebuild the models with no config on hand.
            ``None`` still writes loadable checkpoints, but reloading them then requires passing
            the architecture explicitly.
        training_time_seconds: Wall-clock duration of the whole training step, written as the
            ``training_time_seconds`` column. ``None`` (default) leaves the cell empty.
        avg_iteration_time_seconds: Mean wall-clock duration of one co-training iteration,
            written as the ``avg_iteration_time_seconds`` column. ``None`` (default) leaves the
            cell empty — v1 does not track per-iteration durations.

    Returns:
        ``(rmse_weighted, score_weighted)`` for the weighted-ensemble prediction.
    """
    n = len(version_strs)

    print("Saving trained models...")
    ensemble_weights = (
        [float(weight) for weight in ensemble.weights] if ensemble.weights is not None else None)
    for i, version_str in enumerate(version_strs):
        spec = dict(model_specs[i]) if model_specs is not None else {}
        # Position in the ensemble and the full weight vector, so a checkpoint set can be
        # reassembled into a weighted ensemble without the run's CSVs.
        spec["model_index"] = i
        spec["version_str"] = version_str
        spec["ensemble_weights"] = ensemble_weights
        save_module_checkpoint(
            module=ensemble.lightning_modules[i],
            path=f"{checkpoints_path}/{model_version.value}_{version_str}_{i}.pth",
            spec=spec,
        )

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
    row: list[float | None] = []
    for i in range(n):
        columns += [f"test_rmse_h{i}", f"test_score_h{i}"]
        row += [per_model_rmse[i], per_model_score[i]]
    columns += ["test_rmse_weighted", "test_score_weighted"]
    row += [rmse_weighted, score_weighted]
    for i in range(n):
        columns += [f"weight_h{i}"]
        row += [ensemble.weights[i]]
    columns += ["training_time_seconds", "avg_iteration_time_seconds"]
    row += [training_time_seconds, avg_iteration_time_seconds]

    scores = pd.DataFrame(columns=columns)
    scores.loc[0] = row
    scores.to_csv(f"{results_path}/{model_version.value}-scania.csv", index=False)

    return rmse_weighted, score_weighted
