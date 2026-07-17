"""Scania training entry point for :class:`models.CoTrainingEnsemble_v2` (v2).

Same structure as :mod:`scania.utils.utils_cotraining_ensemble_v1` (configurable number of
models, per-model prediction files), but v2 always trains from scratch (no fine-tuning) and
ranks censored units by ``crepes`` conformal interval width, so it takes a ``confidence``
param instead of the fine-tuning knobs.

Shared model/builder construction and output saving come from
:mod:`scania.utils.utils_cotraining_common`; the RMSE weighting callback is reused from
:mod:`scania.utils.utils_coprog`.
"""

import os
from datetime import datetime

from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2
from scania.dataset import ScaniaDataModule
from scania.utils.utils_cotraining_common import parse_models_config, save_ensemble_outputs
from scania.utils.utils_coprog import _criteria_callback_for_coprog, _score_callback_for_coprog
from scania.utils.utils_scania import (
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
    save_train_parameters,
)
from shared.utils import ModelVersion


def train_model(
    checkpoints_path: str,
    results_path: str,
    model_version: ModelVersion,
    # Model params
    models: list[dict],
    # Dataset params
    dataset_root: str,
    sequence_len: int,
    seed: int | None,
    data_fraction: float,
    val_rate: float,
    test_rate: float,
    stratify: bool,
    norm_type: str | None,
    shuffle_loader: bool,
    cache_dir: str | None,
    num_workers: int,
    pin_memory: bool,
    return_sequence_label: bool,
    batch_size: int,
    counter_mode: str,
    # Training params
    iterations: int,
    suspension_pool_size: float,
    add_ratio: float,
    confidence: float,
    inference_batch_size: int | None = None,
    use_monotone_projection: bool = False,
    monotone_residual_weight: float = 1.0,
    # Others
    gpu_ids: list[int] | None = None,
    datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train a :class:`models.CoTrainingEnsemble_v2` on the Scania Component X dataset.

    Args:
        checkpoints_path: Root directory for model checkpoints.
        results_path: Root directory for result CSVs.
        model_version: The dispatched model version (``CO_TRAINING_ENSEMBLE_V2``).
        models: The config ``models`` list; one self-contained entry per model (see
            :func:`scania.utils.utils_cotraining_common.parse_models_config`).
        dataset_root: Path to the Scania dataset.
        sequence_len, seed, val_rate, test_rate, stratify, norm_type, shuffle_loader,
        cache_dir, num_workers, pin_memory, return_sequence_label, batch_size, counter_mode:
            ``ScaniaDataModule`` construction params.
        iterations: Number of co-training iterations.
        suspension_pool_size: Fraction in ``(0, 1]`` of censored units sampled as the pool each
            iteration.
        add_ratio: Fraction in ``(0, 1]`` of the pool to add per iteration.
        confidence: Confidence level in ``(0, 1)`` for the ``crepes`` conformal intervals used
            to rank censored units.
        inference_batch_size: If set, chunk every ``_predict`` forward pass into batches of this
            size so peak (host) memory during conformal scoring / metrics stays ``O(batch)``.
            Needed to fit small budgets (e.g. Colab T4). ``None`` keeps single-shot inference.
        use_monotone_projection: When ``True``, each censored unit's per-window pseudo-labels are
            projected onto the closest non-increasing sequence and clipped up to the per-window
            survival lower bound; the projection residual is blended into unit selection. ``False``
            (default) keeps the legacy width-only scoring.
        monotone_residual_weight: Weight of the residual term in the blended selection score (only
            used when ``use_monotone_projection`` is ``True``).
        gpu_ids: GPU id(s). ``None`` → single GPU / auto (sequential); ``[g]`` → pinned; two or
            more → parallel training across those GPUs.
        datetime_for_folders: Timestamp used to name the output folders.

    Returns:
        ``(rmse_weighted, score_weighted)`` of the weighted-ensemble test prediction.
    """
    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
    )

    checkpoints_path, results_path = create_and_get_checkpoints_results_path(
        model_version=model_version.value,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    if val_rate <= 0:
        raise ValueError(
            "The co-training ensemble needs a validation set for early stopping / best-model "
            "selection, conformal calibration and the ensemble weights. Set validation_rate > 0."
        )

    dataset_kwargs = {
        "data_dir": dataset_root,
        "seed": seed,
        "data_fraction": data_fraction,
        "val_rate": val_rate,
        "test_rate": test_rate,
        "stratify": stratify,
        "norm_type": norm_type,
        "shuffle_loader": shuffle_loader,
        "cache_dir": cache_dir,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "return_sequence_label": return_sequence_label,
        "batch_size": batch_size,
        "sequence_len": sequence_len,
        "counter_mode": counter_mode,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    scania_data_module = ScaniaDataModule(**dataset_kwargs)
    scania_data_module.setup()

    feature_num = len(scania_data_module.feature_cols)

    features_uncensored, targets_uncensored, features_censored, ids_censored = \
        scania_data_module.get_cotraining_tensors("train")
    # Per-window survival lower bounds for the censored data (row-aligned with the censored
    # features/ids above), used by the monotone-projection scoring to clip pseudo-labels up to
    # the observed time-to-study-end. Fetched only when needed.
    suspension_lower_bounds = None
    if use_monotone_projection:
        _, _, suspension_lower_bounds = scania_data_module.get_censored_lower_bounds("train")
    # Labelled (uncensored) validation data: early stopping / best-checkpoint selection,
    # conformal calibration set, and the ensemble weights (instead of the test set).
    val_features, val_targets, _, _ = scania_data_module.get_cotraining_tensors("val")
    test_features, test_targets, _, _ = scania_data_module.get_cotraining_tensors("test")

    nn_modules, module_builders, meta = parse_models_config(
        models_cfg=models,
        feature_num=feature_num,
        sequence_len=sequence_len,
        targets_uncensored=targets_uncensored,
    )
    number_of_models = len(nn_modules)

    training_kwargs = {
        "iterations": iterations,
        "suspension_pool_size": suspension_pool_size,
        "add_ratio": add_ratio,
        "confidence": confidence,
        "inference_batch_size": inference_batch_size,
        "use_monotone_projection": use_monotone_projection,
        "monotone_residual_weight": monotone_residual_weight,
        "lr": meta["lr"],
        "max_epochs": meta["max_epochs"],
        "patiences": meta["patiences"],
    }

    save_train_parameters(
        results_path=results_path,
        dataset_parameters=dataset_kwargs,
        training_parameters=training_kwargs,
        model_parameters={"models": models},
    )

    print(f"Creating co-training ensemble v2 with {number_of_models} models: {meta['version_strs']}")

    ensemble = CoTrainingEnsemble_v2(
        models=nn_modules,
        verbose=1,
        confidence=confidence,
        inference_batch_size=inference_batch_size,
        use_monotone_projection=use_monotone_projection,
        monotone_residual_weight=monotone_residual_weight,
    )

    print(f"Co-training ensemble GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}")

    ensemble.setup_training_builder(
        module_builders=module_builders,
        max_epochs=meta["max_epochs"],
        patiences=meta["patiences"],
        batchs_size=[batch_size] * number_of_models,
        shuffle_dataloaders=[True] * number_of_models,
        gpu_ids=gpu_ids,
    )

    print("Training co-training ensemble (v2)...")

    # Persistent run log next to the results. Created (truncated) here with a metadata
    # header, then handed to the ensemble which appends every log message under it
    # regardless of verbose (mirrors the C_MAPSS co-training entry point).
    log_file_path = os.path.join(results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Co-Training Ensemble (v2) run ===\n")
        f.write(f"Datetime: {datetime_for_folders}\n")
        f.write(f"Model version: {model_version.value}\n")
        f.write(f"Models ({number_of_models}): {meta['version_strs']}\n")
        f.write(f"GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}\n")
        f.write("=====================================\n")

    ensemble.train(
        train_with_censored_data=False,
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        suspension_lower_bounds=suspension_lower_bounds,
        iterations=iterations,
        suspension_pool_size=suspension_pool_size,
        add_ratio=add_ratio,
        val_data=val_features,
        val_label=val_targets,
        # Per-stage metrics: the score columns use the Scania score, while the reported
        # weights use RMSE + "min" (matching calculate_weights below).
        test_data=test_features,
        test_label=test_targets,
        score_callback=_score_callback_for_coprog,
        weight_callback=_criteria_callback_for_coprog,
        weight_mode="min",
        metrics_file=f"{results_path}/{model_version.value}-per-stage-scania.csv",
        log_file=log_file_path,
    )

    ensemble.calculate_weights(
        x_test=val_features,
        target=val_targets,
        criteria_callback=_criteria_callback_for_coprog,
        mode="min",
    )

    return save_ensemble_outputs(
        ensemble=ensemble,
        model_version=model_version,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        test_features=test_features,
        test_targets=test_targets,
        version_strs=meta["version_strs"],
    )
