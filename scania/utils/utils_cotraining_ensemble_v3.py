"""Scania training entry point for :class:`models.CoTrainingEnsemble_v3` (v3).

Same structure as :mod:`scania.utils.utils_cotraining_ensemble_v2` (configurable number of
models, per-model prediction files, builder-style parallel training), but v3 uses owner-based
selection, a latent-kNN pseudo-label estimator over the conformal predictive band, backward
extrapolation of the last-window RUL, and fine-tuning instead of from-scratch retraining. It
therefore takes the extra selection / estimator / fine-tuning knobs and always threads the
per-window ``time_step`` for the backward extrapolation.

Shared model/builder construction and output saving come from
:mod:`scania.utils.utils_cotraining_common`; the RMSE weighting callback is reused from
:mod:`scania.utils.utils_coprog`.
"""

import os
from datetime import datetime

from models.CoTrainingEnsemble_v3 import CoTrainingEnsemble_v3
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
    n_neighbors: int,
    fine_tune_lr_factor: float,
    fine_tune_max_epochs: int,
    fine_tune_patience: int,
    model_pred_blend: float = 0.0,
    inference_batch_size: int | None = None,
    use_monotone_projection: bool = False,
    monotone_residual_weight: float = 1.0,
    # Others
    gpu_ids: list[int] | None = None,
    datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train a :class:`models.CoTrainingEnsemble_v3` on the Scania Component X dataset.

    Args:
        checkpoints_path: Root directory for model checkpoints.
        results_path: Root directory for result CSVs.
        model_version: The dispatched model version (``CO_TRAINING_ENSEMBLE_V3``).
        models: The config ``models`` list; one self-contained entry per model (see
            :func:`scania.utils.utils_cotraining_common.parse_models_config`).
        dataset_root: Path to the Scania dataset.
        sequence_len, seed, data_fraction, val_rate, test_rate, stratify, norm_type,
        shuffle_loader, cache_dir, num_workers, pin_memory, return_sequence_label, batch_size,
        counter_mode: ``ScaniaDataModule`` construction params.
        iterations: Number of co-training iterations.
        suspension_pool_size: Fraction in ``(0, 1]`` of censored units sampled as the pool each
            iteration.
        add_ratio: Per-model top fraction in ``(0, 1]`` — each model selects its top ``add_ratio``
            of the pool (by its own conformal width) per iteration; ownership follows from which
            models selected each unit.
        confidence: Confidence level in ``(0, 1)`` defining the conformal percentile band
            (``a``/``c``/``b`` = the ``(1-confidence)/2`` / 50 / ``(1+confidence)/2`` percentiles).
        n_neighbors: Number of nearest labelled neighbours (latent space) for the k-NN estimator.
        fine_tune_lr_factor: Learning-rate multiplier for fine-tuning (warm start).
        fine_tune_max_epochs: Max epochs per fine-tuning call.
        fine_tune_patience: ``EarlyStopping`` patience per fine-tuning call.
        model_pred_blend: ``alpha`` in ``[0, 1]`` blending the model median ``c`` with the k-NN
            estimate (``0`` = pure k-NN clamped to the conformal band).
        inference_batch_size: If set, chunk every forward pass (prediction + embedding) into
            batches of this size to cap peak memory. ``None`` keeps single-shot inference.
        use_monotone_projection: When ``True``, the monotone-projection residual of each unit's
            raw predictions is blended into each model's own ranking score (a within-model
            self-consistency signal used only to order that model's own units). It does not change
            the injected label.
        monotone_residual_weight: Weight of the residual term (only used when
            ``use_monotone_projection`` is ``True``).
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
    # Per-window time_step for the censored data (row-aligned with the censored features/ids),
    # used to backward-extrapolate each selected unit's last-window RUL to its earlier windows.
    _, _, suspension_time_steps = scania_data_module.get_censored_time_steps("train")
    # Per-window survival lower bounds, only needed when the monotone-projection residual (with
    # censoring clip) feeds the selection score.
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
        "n_neighbors": n_neighbors,
        "model_pred_blend": model_pred_blend,
        "fine_tune_lr_factor": fine_tune_lr_factor,
        "fine_tune_max_epochs": fine_tune_max_epochs,
        "fine_tune_patience": fine_tune_patience,
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

    print(f"Creating co-training ensemble v3 with {number_of_models} models: {meta['version_strs']}")

    ensemble = CoTrainingEnsemble_v3(
        models=nn_modules,
        verbose=1,
        confidence=confidence,
        inference_batch_size=inference_batch_size,
        use_monotone_projection=use_monotone_projection,
        monotone_residual_weight=monotone_residual_weight,
        n_neighbors=n_neighbors,
        model_pred_blend=model_pred_blend,
        fine_tune_lr_factor=fine_tune_lr_factor,
        fine_tune_max_epochs=fine_tune_max_epochs,
        fine_tune_patience=fine_tune_patience,
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

    print("Training co-training ensemble (v3)...")

    log_file_path = os.path.join(results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Co-Training Ensemble (v3) run ===\n")
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
        suspension_time_steps=suspension_time_steps,
        suspension_lower_bounds=suspension_lower_bounds,
        iterations=iterations,
        suspension_pool_size=suspension_pool_size,
        add_ratio=add_ratio,
        val_data=val_features,
        val_label=val_targets,
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
