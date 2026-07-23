"""Scania training entry point for :class:`models.CoTrainingEnsemble_v3` (v3).

Same structure as :mod:`scania.utils.utils_cotraining_ensemble_v2` (configurable number of
models, per-model prediction files), but v3 is single-GPU and uses owner-based confidence
selection over a ``predict_int`` interval, a pseudo-label taken from the owner models' own
predictions, backward extrapolation of the last-window RUL, and fine-tuning with per-model
best-model retention. It therefore takes the extra confidence / fine-tuning knobs, always
threads the per-window ``time_step`` (backward extrapolation) and the per-window survival
``lower_bound`` (confidence score), and can use a dedicated calibration split for ``crepes``.

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
    calib_rate: float,
    stratify: bool,
    norm_type: str | None,
    shuffle_loader: bool,
    cache_dir: str | None,
    num_workers: int,
    pin_memory: bool,
    return_sequence_label: bool,
    batch_size: int,
    counter_mode: str,
    include_histograms: bool,
    histogram_mode: str,
    # Training params
    iterations: int,
    suspension_pool_size: float,
    add_ratio: float,
    confidence: float,
    fine_tune_lr_factor: float,
    fine_tune_max_epochs: int,
    fine_tune_patience: int,
    difficulty_space: str = "raw",
    confidence_threshold: float | None = None,
    w_mono: float = 1.0,
    w_lb: float = 1.0,
    keep_best_model: bool = True,
    inference_batch_size: int | None = None,
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
        sequence_len, seed, data_fraction, val_rate, test_rate, calib_rate, stratify, norm_type,
        shuffle_loader, cache_dir, num_workers, pin_memory, return_sequence_label, batch_size,
        counter_mode, include_histograms, histogram_mode: ``ScaniaDataModule`` construction params.
            ``calib_rate > 0`` carves a dedicated by-vehicle calibration split used to calibrate the
            ``crepes`` conformal regressors (instead of the validation set).
        iterations: Number of co-training iterations.
        suspension_pool_size: Fraction in ``(0, 1]`` of censored units sampled as the pool each
            iteration.
        add_ratio: Per-model top fraction in ``(0, 1]`` — each model selects its top ``add_ratio``
            of the pool (by confidence) per iteration; ownership follows from which models selected
            each unit.
        confidence: Confidence level in ``(0, 1)`` passed to ``crepes`` ``predict_int``.
        fine_tune_lr_factor: Learning-rate multiplier for fine-tuning (warm start).
        fine_tune_max_epochs: Max epochs per fine-tuning call.
        fine_tune_patience: ``EarlyStopping`` patience per fine-tuning call.
        difficulty_space: ``"raw"`` (default) or ``"latent"`` — where the conformal difficulty
            estimator measures each unit's neighbourhood. ``"latent"`` uses each model's own
            pre-head embedding so the models disagree about which units they understand best.
        confidence_threshold: Optional lower bound in ``(0, 1]`` on a unit's confidence; when set,
            only units at or above it are selectable. ``None`` disables the gate.
        w_mono: Non-negative weight of the monotonicity-violation term in the confidence score.
        w_lb: Non-negative weight of the lower-bound-violation term in the confidence score.
        keep_best_model: When ``True`` (default), a fine-tuned model is kept per iteration only if
            its validation RMSE improved (else reverted and its added units dropped). ``False``
            keeps every fine-tuned model — the baseline to compare retention against.
        inference_batch_size: If set, chunk every forward pass into batches of this size to cap
            peak memory. ``None`` keeps single-shot inference.
        gpu_ids: GPU id(s). ``None`` → auto (single GPU); ``[g]`` → pinned to GPU ``g``. v3 is
            single-GPU: if several ids are passed only the first is used.
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
        "calib_rate": calib_rate,
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
        "include_histograms": include_histograms,
        "histogram_mode": histogram_mode,
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
    # Per-window survival lower bounds (always needed: the confidence score's lower-bound term).
    _, _, suspension_lower_bounds = scania_data_module.get_censored_lower_bounds("train")
    # Labelled (uncensored) validation data: early stopping / best-checkpoint selection and the
    # ensemble weights (instead of the test set).
    val_features, val_targets, _, _ = scania_data_module.get_cotraining_tensors("val")
    test_features, test_targets, _, _ = scania_data_module.get_cotraining_tensors("test")
    # Dedicated calibration split for the crepes conformal regressors (else the ensemble falls
    # back to the validation set inside train()).
    calib_features, calib_targets = None, None
    if calib_rate > 0:
        calib_features, calib_targets, _, _ = scania_data_module.get_cotraining_tensors("calib")

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
        "difficulty_space": difficulty_space,
        "confidence_threshold": confidence_threshold,
        "w_mono": w_mono,
        "w_lb": w_lb,
        "keep_best_model": keep_best_model,
        "fine_tune_lr_factor": fine_tune_lr_factor,
        "fine_tune_max_epochs": fine_tune_max_epochs,
        "fine_tune_patience": fine_tune_patience,
        "inference_batch_size": inference_batch_size,
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
        difficulty_space=difficulty_space,
        confidence_threshold=confidence_threshold,
        w_mono=w_mono,
        w_lb=w_lb,
        keep_best_model=keep_best_model,
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
        calib_data=calib_features,
        calib_label=calib_targets,
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
