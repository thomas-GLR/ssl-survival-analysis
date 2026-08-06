"""Scania training entry point for :class:`models.classification.CoTrainingEnsembleOrdinalRegression`.

Same shape as :mod:`scania.utils.utils_cotraining_ensemble_v2`, but for the 5-class ordinal
failure-urgency problem instead of RUL regression. The differences that matter to a caller:

* the data comes from :class:`~scania.dataset.ScaniaClassificationDataModule`, whose val/test
  splits are the fixed ``validation_*``/``test_*`` label files (so it takes no ``val_rate`` /
  ``test_rate`` / ``stratify``), and whose censored vehicles live only in the train split;
* the per-window ``class_upper_bound`` is **always** fetched — it is what makes the
  "impossible class" filter possible, not an optional extra;
* the returned pair is ``(macro-F1, Scania cost)`` rather than ``(RMSE, C-MAPSS score)``.

Model/builder construction and output saving come from
:mod:`scania.utils.utils_cotraining_ordinal_common`.
"""

import os
from datetime import datetime

from models.classification import ordinal_metrics
from models.classification.CoTrainingEnsembleOrdinalRegression import (
    CoTrainingEnsembleOrdinalRegression,
    ComputingWeightMode,
    ImprovingMetricMode,
)
from scania.dataset import ScaniaClassificationDataModule
from scania.utils.utils_cotraining_ordinal_common import (
    parse_ordinal_models_config,
    save_ordinal_ensemble_outputs,
)
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
    p_value_treshold: float,
    # Optional params must follow every non-default one; ``calib_rate`` is a dataset_params key
    # (forwarded to ScaniaClassificationDataModule below) placed here only for that reason.
    calib_rate: float = 0.0,
    inference_batch_size: int | None = None,
    num_classes: int = ordinal_metrics.NUM_CLASSES,
    bagging_failure_data: bool = False,
    use_fine_tuning: bool = False,
    fine_tune_lr_factor: float = 0.1,
    fine_tune_max_epochs: int = 20,
    fine_tune_patience: int = 5,
    keep_best_model_mode: str | None = None,
    computing_weight_mode: str = ComputingWeightMode.F1_SCORE_MACRO.value,
    pool_seed: int | None = None,
    # Others
    gpu_ids: list[int] | None = None,
    datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train a co-training ensemble of ordinal classifiers on the Scania Component X dataset.

    Args:
        checkpoints_path: Root directory for model checkpoints.
        results_path: Root directory for result CSVs.
        model_version: The dispatched model version (``CO_TRAINING_ENSEMBLE_ORDINAL``).
        models: The config ``models`` list; one self-contained entry per model (see
            :func:`~scania.utils.utils_cotraining_ordinal_common.parse_ordinal_models_config`).
        dataset_root: Path to the Scania dataset.
        sequence_len, seed, data_fraction, norm_type, shuffle_loader, cache_dir, num_workers,
        pin_memory, return_sequence_label, batch_size, counter_mode, include_histograms,
        histogram_mode:
            ``ScaniaClassificationDataModule`` construction params.
        iterations: Number of co-training iterations.
        suspension_pool_size: Fraction in ``(0, 1]`` of censored units sampled as the pool each
            iteration.
        add_ratio: Fraction in ``(0, 1]`` of the pool consumed per iteration.
        confidence: Confidence level in ``(0, 1)`` for the conformal prediction sets. A censored
            vehicle is only usable by a model when every one of its windows yields a singleton
            set at this level.
        p_value_treshold: Minimum p-value the predicted class must reach on every window.
        calib_rate: Fraction of *uncensored* train vehicles held out to calibrate the conformal
            classifiers. Must be ``> 0``: unlike the regression ensembles there is no fallback
            to the validation split, since that split also drives early stopping.
        inference_batch_size: If set, chunk every forward pass into batches of this size so peak
            memory during scoring / metrics stays ``O(batch)``. ``None`` keeps single-shot
            inference.
        num_classes: Number of ordered failure-urgency classes.
        bagging_failure_data: Give each model a bootstrap resample of the labelled data, for
            committee diversity.
        use_fine_tuning: Fine-tune from the post-initial-training snapshot on the accumulated
            pseudo-labels instead of retraining from scratch each iteration.
        fine_tune_lr_factor: Learning-rate multiplier for a fine-tune.
        fine_tune_max_epochs: Epoch budget for a fine-tune.
        fine_tune_patience: ``EarlyStopping`` patience for a fine-tune.
        keep_best_model_mode: Name of an :class:`ImprovingMetricMode` (e.g. ``"f1_score_macro"``);
            a retrained model then only replaces the previous one when it improves that metric.
            ``None`` always accepts the new model.
        computing_weight_mode: Name of a :class:`ComputingWeightMode` used to weight the models.
        pool_seed: Seed for the candidate-pool RNG.
        gpu_ids: ``None`` → auto device selection; ``[g]`` → pin to GPU ``g``. Two or more ids
            are rejected: this ensemble is single-GPU only.
        datetime_for_folders: Timestamp used to name the output folders.

    Returns:
        ``(f1_macro_weighted, cost_weighted)`` of the weighted-ensemble test prediction.

    Raises:
        ValueError: If ``calib_rate <= 0``.
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

    if calib_rate <= 0:
        raise ValueError(
            "The ordinal co-training ensemble needs a dedicated calibration split to calibrate "
            "its conformal classifiers (the validation split is already used for early stopping "
            "and the ensemble weights). Set calib_rate > 0."
        )

    dataset_kwargs = {
        "data_dir": dataset_root,
        "seed": seed,
        "data_fraction": data_fraction,
        "calib_rate": calib_rate,
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

    scania_data_module = ScaniaClassificationDataModule(**dataset_kwargs)
    scania_data_module.setup()

    feature_num = len(scania_data_module.feature_cols)

    features_uncensored, targets_uncensored, features_censored, ids_censored = \
        scania_data_module.get_cotraining_tensors("train")
    # Per-window upper bound on the unknown true class of each censored window, row-aligned with
    # the censored features/ids above. The accessor is named for the regression case (where the
    # bound is a lower bound on RUL); for the classification dataset it returns
    # ``class_upper_bound`` — see ScaniaBaseDataset.get_censored_lower_bounds.
    _, _, suspension_class_upper_bounds = scania_data_module.get_censored_lower_bounds("train")

    # Validation drives early stopping / best-checkpoint selection and the ensemble weights;
    # calibration is a separate split so the conformal p-values are not anti-conservative.
    val_features, val_targets, _, _ = scania_data_module.get_cotraining_tensors("val")
    test_features, test_targets, _, _ = scania_data_module.get_cotraining_tensors("test")
    calib_features, calib_targets, _, _ = scania_data_module.get_cotraining_tensors("calib")

    nn_modules, module_builders, meta = parse_ordinal_models_config(
        models_cfg=models,
        feature_num=feature_num,
        sequence_len=sequence_len,
        num_classes=num_classes,
    )
    number_of_models = len(nn_modules)

    keep_best = ImprovingMetricMode(keep_best_model_mode) if keep_best_model_mode else None
    weight_mode = ComputingWeightMode(computing_weight_mode)

    training_kwargs = {
        "iterations": iterations,
        "suspension_pool_size": suspension_pool_size,
        "add_ratio": add_ratio,
        "confidence": confidence,
        "p_value_treshold": p_value_treshold,
        "inference_batch_size": inference_batch_size,
        "num_classes": num_classes,
        "bagging_failure_data": bagging_failure_data,
        "use_fine_tuning": use_fine_tuning,
        "fine_tune_lr_factor": fine_tune_lr_factor,
        "fine_tune_max_epochs": fine_tune_max_epochs,
        "fine_tune_patience": fine_tune_patience,
        "keep_best_model_mode": keep_best_model_mode,
        "computing_weight_mode": computing_weight_mode,
        "pool_seed": pool_seed,
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

    print(f"Creating ordinal co-training ensemble with {number_of_models} models: {meta['version_strs']}")

    ensemble = CoTrainingEnsembleOrdinalRegression(
        models=nn_modules,
        verbose=1,
        num_classes=num_classes,
        bagging_failure_data=bagging_failure_data,
        confidence=confidence,
        p_value_treshold=p_value_treshold,
        inference_batch_size=inference_batch_size,
        use_fine_tuning=use_fine_tuning,
        fine_tune_lr_factor=fine_tune_lr_factor,
        fine_tune_max_epochs=fine_tune_max_epochs,
        fine_tune_patience=fine_tune_patience,
        keep_best_model_mode=keep_best,
        computing_weight_mode=weight_mode,
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

    print("Training ordinal co-training ensemble...")

    # Persistent run log next to the results. Created (truncated) here with a metadata
    # header, then handed to the ensemble which appends every log message under it
    # regardless of verbose.
    log_file_path = os.path.join(results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Co-Training Ensemble (ordinal) run ===\n")
        f.write(f"Datetime: {datetime_for_folders}\n")
        f.write(f"Model version: {model_version.value}\n")
        f.write(f"Models ({number_of_models}): {meta['version_strs']}\n")
        f.write(f"GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}\n")
        f.write("==========================================\n")

    ensemble.train(
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        suspension_class_upper_bounds=suspension_class_upper_bounds,
        iterations=iterations,
        suspension_pool_size=suspension_pool_size,
        add_ratio=add_ratio,
        val_data=val_features,
        val_label=val_targets,
        calib_data=calib_features,
        calib_label=calib_targets,
        test_data=test_features,
        test_label=test_targets,
        metrics_file=f"{results_path}/{model_version.value}-per-stage-scania.csv",
        log_file=log_file_path,
        pool_seed=pool_seed,
    )

    ensemble.calculate_weights(x_test=val_features, target=val_targets)

    return save_ordinal_ensemble_outputs(
        ensemble=ensemble,
        model_version=model_version,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        test_features=test_features,
        test_targets=test_targets,
        version_strs=meta["version_strs"],
    )
