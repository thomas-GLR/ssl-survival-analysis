"""Scania training entry point for :class:`models.CoTrainingEnsemble` (v1).

Mirrors :mod:`scania.utils.utils_coprog` but for a co-training ensemble with a **configurable
number of models** (driven by the config ``models`` list) using the four allowed architectures
(``cnn``, ``lstm``, ``transformer_features``, ``transformer_time_sequence``). At the end it
writes one prediction file per model plus the weighted-ensemble prediction (like COPROG).

The heavy lifting shared with v2 (model/builder construction, output saving) lives in
:mod:`scania.utils.utils_cotraining_common`; the model-building primitives and the RMSE
weighting callback are reused from :mod:`scania.utils.utils_coprog`.
"""

import os
from datetime import datetime

from models.CoTrainingEnsemble import CoTrainingEnsemble, SelectionMode
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
    is_fine_tuning_during_finding_best_suspension_data: bool,
    is_fine_tuning_for_last_step: bool,
    fine_tune_lr_factor: float,
    fine_tune_max_epochs: int,
    inference_batch_size: int | None = None,
    # Others
    gpu_ids: list[int] | None = None,
    datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train a :class:`models.CoTrainingEnsemble` on the Scania Component X dataset.

    Args:
        checkpoints_path: Root directory for model checkpoints.
        results_path: Root directory for result CSVs.
        model_version: The dispatched model version (``CO_TRAINING_ENSEMBLE``).
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
        is_fine_tuning_during_finding_best_suspension_data: Fine-tune (vs from-scratch) during
            the candidate search.
        is_fine_tuning_for_last_step: Fine-tune (vs from-scratch) after selecting censored data.
        fine_tune_lr_factor: LR multiplier applied while fine-tuning (see ``CoTrainingEnsemble``).
        fine_tune_max_epochs: Max epochs per fine-tuning call (candidate search and last-step
            retrain). Capped well below the from-scratch ``max_epochs`` so the many fine-tune
            fits stay cheap; applied to every model.
        inference_batch_size: If set, chunk every ``_predict`` forward pass into batches of this
            size so peak (host) memory during candidate scoring / metrics stays ``O(batch)``.
            Needed to fit small budgets (e.g. Colab T4). ``None`` keeps single-shot inference.
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
            "selection and for the ensemble weights. Set validation_rate > 0 in the config."
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
    # Labelled (uncensored) validation data: used both for early stopping / best-checkpoint
    # selection during training and to compute the ensemble weights (instead of the test set).
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
        "is_fine_tuning_during_finding_best_suspension_data": is_fine_tuning_during_finding_best_suspension_data,
        "is_fine_tuning_for_last_step": is_fine_tuning_for_last_step,
        "fine_tune_lr_factor": fine_tune_lr_factor,
        "fine_tune_max_epochs": fine_tune_max_epochs,
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

    print(f"Creating co-training ensemble with {number_of_models} models: {meta['version_strs']}")

    ensemble = CoTrainingEnsemble(
        models=nn_modules,
        verbose=1,
        fine_tune_lr_factor=fine_tune_lr_factor,
        inference_batch_size=inference_batch_size,
    )

    # gpu_ids (from --gpu-ids): None -> single GPU / auto; [g] -> pinned; [g0, g1, ...] ->
    # parallel training distributed round-robin across those GPUs. Builders must be picklable
    # (module-level function + functools.partial) so they survive the process boundary.
    print(f"Co-training ensemble GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}")

    ensemble.setup_training_builder(
        module_builders=module_builders,
        max_epochs=meta["max_epochs"],
        patiences=meta["patiences"],
        batchs_size=[batch_size] * number_of_models,
        shuffle_dataloaders=[True] * number_of_models,
        # Same fine-tune epoch budget for every model; capped below the from-scratch
        # max_epochs so the many per-candidate / last-step fine-tune fits stay cheap.
        fine_tune_max_epochs=[fine_tune_max_epochs] * number_of_models,
        gpu_ids=gpu_ids,
    )

    print("Training co-training ensemble (v1)...")

    # Persistent run log next to the results. Created (truncated) here with a metadata
    # header, then handed to the ensemble which appends every log message under it
    # regardless of verbose (mirrors the C_MAPSS co-training entry point).
    log_file_path = os.path.join(results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Co-Training Ensemble (v1) run ===\n")
        f.write(f"Datetime: {datetime_for_folders}\n")
        f.write(f"Model version: {model_version.value}\n")
        f.write(f"Models ({number_of_models}): {meta['version_strs']}\n")
        f.write(f"GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}\n")
        f.write("=====================================\n")

    ensemble.train(
        is_fine_tuning_during_finding_best_suspension_data=is_fine_tuning_during_finding_best_suspension_data,
        is_fine_tuning_for_last_step=is_fine_tuning_for_last_step,
        selection_mode=SelectionMode.VOTING,
        train_with_censored_data=False,
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        iterations=iterations,
        suspension_pool_size=suspension_pool_size,
        add_ratio=add_ratio,
        val_data=val_features,
        val_label=val_targets,
        # Per-stage metrics (initial / iteration_k / final). The score columns use the Scania
        # score, while the reported weights use RMSE + "min" (matching calculate_weights below).
        # Runs in the main process (safe for the parallel path).
        test_data=test_features,
        test_label=test_targets,
        score_callback=_score_callback_for_coprog,
        weight_callback=_criteria_callback_for_coprog,
        weight_mode="min",
        metrics_file=f"{results_path}/{model_version.value}-per-stage-scania.csv",
        log_file=log_file_path,
    )

    # Ensemble weights are computed on the validation set (not the test set) to avoid leakage.
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
