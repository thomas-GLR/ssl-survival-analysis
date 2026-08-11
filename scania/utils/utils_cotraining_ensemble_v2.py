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
from shared.utils import ModelVersion, set_seed


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
    include_histograms: bool,
    histogram_mode: str,
    # Training params
    iterations: int,
    suspension_pool_size: float,
    add_ratio: float,
    confidence: float,
    # calib_rate is a dataset_params key (forwarded to ScaniaDataModule below); it is placed
    # here, after the required params, only because it is optional (default 0.0, backward
    # compatible with configs predating it) and Python requires defaulted params to follow
    # every non-default one.
    calib_rate: float = 0.0,
    inference_batch_size: int | None = None,
    use_monotone_projection: bool = False,
    monotone_residual_weight: float = 1.0,
    # Opt-in CoTrainingEnsemble_v2 levers (all default to legacy behavior). Settable from the
    # ``training_params`` block of the config JSON.
    use_fine_tuning: bool = False,
    fine_tune_lr_factor: float = 0.1,
    fine_tune_max_epochs: int = 20,
    fine_tune_patience: int = 5,
    fine_tune_from_initial_model: bool = False,
    peer_weighted_pseudo_label: bool = False,
    keep_best_model_mode: str | None = None,
    isotonic_time_weighting: bool = False,
    bagging_failure_data: bool = False,
    computing_weight_mode: str = "val_rmse",
    train_with_censored_data: bool = False,
    use_cotraining_ensemble_survival_loss_function: bool = False,
    cotraining_survival_loss_lambda: float = 1.0,
    # Others
    force_load_from_cache: bool = False,
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
        cache_dir, num_workers, pin_memory, return_sequence_label, batch_size, counter_mode,
        include_histograms, histogram_mode:
            ``ScaniaDataModule`` construction params.
        calib_rate: Optional fraction (sibling of ``val_rate``/``test_rate``) of vehicles set
            aside as a dedicated calibration split. When ``> 0``, the ``crepes`` conformal
            regressors are calibrated on this held-out set instead of ``val_data`` (which is
            also used for early stopping), avoiding the anti-conservative interval bias from
            calibrating on the same data used for model selection. ``0.0`` (default) keeps the
            legacy behavior of calibrating on ``val_data``.
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
        use_fine_tuning: When ``True``, receivers are warm-start fine-tuned each iteration instead
            of retrained from scratch. ``False`` (default) keeps from-scratch retraining.
        fine_tune_lr_factor: LR multiplier for a fine-tune (only used when ``use_fine_tuning``).
        fine_tune_max_epochs: Max epochs per fine-tune (only used when ``use_fine_tuning``).
        fine_tune_patience: ``EarlyStopping`` patience per fine-tune (only used when
            ``use_fine_tuning``).
        fine_tune_from_initial_model: When ``True``, every iteration's fine-tune warm-starts
            from the model produced by Initial training instead of the previous iteration's
            model. Only meaningful (and only allowed) combined with ``use_fine_tuning=True``.
            ``False`` (default) keeps warm-starting from the previous iteration's model.
        peer_weighted_pseudo_label: When ``True``, a unit's pseudo-label is the
            ``1/confidence_score**2``-weighted average of all peers' predictions (each peer's
            own per-unit conformal-interval confidence score) instead of the single
            most-confident peer's. ``False`` (default) keeps the single-peer label.
        keep_best_model_mode: Controls whether each iteration's candidate is accepted or reverted.
            ``None`` (default) always accepts the candidate. ``"val_rmse"`` keeps it only if
            validation RMSE improves on that model's best so far. ``"delta_criterion"`` keeps it
            only if ``delta = MSE(h_j, L) - MSE(h'_j, L) > 0`` on the pre-iteration labelled set
            (same pattern as ``Coprog``'s confidence measure). A rejection reverts the
            model/dataset and permanently drops the iteration's added units.
        isotonic_time_weighting: When ``True``, the monotone projection is fitted with per-window
            ``sample_weight`` proportional to the local time gap ``Delta t``. Requires
            ``use_monotone_projection=True``; the entry point fetches the per-window
            ``suspension_time_steps`` from the data module and passes them to ``train``. ``False``
            (default) uses the unweighted projection.
        bagging_failure_data: When ``True``, each model's initial (pre co-training) failure
            dataset is an independent bootstrap resample (with replacement) of the failure data,
            instead of every model sharing the identical failure dataset. ``False`` (default)
            keeps the legacy shared-dataset behavior.
        computing_weight_mode: Selects how the final per-model ensemble weights are derived.
            ``"val_rmse"`` (default) weights models inversely to their validation RMSE.
            ``"confidence"`` wraps each model in a calibrated conformal regressor and weights
            models by their average ``predict_p`` confidence on the validation set.
        train_with_censored_data: When ``True``, Initial training (only) additionally trains on
            the censored data using ``BasicLightningModule.survival_loss_function`` instead of
            plain MSE. Sequential-only (like every other opt-in lever here). ``False`` (default)
            keeps Initial training on failure data with plain MSE.
        use_cotraining_ensemble_survival_loss_function: When ``True``, every iteration that
            trains on real peer-assigned pseudo-labels -- a fine-tune call (see
            ``use_fine_tuning``) or a from-scratch retrain -- trains with
            ``BasicLightningModule.cotraining_ensemble_survival_loss_function`` instead of plain
            MSE. Independent of ``use_fine_tuning``. ``False`` (default) keeps plain MSE.
        cotraining_survival_loss_lambda: Weight of the pseudo-label MSE term in
            ``cotraining_ensemble_survival_loss_function``. Only used when
            ``use_cotraining_ensemble_survival_loss_function`` is ``True``.
        force_load_from_cache: When ``True``, the data module reads the splits straight out of
            ``cache_dir`` and adopts every split-defining param from its ``manifest.json``,
            never re-preprocessing and never overwriting the cache. Used to pin every
            benchmarked model to one identical set of vehicles.
        gpu_ids: GPU id(s). ``None`` → single GPU / auto (sequential); ``[g]`` → pinned; two or
            more → parallel training across those GPUs.
        datetime_for_folders: Timestamp used to name the output folders.

    Returns:
        ``(rmse_weighted, score_weighted)`` of the weighted-ensemble test prediction.
    """
    set_seed(seed)

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
        "force_load_from_cache": force_load_from_cache,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    scania_data_module = ScaniaDataModule(**dataset_kwargs)
    scania_data_module.setup()

    feature_num = len(scania_data_module.feature_cols)

    features_uncensored, targets_uncensored, features_censored, ids_censored = \
        scania_data_module.get_cotraining_tensors("train")
    # Per-window survival lower bounds for the censored data (row-aligned with the censored
    # features/ids above). Always fetched: besides clipping monotone-projected pseudo-labels
    # up to the observed time-to-study-end (when use_monotone_projection is on), the ensemble
    # always uses these bounds as a hard physical-validity filter on pseudo-label selection,
    # independent of use_monotone_projection.
    _, _, suspension_lower_bounds = scania_data_module.get_censored_lower_bounds("train")
    # Per-window operational time steps for the censored data (row-aligned with the censored
    # features/ids above), used by the Delta-t-weighted isotonic projection. Fetched only when
    # that feature is on.
    suspension_time_steps = None
    if isotonic_time_weighting:
        _, _, suspension_time_steps = scania_data_module.get_censored_time_steps("train")
    # Labelled (uncensored) validation data: early stopping / best-checkpoint selection and
    # the ensemble weights (instead of the test set). Also the fallback conformal calibration
    # set when calib_rate == 0 (no dedicated calib split configured).
    val_features, val_targets, _, _ = scania_data_module.get_cotraining_tensors("val")
    test_features, test_targets, _, _ = scania_data_module.get_cotraining_tensors("test")
    # Dedicated calibration split (decoupled from the early-stopping val set) — only fetched
    # when calib_rate > 0; ensemble.train() falls back to val_data/val_label otherwise.
    calib_features, calib_targets = None, None
    if scania_data_module.calib_rate > 0:
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
        "inference_batch_size": inference_batch_size,
        "use_monotone_projection": use_monotone_projection,
        "monotone_residual_weight": monotone_residual_weight,
        "use_fine_tuning": use_fine_tuning,
        "fine_tune_lr_factor": fine_tune_lr_factor,
        "fine_tune_max_epochs": fine_tune_max_epochs,
        "fine_tune_patience": fine_tune_patience,
        "fine_tune_from_initial_model": fine_tune_from_initial_model,
        "peer_weighted_pseudo_label": peer_weighted_pseudo_label,
        "keep_best_model_mode": keep_best_model_mode,
        "isotonic_time_weighting": isotonic_time_weighting,
        "bagging_failure_data": bagging_failure_data,
        "computing_weight_mode": computing_weight_mode,
        "train_with_censored_data": train_with_censored_data,
        "use_cotraining_ensemble_survival_loss_function": use_cotraining_ensemble_survival_loss_function,
        "cotraining_survival_loss_lambda": cotraining_survival_loss_lambda,
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
        use_fine_tuning=use_fine_tuning,
        fine_tune_lr_factor=fine_tune_lr_factor,
        fine_tune_max_epochs=fine_tune_max_epochs,
        fine_tune_patience=fine_tune_patience,
        fine_tune_from_initial_model=fine_tune_from_initial_model,
        peer_weighted_pseudo_label=peer_weighted_pseudo_label,
        keep_best_model_mode=keep_best_model_mode,
        isotonic_time_weighting=isotonic_time_weighting,
        bagging_failure_data=bagging_failure_data,
        computing_weight_mode=computing_weight_mode,
        use_cotraining_ensemble_survival_loss_function=use_cotraining_ensemble_survival_loss_function,
        cotraining_survival_loss_lambda=cotraining_survival_loss_lambda,
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
        train_with_censored_data=train_with_censored_data,
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        suspension_lower_bounds=suspension_lower_bounds,
        suspension_time_steps=suspension_time_steps,
        iterations=iterations,
        suspension_pool_size=suspension_pool_size,
        add_ratio=add_ratio,
        val_data=val_features,
        val_label=val_targets,
        calib_data=calib_features,
        calib_label=calib_targets,
        # Per-stage metrics: the score columns use the class-based Scania cost
        # (scania.metrics.scania_score), while the reported weights use RMSE + "min"
        # (matching calculate_weights below) -- the cost is reported, never used for selection.
        test_data=test_features,
        test_label=test_targets,
        score_callback=_score_callback_for_coprog,
        weight_callback=_criteria_callback_for_coprog,
        weight_mode="min",
        metrics_file=f"{results_path}/{model_version.value}-per-stage-scania.csv",
        log_file=log_file_path,
        pool_seed=seed,
    )

    ensemble.calculate_weights(
        x_test=val_features,
        target=val_targets,
        criteria_callback=_criteria_callback_for_coprog,
        mode="min",
        calib_data=calib_features,
        calib_label=calib_targets,
    )

    return save_ensemble_outputs(
        ensemble=ensemble,
        model_version=model_version,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        test_features=test_features,
        test_targets=test_targets,
        version_strs=meta["version_strs"],
        model_specs=meta["model_specs"],
        training_time_seconds=ensemble.training_duration_seconds,
        avg_iteration_time_seconds=ensemble.average_iteration_duration_seconds,
        data_module=scania_data_module,
    )
