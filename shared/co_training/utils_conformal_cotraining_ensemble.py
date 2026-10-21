import os
from datetime import datetime
from typing import Any

from shared.co_training import ConformalCoTrainingEnsembleData
from shared.co_training.utils_cotraining_common import parse_models_config, save_ensemble_outputs
from constants import necessary_shared_keys as shared_keys
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from shared.enums import ModelVersion, DatasetVersion
from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2
from scania.dataset import ScaniaDataModule
from shared.utils import (
    set_seed,
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
    save_train_parameters,
    get_or_train_initial_models,
    criteria_score_callback,
    criteria_rmse_callback,
)


def train_model(
    checkpoints_path: str,
    results_path: str,
    model_version: ModelVersion,
    dataset_version: DatasetVersion,
    # Dataset params
    dataset_params: dict[str, Any],
    dataset_root: str,
    # Model params
    models_params: list[dict],
    # Training params
    training_params: dict[str, Any],
    force_load_from_cache: bool,
    pretrained_model_dir: str | None,
    gpu_ids: list[int] | None = None,
    datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    sub_dataset: str | None = None,
    percent_of_broken_data: float | None = None,
    percent_of_censored_data: float | None = None,
) -> tuple[float, float]:
    seed = dataset_params[shared_keys.SEED]

    models_params = models_params[shared_keys.MODELS]

    set_seed(seed)

    training_kwargs = training_params.copy()

    sequence_len = dataset_params[shared_keys.SEQUENCE_LEN]
    batch_size = dataset_params[shared_keys.BATCH_SIZE]

    # Training params for conformal co-training ensemble
    isotonic_time_weighting = training_params[shared_keys.ISOTONIC_TIME_WEIGHTING]
    iterations = training_params.pop(shared_keys.ITERATIONS)
    suspension_pool_size = training_params.pop(shared_keys.SUSPENSION_POOL_SIZE)
    add_ratio = training_params.pop(shared_keys.ADD_RATIO)

    train_with_censored_data = False

    if shared_keys.TRAIN_WITH_CENSORED_DATA in training_params:
        train_with_censored_data = training_params.pop(shared_keys.TRAIN_WITH_CENSORED_DATA)

    if dataset_version == DatasetVersion.CMAPSS:
        assert sub_dataset is not None, "sub_dataset is required for CMAPSS"
        assert percent_of_censored_data is not None, "percent_of_censored_data is required for CMAPSS"
        assert percent_of_broken_data is not None, "percent_of_broken_data is required for CMAPSS"

    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )

    final_checkpoints_path, final_results_path = create_and_get_checkpoints_results_path(
        model_version=model_version,
        dataset_version=dataset_version,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        sub_dataset=sub_dataset,
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
    )

    if shared_keys.VALIDATION_RATE not in dataset_params and dataset_params[shared_keys.VALIDATION_RATE] <= 0:
        raise ValueError(
            "The co-training ensemble needs a validation set for early stopping / best-model "
            "selection, conformal calibration and the ensemble weights. Set validation_rate > 0."
        )

    print("Creating data loader with the following parameters :")
    print(dataset_params)

    conformal_cotraining_ensemble_data: ConformalCoTrainingEnsembleData = _get_cotraining_ensemble_dataset_for_training(
        dataset_version=dataset_version,
        dataset_params=dataset_params,
        dataset_root=dataset_root,
        isotonic_time_weighting=isotonic_time_weighting,
        sub_dataset=sub_dataset,
        force_load_from_cache=force_load_from_cache,
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
    )

    nn_modules, module_builders, meta = parse_models_config(
        models_cfg=models_params,
        feature_num=conformal_cotraining_ensemble_data.feature_num,
        sequence_len=sequence_len,
        targets_uncensored=conformal_cotraining_ensemble_data.targets_uncensored,
    )
    number_of_models = len(nn_modules)

    training_kwargs["lr"]=meta["lr"]
    training_kwargs["max_epochs"]=meta["max_epochs"]
    training_kwargs["patiences"]=meta["patiences"]

    save_train_parameters(
        results_path=final_results_path,
        dataset_parameters=dataset_params,
        training_parameters=training_kwargs,
        model_parameters={"models": models_params},
    )

    print(f"Creating conformal co-training ensemble with {number_of_models} models: {meta['version_strs']}")

    ensemble = CoTrainingEnsemble_v2(
        models=nn_modules,
        verbose=1,
        dataset_version=dataset_version,
        **training_params
    )

    print(f"Conformal co-training ensemble GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}")

    ensemble.setup_training_builder(
        module_builders=module_builders,
        max_epochs=meta["max_epochs"],
        patiences=meta["patiences"],
        batchs_size=[batch_size] * number_of_models,
        shuffle_dataloaders=[True] * number_of_models,
        gpu_ids=gpu_ids,
    )

    print("Training conformal co-training ensemble...")

    # Persistent run log next to the results. Created (truncated) here with a metadata
    # header, then handed to the ensemble which appends every log message under it
    # regardless of verbose (mirrors the C_MAPSS co-training entry point).
    log_file_path = os.path.join(final_results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Conformal co-training ensemble run ===\n")
        f.write(f"Datetime: {datetime_for_folders}\n")
        f.write(f"Model version: {model_version.value}\n")
        f.write(f"Models ({number_of_models}): {meta['version_strs']}\n")
        f.write(f"GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}\n")
        f.write("=====================================\n")

    pretrained_models = None

    if pretrained_model_dir is not None:
        print("Loading pre-trained model...")
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write("Loading pre-trained model...\n")

        pretrained_models = get_or_train_initial_models(
            initial_models_dir=pretrained_model_dir,
            module_builders=module_builders,
            batch_size=[batch_size] * number_of_models,
            number_of_models=number_of_models,
            max_epochs=meta["max_epochs"],
            patiences=meta["patiences"],
            shuffle_dataloaders=[True] * number_of_models,
            version_strs=meta["version_strs"],
            failure_data=conformal_cotraining_ensemble_data.features_uncensored,
            failure_label=conformal_cotraining_ensemble_data.targets_uncensored,
            val_data=conformal_cotraining_ensemble_data.val_features,
            val_label=conformal_cotraining_ensemble_data.val_targets,
            seed=seed,
            run_log_path=log_file_path,
        )

    ensemble.train(
        train_with_censored_data=train_with_censored_data,
        iterations=iterations,
        suspension_pool_size=suspension_pool_size,
        add_ratio=add_ratio,
        failure_data=conformal_cotraining_ensemble_data.features_uncensored,
        failure_label=conformal_cotraining_ensemble_data.targets_uncensored,
        suspension_data=conformal_cotraining_ensemble_data.features_censored,
        suspension_ids=conformal_cotraining_ensemble_data.ids_censored,
        val_data=conformal_cotraining_ensemble_data.val_features,
        val_label=conformal_cotraining_ensemble_data.val_targets,
        calib_data=conformal_cotraining_ensemble_data.calib_features,
        calib_label=conformal_cotraining_ensemble_data.calib_targets,
        suspension_lower_bounds=conformal_cotraining_ensemble_data.censored_lower_bounds,
        suspension_time_steps=conformal_cotraining_ensemble_data.censored_time_steps,
        test_data=conformal_cotraining_ensemble_data.test_features,
        test_label=conformal_cotraining_ensemble_data.test_targets,
        score_callback=criteria_score_callback,
        weight_callback=criteria_rmse_callback,
        weight_mode="min",
        metrics_file=f"{final_results_path}/{model_version.value}-per-stage-scania.csv",
        log_file=log_file_path,
        pool_seed=seed,
        pretrained_models=pretrained_models,
    )

    ensemble.calculate_weights(
        x_test=conformal_cotraining_ensemble_data.val_features,
        target=conformal_cotraining_ensemble_data.val_targets,
        criteria_callback=criteria_rmse_callback,
        mode="min",
        calib_data=conformal_cotraining_ensemble_data.calib_features,
        calib_label=conformal_cotraining_ensemble_data.calib_targets,
    )

    return save_ensemble_outputs(
        ensemble=ensemble,
        model_version=model_version,
        dataset_version=dataset_version,
        checkpoints_path=final_checkpoints_path,
        results_path=final_results_path,
        test_features=conformal_cotraining_ensemble_data.test_features,
        test_targets=conformal_cotraining_ensemble_data.test_targets,
        version_strs=meta["version_strs"],
        model_specs=meta["model_specs"],
        training_time_seconds=ensemble.training_duration_seconds,
        avg_iteration_time_seconds=ensemble.average_iteration_duration_seconds,
        data_module=conformal_cotraining_ensemble_data.scania_data_module,
    )


def _get_cotraining_ensemble_dataset_for_training(
    dataset_version: DatasetVersion,
    dataset_params: dict[str, Any],
    dataset_root: str,
    isotonic_time_weighting: bool,
    force_load_from_cache: bool,
    sub_dataset: str | None = None,
    percent_of_censored_data: float | None = None,
    percent_of_broken_data: float | None = None,
) -> ConformalCoTrainingEnsembleData:
    """
    Get the ConformalCoTrainingEnsembleData data for ConformalCoTrainingEnsemble depending on the dataset version.
    Only Scania and CMAPSS are implemented for the moment.

    Args:
        dataset_version:the dataset version (SCANIA, CMAPSS)
        dataset_params: the params to get datasets
        dataset_root: the root where the files of the dataset are stored

    Returns:
        All information for ConformalCoTrainingEnsemble stored inside ConformalCoTrainingEnsembleData
    """
    match dataset_version:
        case DatasetVersion.CMAPSS:
            # raise NotImplementedError(f"The code to get the dataset for {DatasetVersion.CMAPSS.value} is not yet implemented.")
            dataset_params.pop(shared_keys.BATCH_SIZE)

            assert sub_dataset is not None, f"sub_dataset for CMAPSS can't be None"
            assert percent_of_censored_data is not None, f"percent_of_censored_data for CMAPSS can't be None"
            assert percent_of_broken_data is not None, f"percent_of_broken_data for CMAPSS can't be None"

            train_dataset, test_dataset, valid_dataset, calib_dataset = CMAPSSLoader.get_datasets(
                dataset_root=dataset_root,
                sub_dataset=sub_dataset,
                percent_of_censored_data=percent_of_censored_data,
                percent_of_broken_data=percent_of_broken_data,
                **dataset_params,
            )

            feature_num = len(train_dataset.feature_cols)

            features_uncensored, targets_uncensored, features_censored, ids_censored = train_dataset.get_censored_split_tensors()
            val_features, val_targets, _, _ = valid_dataset.get_censored_split_tensors()if valid_dataset is not None else (None, None, None, None)
            test_features, test_targets = test_dataset.get_features_targets()
            calib_features, calib_targets, _, _ = calib_dataset.get_censored_split_tensors() if calib_dataset is not None else (None, None, None, None)

            _, _, censored_lower_bounds = train_dataset.get_censored_lower_bounds()

            censored_time_steps = None
            if isotonic_time_weighting:
                _, _, censored_time_steps = train_dataset.get_censored_time_steps()

            return ConformalCoTrainingEnsembleData(
                feature_num=feature_num,
                features_uncensored=features_uncensored,
                targets_uncensored=targets_uncensored,
                features_censored=features_censored,
                ids_censored=ids_censored,
                test_features=test_features,
                test_targets=test_targets,
                val_features=val_features,
                val_targets=val_targets,
                scania_data_module=None,
                censored_lower_bounds=censored_lower_bounds,
                censored_time_steps=censored_time_steps,
                calib_features=calib_features,
                calib_targets=calib_targets,
            )
        case DatasetVersion.SCANIA:
            scania_data_module = ScaniaDataModule(
                data_dir=dataset_root,
                force_load_from_cache=force_load_from_cache,
                **dataset_params,
            )
            scania_data_module.setup()

            feature_num = len(scania_data_module.feature_cols)

            features_uncensored, targets_uncensored, features_censored, ids_censored = \
                scania_data_module.get_cotraining_tensors("train")
            # Per-window survival lower bounds for the censored data (row-aligned with the censored
            # features/ids above). Always fetched: besides clipping monotone-projected pseudo-labels
            # up to the observed time-to-study-end (when use_monotone_projection is on), the ensemble
            # always uses these bounds as a hard physical-validity filter on pseudo-label selection,
            # independent of use_monotone_projection.
            _, _, censored_lower_bounds = scania_data_module.get_censored_lower_bounds("train")
            # Per-window operational time steps for the censored data (row-aligned with the censored
            # features/ids above), used by the Delta-t-weighted isotonic projection. Fetched only when
            # that feature is on.
            censored_time_steps = None
            if isotonic_time_weighting:
                _, _, censored_time_steps = scania_data_module.get_censored_time_steps("train")
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

            return ConformalCoTrainingEnsembleData(
                feature_num=feature_num,
                features_uncensored=features_uncensored,
                targets_uncensored=targets_uncensored,
                features_censored=features_censored,
                ids_censored=ids_censored,
                test_features=test_features,
                test_targets=test_targets,
                val_features=val_features,
                val_targets=val_targets,
                scania_data_module=scania_data_module,
                censored_lower_bounds=censored_lower_bounds,
                censored_time_steps=censored_time_steps,
                calib_features=calib_features,
                calib_targets=calib_targets,
            )
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")