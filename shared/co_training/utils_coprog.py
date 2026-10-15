import functools
import time
from datetime import datetime
from typing import Any

import pandas as pd
import torch

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from challenge import run_challenge_evaluation
from co_training.CoTrainingData import CoTrainingData
from constants import necessary_keys_cmapss
from constants import necessary_keys_scania
from constants.necessary_shared_keys import VALIDATION_RATE, SEQUENCE_LEN, BATCH_SIZE, SEED
from models import Coprog
from scania.dataset import ScaniaDataModule
from shared.enums import (
    ModelVersion,
    DatasetVersion
)
from shared.utils import (
    creating_model,
    build_scania_module,
    criteria_rmse_callback,
    criteria_score_callback,
    save_train_parameters,
    get_or_train_initial_models,
    generate_and_save_model_prediction,
    set_seed,
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
)
from constants import necessary_shared_keys as shared_keys


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
        # Others
        force_load_from_cache: bool = False,  # scania specific
        gpu_ids: list[int] | None = None,
        datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        pretrained_model_dir: str | None = None,
        sub_dataset: str | None = None,
        percent_of_broken_data: float | None = None,
        percent_of_censored_data: float | None = None,
) -> tuple[float, float]:
    seed = dataset_params[SEED]

    set_seed(seed)

    first_model = models_params[shared_keys.COPROG_FIRST_MODEL]
    second_model = models_params[shared_keys.COPROG_SECOND_MODEL]

    lr = training_params[shared_keys.LEARNING_RATE]
    patiences = training_params[shared_keys.PATIENCES]
    max_epochs = training_params[shared_keys.MAX_EPOCHS]
    coprog_iterations = training_params[shared_keys.COPROG_ITERATIONS]
    coprog_suspension_pool_size = training_params[shared_keys.COPROG_SUSPENSION_POOL_SIZE]

    rul_target_standardization = [False, False]
    if "rul_target_standardization" in training_params:
        rul_target_standardization = training_params["rul_target_standardization"]

    sequence_len = dataset_params[SEQUENCE_LEN]
    batch_size = dataset_params[BATCH_SIZE]

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

    if len(lr) != 2:
        raise ValueError(f"lr should have 2 values instead of {len(lr)}")

    if len(patiences) != 2:
        raise ValueError(f"patience should have 2 values instead of {len(patiences)}")

    if len(max_epochs) != 2:
        raise ValueError(f"max_epochs should have 2 values instead of {len(max_epochs)}")

    if VALIDATION_RATE not in dataset_params and dataset_params[VALIDATION_RATE] <= 0:
        raise ValueError(
            "Coprog needs a validation set for early stopping / best-model selection and for the "
            "ensemble weights. Set validation_rate > 0 in the config."
        )

    print("Loading datasets with the following parameters :")
    print(dataset_params)

    coprog_dataset: CoTrainingData = _get_coprog_dataset_for_training(
        dataset_version=dataset_version,
        dataset_params=dataset_params,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        force_load_from_cache=force_load_from_cache,
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
    )

    first_model_params, first_model_version = _extract_model_coprog_params(dataset_version, first_model)
    second_model_params, second_model_version = _extract_model_coprog_params(dataset_version, second_model)

    targets_means = []
    targets_stds = []

    for i in range(2):
        if rul_target_standardization[i]:
            # Standardize the RUL target: the network trains/predicts in normalized
            # target space (see BasicLightningModule) so the MSE gradient is not
            # dominated by the raw target magnitude. Stats are computed on the training
            # (uncensored) window labels only, via the public train dataloader, to avoid
            # leakage and without touching the dataset classes. BasicLightningModule
            # de-normalizes predictions back to real RUL units for all metrics/outputs.
            target_mean = float(coprog_dataset.targets_uncensored.mean())
            target_std = float(coprog_dataset.targets_uncensored.std())
            if target_std < 1e-6:
                target_std = 1.0
            print(f"RUL target standardization for model {i + 1}: mean={target_mean:.4f} std={target_std:.4f}")
        else:
            target_mean = 0.0
            target_std = 1.0
        targets_means.append(target_mean)
        targets_stds.append(target_std)

    training_kwargs = {
        "lr": lr,
        "patiences": patiences,
        "max_epochs": max_epochs,
        "coprog_iterations": coprog_iterations,
        "coprog_suspension_pool_size": coprog_suspension_pool_size,
        "rul_target_standardization": rul_target_standardization,
        "target_mean": targets_means,
        "target_std": targets_stds,
    }

    model_kwargs = {
        "first_model": first_model,
        "second_model": second_model,
    }

    save_train_parameters(
        results_path=final_results_path,
        dataset_parameters=dataset_params,
        training_parameters=training_kwargs,
        model_parameters=model_kwargs,
    )

    print(f"Creating first model ({first_model_version.value})...")
    # dict(...) copies: _creating_model mutates its params dict via .update, so we keep the
    # extracted params pristine for the (picklable) builders used below.
    first_model_module = creating_model(dict(first_model_params), first_model_version, coprog_dataset.feature_num,
                                        sequence_len)

    print(f"Creating second model ({second_model_version.value})...")
    second_model_module = creating_model(dict(second_model_params), second_model_version, coprog_dataset.feature_num,
                                         sequence_len)

    coprog = Coprog(
        first_model=first_model_module,
        second_model=second_model_module,
        dataset_version=dataset_version,
        verbose=1,
    )

    # Builder-style setup: Coprog rebuilds a fresh module (and its Trainer) for every
    # from-scratch training. The builders MUST be picklable (module-level function +
    # functools.partial, no closures) so they survive the process boundary when training
    # is distributed across GPUs. Coprog snapshots the initial weights once so every
    # training starts from identical weights.
    module_builders = [
        functools.partial(
            build_scania_module,
            model_params=dict(first_model_params),
            model_version_value=first_model_version.value,
            feature_num=coprog_dataset.feature_num,
            sequence_len=sequence_len,
            lr=lr[0],
            target_mean=targets_means[0],
            target_std=targets_stds[0],
        ),
        functools.partial(
            build_scania_module,
            model_params=dict(second_model_params),
            model_version_value=second_model_version.value,
            feature_num=coprog_dataset.feature_num,
            sequence_len=sequence_len,
            lr=lr[1],
            target_mean=targets_means[1],
            target_std=targets_stds[1],
        ),
    ]

    # gpu_ids (from the --gpu-ids CLI option): None -> single GPU / auto; [g] -> pinned to
    # GPU g; [g0, g1, ...] -> train the two models in parallel on separate GPUs.
    print(f"COPROG GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU)'}")

    coprog.setup_training_builder(
        module_builders=module_builders,
        max_epochs=max_epochs,
        patiences=patiences,
        batch_sizes=[batch_size, batch_size],
        shuffle_dataloaders=[True, True],
        gpu_ids=gpu_ids,
    )

    print("Training Coprog model...")
    training_start = time.perf_counter()

    pretrained_models = None
    if pretrained_model_dir is not None:
        print("Loading pre-trained model...")
        number_of_models = 2
        version_strs = []
        for model_name in first_model.keys():
            version_strs.append(model_name)
        for model_name in second_model.keys():
            version_strs.append(model_name)

        pretrained_models = get_or_train_initial_models(
            initial_models_dir=pretrained_model_dir,
            module_builders=module_builders,
            batch_size=[batch_size] * number_of_models,
            number_of_models=number_of_models,
            max_epochs=max_epochs,
            patiences=patiences,
            shuffle_dataloaders=[True] * number_of_models,
            version_strs=version_strs,
            failure_data=coprog_dataset.features_uncensored,
            failure_label=coprog_dataset.targets_uncensored,
            val_data=coprog_dataset.val_features,
            val_label=coprog_dataset.val_targets,
            seed=seed,
            run_log_path=None,
        )

    coprog.train(
        failure_data=coprog_dataset.features_uncensored,
        failure_label=coprog_dataset.targets_uncensored,
        suspension_data=coprog_dataset.features_censored,
        suspension_ids=coprog_dataset.ids_censored,
        iterations=coprog_iterations,
        suspension_pool_size=coprog_suspension_pool_size,
        val_data=coprog_dataset.val_features,
        val_label=coprog_dataset.val_targets,
        # Per-stage metrics tracking (initial / iteration_k / final). The score columns use
        # the class-based cost; the reported weights use
        # RMSE + "min", matching calculate_weights
        # below. Runs in the main process, so it is safe for the parallel training path too.
        test_data=coprog_dataset.test_features,
        test_label=coprog_dataset.test_targets,
        score_callback=criteria_score_callback,
        weight_callback=criteria_rmse_callback,
        weight_mode="min",
        metrics_file=f"{final_results_path}/{model_version.value}-per-stage-{dataset_version.value}.csv",
        pretrained_models=pretrained_models,
    )

    training_time_seconds = time.perf_counter() - training_start
    print(f"Coprog trained in {training_time_seconds:.1f}s")

    # Ensemble weights are computed on the validation set, not the test set,
    # to avoid leaking test information into the weighting.
    coprog.calculate_weights(
        x_test=coprog_dataset.val_features,
        target=coprog_dataset.val_targets,
        criteria_callback=criteria_rmse_callback,
        mode="min",
    )

    print("Saving first and second trained models...")
    torch.save(coprog._h1, f"{final_checkpoints_path}/coprog_{first_model_version.value}.pth")
    torch.save(coprog._h2, f"{final_checkpoints_path}/coprog_{second_model_version.value}.pth")

    # Flatten both sides so we compute a real element-wise RMSE.
    y_hat = coprog.predict(coprog_dataset.test_features).detach().cpu().view(-1)
    targets_flat = coprog_dataset.test_targets.detach().cpu().view(-1)

    rmse_weighted, score_weighted = generate_and_save_model_prediction(
        predictions=y_hat,
        targets=targets_flat,
        model_version=model_version,
        prediction_type="test_weighted",
        sub_dataset=sub_dataset,
        dataset_version=dataset_version,
        results_path=final_results_path,
        score_callback=criteria_score_callback,
    )

    print(f"Test RMSE: {rmse_weighted}")
    print(f"Score: {score_weighted}")

    predictions_first_model = coprog.prediction_for_first_model(coprog_dataset.test_features)
    rmse_h1, score_h1 = generate_and_save_model_prediction(
        predictions=predictions_first_model,
        targets=targets_flat,
        model_version=model_version,
        prediction_type="test_h1",
        sub_dataset=sub_dataset,
        dataset_version=dataset_version,
        results_path=final_results_path,
        score_callback=criteria_score_callback,
    )

    predictions_second_model = coprog.prediction_for_second_model(coprog_dataset.test_features)
    rmse_h2, score_h2 = generate_and_save_model_prediction(
        predictions=predictions_second_model,
        targets=targets_flat,
        model_version=model_version,
        prediction_type="test_h2",
        sub_dataset=sub_dataset,
        dataset_version=dataset_version,
        results_path=final_results_path,
        score_callback=criteria_score_callback,
    )

    scores = pd.DataFrame(columns=[
        "test_rmse_h1",
        "test_score_h1",
        "test_rmse_h2",
        "test_score_h2",
        "test_rmse_weighted",
        "test_score_weighted",
        "weight_h1",
        "weight_h2",
        "training_time_seconds",
        "avg_iteration_time_seconds",
    ])

    scores.loc[0] = [
        rmse_h1,
        score_h1,
        rmse_h2,
        score_h2,
        rmse_weighted,
        score_weighted,
        coprog.w1,
        coprog.w2,
        training_time_seconds,
        coprog.average_iteration_duration_seconds,
    ]

    match dataset_version:
        case DatasetVersion.CMAPSS:
            scores_path = f"{final_results_path}/{model_version.value}-{dataset_version.value}-{sub_dataset}.csv"
        case DatasetVersion.SCANIA:
            scores_path = f"{final_results_path}/{model_version.value}-{dataset_version.value}.csv"
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")

    scores.to_csv(scores_path, index=False)

    if dataset_version == DatasetVersion.SCANIA and coprog_dataset.scania_data_module is not None:
        # Additionally score every predictor on the official Scania Component X held-out sets (test
        # and validation). Never raises: the run's own results are already written by the time this
        # runs.
        challenge_predict_fn = lambda features: {
            "weighted": coprog.predict(features),
            "h1": coprog.prediction_for_first_model(features),
            "h2": coprog.prediction_for_second_model(features),
        }
        run_challenge_evaluation(
            data_module=coprog_dataset.scania_data_module,
            predict_fn=challenge_predict_fn,
            model_version=model_version.value,
            results_path=final_results_path,
        )
        run_challenge_evaluation(
            data_module=coprog_dataset.scania_data_module,
            predict_fn=challenge_predict_fn,
            model_version=model_version.value,
            results_path=final_results_path,
            split="validation",
        )

    return rmse_weighted, score_weighted


def _get_coprog_dataset_for_training(
        dataset_version: DatasetVersion,
        dataset_params: dict[str, Any],
        dataset_root: str,
        force_load_from_cache: bool,
        sub_dataset: str | None = None,
        percent_of_censored_data: float | None = None,
        percent_of_broken_data: float | None = None,
) -> CoTrainingData:
    """
    Get the CoTrainingData data for Coprog depending on the dataset version.
    Only Scania and CMAPSS are implemented for the moment.

    Args:
        dataset_version:the dataset version (SCANIA, CMAPSS)
        dataset_params: the params to get datasets
        dataset_root: the root where the files of the dataset are stored

    Returns:
        All information for Coprog stored inside CoTrainingData
    """
    scania_data_module = None

    match dataset_version:
        case DatasetVersion.CMAPSS:
            dataset_params.pop(BATCH_SIZE)

            assert sub_dataset is not None, f"sub_dataset for CMAPSS can't be None"
            assert percent_of_censored_data is not None, f"percent_of_censored_data for CMAPSS can't be None"
            assert percent_of_broken_data is not None, f"percent_of_broken_data for CMAPSS can't be None"

            train_dataset, test_dataset, valid_dataset, _ = CMAPSSLoader.get_datasets(
                dataset_root=dataset_root,
                sub_dataset=sub_dataset,
                percent_of_censored_data=percent_of_censored_data,
                percent_of_broken_data=percent_of_broken_data,
                **dataset_params,
            )

            feature_num = len(train_dataset.feature_cols)

            features_uncensored, targets_uncensored, features_censored, ids_censored = train_dataset.get_censored_split_tensors()
            val_features, val_targets, _, _ = valid_dataset.get_censored_split_tensors()
            features_test, targets_test = test_dataset.get_features_targets()

        case DatasetVersion.SCANIA:
            scania_data_module = ScaniaDataModule(
                data_dir=dataset_root,
                force_load_from_cache=force_load_from_cache,
                **dataset_params,
            )

            scania_data_module.setup()

            feature_num = len(scania_data_module.feature_cols)

            features_uncensored, targets_uncensored, features_censored, ids_censored = scania_data_module.get_cotraining_tensors(
                "train")
            # Labelled (uncensored) validation data: used both for early stopping / best-checkpoint
            # selection during training and to compute the ensemble weights (instead of the test set).
            val_features, val_targets, _, _ = scania_data_module.get_cotraining_tensors("val")
            features_test, targets_test, _, _ = scania_data_module.get_cotraining_tensors("test")
        case _:
            raise ValueError(f"{dataset_version} is not a valid dataset version")

    return CoTrainingData(
        feature_num=feature_num,
        features_uncensored=features_uncensored,
        targets_uncensored=targets_uncensored,
        features_censored=features_censored,
        ids_censored=ids_censored,
        test_features=features_test,
        test_targets=targets_test,
        val_features=val_features,
        val_targets=val_targets,
        scania_data_module=scania_data_module,
    )


def _extract_model_coprog_params(
        dataset_version: DatasetVersion,
        model_params: dict,
) -> tuple[dict, ModelVersion]:
    """Validate and extract single submodel config and its ModelVersion.

    :param model_params: Single-entry dictionary mapping model name to its config dict.
    :return: Tuple of (inner_params_dict, ModelVersion).
    """
    if len(model_params.keys()) != 1:
        raise ValueError("'model_params' must contain exactly 1 key")

    model_key = list(model_params.keys())[0]
    try:
        model_version = ModelVersion(model_key)
    except ValueError:
        raise ValueError(f"{model_key} in 'model_params' is not a valid model version")

    if dataset_version == DatasetVersion.CMAPSS:
        necessary_keys_dataset = necessary_keys_cmapss
    else:
        necessary_keys_dataset = necessary_keys_scania

    match model_version:
        case ModelVersion.CNN:
            necessary_keys = necessary_keys_dataset.NECESSARY_CNN_KEYS
        case ModelVersion.LSTM:
            necessary_keys = necessary_keys_dataset.NECESSARY_LSTM_KEYS
        case ModelVersion.TRANSFORMER_FEATURES:
            necessary_keys = necessary_keys_dataset.NECESSARY_TRANSFORMER_FEATURES_KEYS
        case _:
            raise ValueError(f"{model_version.value} in 'model_params' is not a valid model version for COPROG")

    for key in necessary_keys:
        if key not in model_params[model_key]:
            raise KeyError(f"{key} is missing in 'model_params' for {model_version.value}")

    return model_params[model_key], model_version


def _cmapss_score(predict: torch.Tensor, label: torch.Tensor) -> float:
    """Compute the asymmetric C-MAPSS scoring metric on PyTorch tensors.

    :param predict: Predicted Remaining Useful Life tensor.
    :param label: True Remaining Useful Life tensor.
    :return: The asymmetric C-MAPSS score as a Python float.
    """
    a1 = 13
    a2 = 10
    error = predict.view(-1) - label.view(-1)
    pos_e = torch.exp(-error[error < 0] / a1) - 1
    neg_e = torch.exp(error[error >= 0] / a2) - 1
    return torch.sum(pos_e).item() + torch.sum(neg_e).item()
