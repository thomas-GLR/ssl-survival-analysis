import functools
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from constants import necessary_keys_scania
from models import CNN1D, Simple_LSTM, TransformerFeatures, TransformerTimeSequence
from models import Coprog
from scania.dataset import ScaniaDataModule
from scania.lightning_module import BasicLightningModule
from scania.utils.utils_scania import (
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
    save_train_parameters,
    generate_and_save_model_prediction,
    _scania_score,
)
from shared.utils import ModelVersion


def train_model(
    checkpoints_path: str,
    results_path: str,
    model_version: ModelVersion,
    # Model params
    first_model: dict,
    second_model: dict,
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
    lr: list[float],
    patiences: list[int],
    max_epochs: list[int],
    coprog_iterations: int,
    coprog_suspension_pool_size: int,
    rul_target_standardization: list[bool],
    # Others
    gpu_ids: list[int] | None = None,
    datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
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

    if len(lr) != 2:
        raise ValueError(f"lr should have 2 values instead of {len(lr)}")

    if len(patiences) != 2:
        raise ValueError(f"patience should have 2 values instead of {len(patiences)}")

    if len(max_epochs) != 2:
        raise ValueError(f"max_epochs should have 2 values instead of {len(max_epochs)}")

    if val_rate <= 0:
        raise ValueError(
            "Coprog needs a validation set for early stopping / best-model selection and for the "
            "ensemble weights. Set validation_rate > 0 in the config."
        )

    dataset_kwargs = {
        'data_dir': dataset_root,
        'seed': seed,
        'data_fraction': data_fraction,
        'val_rate': val_rate,
        'test_rate': test_rate,
        'stratify': stratify,
        'norm_type': norm_type,
        'shuffle_loader': shuffle_loader,
        'cache_dir': cache_dir,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'return_sequence_label': return_sequence_label,
        'batch_size': batch_size,
        'sequence_len': sequence_len,
        'counter_mode': counter_mode,
        'include_histograms': include_histograms,
        'histogram_mode': histogram_mode,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    scania_data_module = ScaniaDataModule(
        data_dir=dataset_root,
        seed=seed,
        data_fraction=data_fraction,
        val_rate=val_rate,
        test_rate=test_rate,
        stratify=stratify,
        norm_type=norm_type,
        shuffle_loader=shuffle_loader,
        cache_dir=cache_dir,
        num_workers=num_workers,
        pin_memory=pin_memory,
        return_sequence_label=return_sequence_label,
        batch_size=batch_size,
        sequence_len=sequence_len,
        counter_mode=counter_mode,
        include_histograms=include_histograms,
        histogram_mode=histogram_mode,
    )

    scania_data_module.setup()

    feature_num = len(scania_data_module.feature_cols)

    features_uncensored, targets_uncensored, features_censored, ids_censored = scania_data_module.get_cotraining_tensors("train")
    # Labelled (uncensored) validation data: used both for early stopping / best-checkpoint
    # selection during training and to compute the ensemble weights (instead of the test set).
    val_features, val_targets, _, _ = scania_data_module.get_cotraining_tensors("val")
    test_features, test_targets, _, _ = scania_data_module.get_cotraining_tensors("test")

    first_model_params, first_model_version = _extract_model_coprog_params(first_model)
    second_model_params, second_model_version = _extract_model_coprog_params(second_model)

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
            target_mean = float(targets_uncensored.mean())
            target_std = float(targets_uncensored.std())
            if target_std < 1e-6:
                target_std = 1.0
            print(f"RUL target standardization for model {i+1}: mean={target_mean:.4f} std={target_std:.4f}")
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
        results_path=results_path,
        dataset_parameters=dataset_kwargs,
        training_parameters=training_kwargs,
        model_parameters=model_kwargs,
    )

    print(f"Creating first model ({first_model_version.value})...")

    # dict(...) copies: _creating_model mutates its params dict via .update, so we keep the
    # extracted params pristine for the (picklable) builders used below.
    first_model = _creating_model(dict(first_model_params), first_model_version, feature_num, sequence_len)

    print(f"Creating second model ({second_model_version.value})...")

    second_model = _creating_model(dict(second_model_params), second_model_version, feature_num, sequence_len)

    coprog = Coprog(
        first_model=first_model,
        second_model=second_model,
        verbose=1,
    )

    # Builder-style setup: Coprog rebuilds a fresh module (and its Trainer) for every
    # from-scratch training. The builders MUST be picklable (module-level function +
    # functools.partial, no closures) so they survive the process boundary when training
    # is distributed across GPUs. Coprog snapshots the initial weights once so every
    # training starts from identical weights.
    module_builders = [
        functools.partial(
            _build_scania_module,
            model_params=dict(first_model_params),
            model_version_value=first_model_version.value,
            feature_num=feature_num,
            sequence_len=sequence_len,
            lr=lr[0],
            target_mean=targets_means[0],
            target_std=targets_stds[0],
        ),
        functools.partial(
            _build_scania_module,
            model_params=dict(second_model_params),
            model_version_value=second_model_version.value,
            feature_num=feature_num,
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

    print(f"Training Coprog model...")

    coprog.train(
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        iterations=coprog_iterations,
        suspension_pool_size=coprog_suspension_pool_size,
        val_data=val_features,
        val_label=val_targets,
        # Per-stage metrics tracking (initial / iteration_k / final). The score columns use
        # the Scania score; the reported weights use RMSE + "min", matching calculate_weights
        # below. Runs in the main process, so it is safe for the parallel training path too.
        test_data=test_features,
        test_label=test_targets,
        score_callback=_score_callback_for_coprog,
        weight_callback=_criteria_callback_for_coprog,
        weight_mode="min",
        metrics_file=f"{results_path}/{model_version.value}-per-stage-scania.csv",
    )

    # Ensemble weights are computed on the validation set, not the test set,
    # to avoid leaking test information into the weighting.
    coprog.calculate_weights(
        x_test=val_features,
        target=val_targets,
        criteria_callback=_criteria_callback_for_coprog,
        mode="min",
    )

    print("Saving first and second trained models...")

    torch.save(coprog._h1, f"{checkpoints_path}/coprog_cnn.pth")
    torch.save(coprog._h2, f"{checkpoints_path}/coprog_lstm.pth")

    # Flatten both sides so we compute a real element-wise RMSE. targets_tensor is (N, 1)
    # and predict() returns (N,); subtracting them directly would broadcast to (N, N).
    y_hat = coprog.predict(test_features).detach().cpu().view(-1)
    targets_flat = test_targets.detach().cpu().view(-1)

    rmse_weighted, score_weighted = generate_and_save_model_prediction(
        predictions=y_hat,
        targets=targets_flat,
        model_version=model_version.value,
        prediction_type="test_weighted",
        results_path=results_path,
    )

    print(f"Test RMSE: {rmse_weighted}")
    print(f"Score: {score_weighted}")

    predictions_first_model = coprog.prediction_for_first_model(test_features)

    rmse_h1, score_h1 = generate_and_save_model_prediction(
        predictions=predictions_first_model,
        targets=targets_flat,
        model_version=model_version.value,
        prediction_type="test_h1",
        results_path=results_path,
    )

    predictions_second_model = coprog.prediction_for_second_model(test_features)

    rmse_h2, score_h2 = generate_and_save_model_prediction(
        predictions=predictions_second_model,
        targets=targets_flat,
        model_version=model_version.value,
        prediction_type="test_h2",
        results_path=results_path,
    )

    scores = pd.DataFrame(columns=[
        'test_rmse_h1',
        'test_score_h1',
        'test_rmse_h2',
        'test_score_h2',
        'test_rmse_weighted',
        'test_score_weighted',
        'weight_h1',
        'weight_h2'
    ])

    scores.loc[0] = [
        rmse_h1,
        score_h1,
        rmse_h2,
        score_h2,
        rmse_weighted,
        score_weighted,
        coprog.w1,
        coprog.w2
    ]

    # Save the results
    scores.to_csv(f'{results_path}/{model_version.value}-scania.csv', index=False)

    return rmse_weighted, score_weighted


def _criteria_callback_for_coprog(preds: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(preds, target)

    # Calculate Root Mean Squared Error
    rmse = torch.sqrt(mse)
    return rmse.item()


def _score_callback_for_coprog(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Scania score (a1=13, a2=10) for a prediction/target tensor pair.

    Wraps the numpy-based :func:`_scania_score` so it can be used as the ``score_callback``
    for Coprog's per-stage metrics (the ``test_score`` columns), matching the score reported
    by :func:`generate_and_save_model_prediction`.

    :param preds: Predicted RUL, shape (N,).
    :param target: True RUL, shape (N,).
    :return: The Scania score as a Python float.
    """
    return _scania_score(
        preds.detach().cpu().numpy().flatten(),
        target.detach().cpu().numpy().flatten(),
    )


def _extract_model_coprog_params(model_params: dict) -> tuple[dict, ModelVersion]:
    if len(model_params.keys()) != 1:
        raise ValueError(f"'model_params' must contain exactly 1 key'")

    for key in model_params.keys():
        try:
            model_key = key
            model_version = ModelVersion(key)
        except ValueError:
            raise ValueError(f"{key} in 'model_parms' is not a valid model version")

    match model_version:
        case ModelVersion.CNN:
            necessary_keys = necessary_keys_scania.NECESSARY_CNN_KEYS
        case ModelVersion.LSTM:
            necessary_keys = necessary_keys_scania.NECESSARY_LSTM_KEYS
        case ModelVersion.TRANSFORMER_FEATURES:
            necessary_keys = necessary_keys_scania.NECESSARY_TRANSFORMER_FEATURES_KEYS
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            necessary_keys = necessary_keys_scania.NECESSARY_TRANSFORMER_TIME_SEQUENCE_KEYS
        case _:
            raise ValueError(f"{model_version.value} in 'model_params' is not a valid model version for COPROG")

    for key in necessary_keys:
        if key not in model_params[model_key]:
            raise KeyError(f"{key} is missing in 'model_params' for {model_version.value}")

    return model_params[model_key], model_version


def _build_scania_module(
    model_params: dict,
    model_version_value: str,
    feature_num: int,
    sequence_len: int,
    lr: float,
    target_mean: float,
    target_std: float,
) -> LightningModule:
    """Build a fresh ``BasicLightningModule`` wrapping a fresh Scania model.

    This is a **module-level** function (not a closure) so that, wrapped in
    ``functools.partial``, it is picklable and can be sent to per-GPU worker processes for
    parallel COPROG training. It rebuilds the underlying ``nn.Module`` from its config via
    :func:`_creating_model` and wraps it in :class:`BasicLightningModule`.

    :param model_params: The model's constructor params (inner config dict). Copied before
        use because :func:`_creating_model` mutates it via ``.update``.
    :param model_version_value: The model version string (e.g. ``"cnn"``), converted back to
        a :class:`ModelVersion` (an enum is passed as its value to keep the partial picklable).
    :param feature_num: Number of input features.
    :param sequence_len: Input sequence length.
    :param lr: Learning rate for the wrapping module.
    :param target_mean: RUL target mean for standardization (0.0 disables it).
    :param target_std: RUL target std for standardization (1.0 disables it).
    :return: A fresh ``BasicLightningModule`` ready to train.
    """
    model = _creating_model(dict(model_params), ModelVersion(model_version_value), feature_num, sequence_len)
    return BasicLightningModule(
        lr=lr,
        model=model,
        target_mean=target_mean,
        target_std=target_std,
    )


def _creating_model(model_params: dict, model_version: ModelVersion, num_features: int, sequence_len: int) -> nn.Module:
    match model_version:
        case ModelVersion.CNN:
            return CNN1D(num_features=num_features)
        case ModelVersion.LSTM:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
            })
            return Simple_LSTM(**model_params)
        case ModelVersion.TRANSFORMER_FEATURES:
            model_params.update({
                "feature_num": num_features,
                "sequence_len": sequence_len,
            })
            return TransformerFeatures(**model_params)
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
            return TransformerTimeSequence(**model_params)
        case _:
            raise ValueError(f"{model_version.value} is not a valid model version for COPROG")
