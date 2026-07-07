import shutil
import tempfile
from datetime import datetime
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch import callbacks
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
    generate_and_save_model_prediction
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
    # Training params
    lr: list[float],
    patiences: list[int],
    max_epochs: list[int],
    coprog_iterations: int,
    coprog_suspension_pool_size: int,
    # Others
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
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    scania_data_module = ScaniaDataModule(
        data_dir=dataset_root,
        seed=seed,
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

    training_kwargs = {
        "lr": lr,
        "patiences": patiences,
        "max_epochs": max_epochs,
        "coprog_iterations": coprog_iterations,
        "coprog_suspension_pool_size": coprog_suspension_pool_size
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

    first_model = _creating_model(first_model_params, first_model_version, feature_num, sequence_len)

    print(f"Creating second model ({second_model_version.value})...")

    second_model = _creating_model(second_model_params, second_model_version, feature_num, sequence_len)

    coprog = Coprog(
        first_model=first_model,
        second_model=second_model,
        verbose=1,
    )

    # Wrap each model in a Lightning module.
    first_module = BasicLightningModule(lr=lr[0], model=first_model)
    second_module = BasicLightningModule(lr=lr[1], model=second_model)

    # Each _train_fun call builds a fresh Trainer from these factories. The ModelCheckpoint
    # lets Coprog reload the best (val_loss) weights instead of the last-epoch ones, and
    # EarlyStopping avoids over/under-training. Checkpoints go to throwaway temp dirs that
    # we remove at the end (there is one fit per candidate, so this can be a lot of files).
    created_ckpt_dirs: list[str] = []

    def make_trainer_factory(max_epochs: int, patience: int) -> Callable[[], Trainer]:
        def factory() -> Trainer:
            ckpt_dir = tempfile.mkdtemp(prefix="coprog_ckpt_")
            created_ckpt_dirs.append(ckpt_dir)

            early_stop_callback = callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode='min',
            )
            checkpoint_callback = callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor='val_loss',
                filename='best-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min',
            )

            return Trainer(
                default_root_dir=ckpt_dir,
                accelerator="auto",
                max_epochs=max_epochs,
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

        return factory

    coprog.setup_training(
        lightning_modules=[first_module, second_module],
        trainer_factories=[
            make_trainer_factory(max_epochs=max_epochs[0], patience=patiences[0]),
            make_trainer_factory(max_epochs=max_epochs[1], patience=patiences[1]),
        ],
        batch_sizes=[batch_size, batch_size],
        shuffle_dataloaders=[True, True],
    )

    print(f"Training Coprog model...")

    try:
        coprog.train(
            failure_data=features_uncensored,
            failure_label=targets_uncensored,
            suspension_data=features_censored,
            suspension_ids=ids_censored,
            iterations=coprog_iterations,
            suspension_pool_size=coprog_suspension_pool_size,
            val_data=val_features,
            val_label=val_targets,
        )

        # Ensemble weights are computed on the validation set, not the test set,
        # to avoid leaking test information into the weighting.
        coprog.calculate_weights(
            x_test=val_features,
            target=val_targets,
            criteria_callback=_criteria_callback_for_coprog,
            mode="min",
        )
    finally:
        for ckpt_dir in created_ckpt_dirs:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

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
            return TransformerTimeSequence(**model_params)
        case _:
            raise ValueError(f"{model_version.value} is not a valid model version for COPROG")
