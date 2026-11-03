import time
from datetime import datetime
from typing import Any

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch import callbacks

from models import TransformerEncoder_LSTM_1, Simple_LSTM, TransformerFeatures, RULCNN, RULMLP, BiLSTMAttention, CNN1D
from models.TransformerTimeSequence import TransformerTimeSequence
from scania.challenge import run_challenge_evaluation
from scania.dataset import ScaniaDataModule
from scania.lightning_module.BasicLightningModule import BasicLightningModule
from scania.utils.utils_scania import (
    generate_and_save_model_prediction,
)
from shared.utils import (
    set_seed,
    save_train_parameters,
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
)
from shared.enums import ModelVersion, DatasetVersion


def train_model(
        checkpoints_path: str,
        results_path: str,
        model_version: ModelVersion,
        dataset_version: DatasetVersion,
        dataset_params: dict[str, Any],
        dataset_root: str,
        models_params: dict[str, Any],
        training_params: dict[str, Any],
        force_load_from_cache: bool = False,
        pretrained_model_dir: str | None = None,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train a supervised Scania model using the runner's config dictionaries.

    ``pretrained_model_dir`` is accepted for the common runner interface; this
    trainer always fits a fresh model.
    """
    if dataset_version != DatasetVersion.SCANIA:
        raise ValueError("This trainer currently supports only the Scania dataset")

    sequence_len = dataset_params["sequence_len"]
    batch_size = dataset_params["batch_size"]
    lr = training_params["lr"]
    weight_decay = training_params["weight_decay"]
    patience = training_params["patience"]
    max_epochs = training_params["max_epochs"]
    rul_target_standardization = training_params["rul_target_standardization"]

    transformer_encoder_head_num = models_params.get("transformer_encoder_head_num")
    transformer_num_layer = models_params.get("transformer_num_layer")
    hidden_dim = models_params.get("hidden_dim")
    lstm_num_layers = models_params.get("lstm_num_layers")
    lstm_dropout = models_params.get("lstm_dropout")
    fc_layer_dim = models_params.get("fc_layer_dim")
    fc_dropout = models_params.get("fc_dropout")

    set_seed(dataset_params["seed"])

    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
    )

    checkpoints_path, results_path = create_and_get_checkpoints_results_path(
        model_version=model_version,
        dataset_version=dataset_version,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    scores = pd.DataFrame(
        columns=['train_rmse', 'val_rmse', 'test_rmse', 'test_score', 'training_time_seconds'])

    dataset_kwargs = {
        **dataset_params,
        "data_dir": dataset_root,
        "force_load_from_cache": force_load_from_cache,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    scania_data_module = ScaniaDataModule(**dataset_kwargs)

    scania_data_module.setup()

    feature_num = len(scania_data_module.feature_cols)

    if rul_target_standardization:
        # Standardize the RUL target: the network trains/predicts in normalized
        # target space (see BasicLightningModule) so the MSE gradient is not
        # dominated by the raw target magnitude. Stats are computed on the training
        # (uncensored) window labels only, via the public train dataloader, to avoid
        # leakage and without touching the dataset classes. BasicLightningModule
        # de-normalizes predictions back to real RUL units for all metrics/outputs.
        train_targets = torch.cat([y for _, y in scania_data_module.train_dataloader()])
        target_mean = float(train_targets.mean())
        target_std = float(train_targets.std())
        if target_std < 1e-6:
            target_std = 1.0
        print(f"RUL target standardization : mean={target_mean:.4f} std={target_std:.4f}")
    else:
        target_mean = 0.0
        target_std = 1.0

    print('Training model with the following parameters:')
    print(f"Sequence length : {sequence_len}")
    print(f"Patience : {patience}")

    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
            model_kwargs = {
                "sequence_len": sequence_len,
                "feature_num": feature_num,
                "transformer_encoder_head_num": transformer_encoder_head_num,
                "hidden_dim": hidden_dim,
                "lstm_num_layers": lstm_num_layers,
                "lstm_dropout": lstm_dropout,
                "fc_layer_dim": fc_layer_dim,
                "fc_dropout": fc_dropout,
            }

            model = TransformerEncoder_LSTM_1(**model_kwargs)
        case ModelVersion.LSTM:
            model_kwargs = {
                "sequence_len": sequence_len,
                "feature_num": feature_num,
                "transformer_encoder_head_num": transformer_encoder_head_num,
                "hidden_dim": hidden_dim,
                "lstm_num_layers": lstm_num_layers,
                "lstm_dropout": lstm_dropout,
                "fc_layer_dim": fc_layer_dim,
                "fc_dropout": fc_dropout,
            }

            model = Simple_LSTM(**model_kwargs)
        case ModelVersion.CNN:
            model_kwargs = {
                "num_features": feature_num,
            }

            model = CNN1D(**model_kwargs)
        case ModelVersion.RULCNN:
            model_kwargs = {
                "num_features": feature_num,
                "sequence_len": sequence_len,
            }

            model = RULCNN(**model_kwargs)
        case ModelVersion.BILSTM_ATTENTION:
            model_kwargs = {
                "num_features": feature_num,
                "sequence_len": sequence_len,
                "hidden_dim": models_params.get("hidden_dim", 128),
                "lstm_num_layers": models_params.get("lstm_num_layers", 4),
            }
            model = BiLSTMAttention(**model_kwargs)
        case ModelVersion.MLP:
            model_kwargs = {
                "num_features": feature_num,
                "sequence_len": sequence_len,
            }
            model = RULMLP(**model_kwargs)
        case ModelVersion.TRANSFORMER_FEATURES:
            model_kwargs = {
                "sequence_len": sequence_len,
                "feature_num": feature_num,
                "transformer_encoder_head_num": transformer_encoder_head_num,
                "transformer_num_layer": transformer_num_layer,
                "fc_layer_dim": fc_layer_dim,
                "fc_dropout": fc_dropout,
            }

            model = TransformerFeatures(**model_kwargs)
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            model_kwargs = {
                "sequence_len": sequence_len,
                "d_model": sequence_len,
                "feature_num": feature_num,
                "transformer_encoder_head_num": transformer_encoder_head_num,
                "transformer_num_layer": transformer_num_layer,
                "fc_layer_dim": fc_layer_dim,
                "fc_dropout": fc_dropout,
            }

            model = TransformerTimeSequence(**model_kwargs)
        case _:
            raise ValueError(f"Model version {model_version} is not supported")

    print(f"Models parameters : {model_kwargs}")

    training_kwargs = {
        **training_params,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "max_epochs": max_epochs,
        "rul_target_standardization": rul_target_standardization,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    print(f"Training parameters : {training_kwargs}")

    save_train_parameters(
        results_path=results_path,
        dataset_parameters=dataset_kwargs,
        training_parameters=training_kwargs,
        model_parameters=model_kwargs,
    )

    lightning_module = BasicLightningModule(
        lr=lr,
        model=model,
        weight_decay=weight_decay,
        target_mean=target_mean,
        target_std=target_std,
    )

    early_stop_callback = callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(
        default_root_dir=checkpoints_path,
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    training_start = time.perf_counter()

    trainer.fit(lightning_module, datamodule=scania_data_module)

    training_time_seconds = time.perf_counter() - training_start
    print(f"{model_version.value} trained in {training_time_seconds:.1f}s")

    callbacks_metrics = trainer.callback_metrics
    train_rmse = callbacks_metrics['train_rmse']
    val_rmse = callbacks_metrics['val_rmse']

    trainer.test(datamodule=scania_data_module, ckpt_path='best')

    callbacks_metrics = trainer.callback_metrics

    test_rmse = callbacks_metrics['test_rmse']
    test_score = callbacks_metrics['test_score']

    # Add the results to the dataframe
    scores.loc[0] = [train_rmse, val_rmse, test_rmse, test_score, training_time_seconds]

    # Save the results
    scores.to_csv(f'{results_path}/{model_version.value}-scania.csv', index=False)

    print(f"Scores from train and test :\n{scores}")

    # Save model predictions
    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
            model_for_reload = TransformerEncoder_LSTM_1(**model_kwargs)
        case ModelVersion.LSTM:
            model_for_reload = Simple_LSTM(**model_kwargs)
        case ModelVersion.CNN:
            model_for_reload = CNN1D(**model_kwargs)
        case ModelVersion.RULCNN:
            model_for_reload = RULCNN(**model_kwargs)
        case ModelVersion.BILSTM_ATTENTION:
            model_for_reload = BiLSTMAttention(**model_kwargs)
        case ModelVersion.MLP:
            model_for_reload = RULMLP(**model_kwargs)
        case ModelVersion.TRANSFORMER_FEATURES:
            model_for_reload = TransformerFeatures(**model_kwargs)
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            model_for_reload = TransformerTimeSequence(**model_kwargs)
        case _:
            raise ValueError(f"Model version {model_version} is not supported")

    # target_mean/target_std are restored from the checkpoint hyperparameters;
    # passed explicitly as belt-and-suspenders so de-normalization is correct.
    lightning_module_with_trained_model = BasicLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model=model_for_reload,
        target_mean=target_mean,
        target_std=target_std,
    )

    lightning_module_with_trained_model.eval()

    trainer = Trainer(
        accelerator="auto",
    )

    outputs = trainer.predict(lightning_module_with_trained_model, datamodule=scania_data_module)

    predictions = torch.cat([preds for preds, _ in outputs])
    targets = torch.cat([y for _, y in outputs])

    # Additionally score the trained model on the official Scania Component X held-out sets (test
    # and validation). Never raises: the run's own results are already written by the time this
    # runs.
    challenge_predict_fn = lambda features: {
        "test": _predict_challenge(lightning_module_with_trained_model, features, batch_size),
    }
    run_challenge_evaluation(
        data_module=scania_data_module,
        predict_fn=challenge_predict_fn,
        model_version=model_version.value,
        results_path=results_path,
    )
    run_challenge_evaluation(
        data_module=scania_data_module,
        predict_fn=challenge_predict_fn,
        model_version=model_version.value,
        results_path=results_path,
        split="validation",
    )

    return generate_and_save_model_prediction(
        predictions=predictions,
        targets=targets,
        model_version=model_version.value,
        prediction_type="test",
        results_path=results_path,
    )


def _predict_challenge(
        lightning_module: BasicLightningModule,
        features: torch.Tensor,
        batch_size: int,
) -> torch.Tensor:
    """Predict RUL for the challenge windows in batches, on the module's own device.

    ``BasicLightningModule.forward`` de-normalizes, so the returned values are already in real RUL
    units. Batched because the held-out set is one window per vehicle for ~5000 vehicles and the
    module may sit on a GPU with little headroom left after training.

    Args:
        lightning_module: The trained, ``eval()``-ed module.
        features: ``(N, sequence_len, n_features)`` challenge windows, on CPU.
        batch_size: Inference batch size.

    Returns:
        A ``(N,)`` CPU tensor of de-normalized RUL predictions.
    """
    device = next(lightning_module.parameters()).device
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            chunk = features[start:start + batch_size].to(device)
            predictions.append(lightning_module(chunk).detach().cpu())
    return torch.cat(predictions).view(-1)
