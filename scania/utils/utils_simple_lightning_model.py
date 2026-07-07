from datetime import datetime

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch import callbacks

from models import TransformerEncoder_LSTM_1, Simple_LSTM, CNN1D, TransformerFeatures
from models.TransformerTimeSequence import TransformerTimeSequence
from scania.dataset import ScaniaDataModule
from scania.lightning_module.BasicLightningModule import BasicLightningModule
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
        # Training
        lr: float,
        patience: int,
        max_epochs: int,
        # Model params
        transformer_encoder_head_num: int | None=None,
        transformer_num_layer: int | None=None,
        hidden_dim: int | None=None,
        lstm_num_layers: int | None=None,
        lstm_dropout: float | None=None,
        fc_layer_dim: int | None=None,
        fc_dropout: float | None=None,
        # Others
        datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    scores = pd.DataFrame(columns=['train_rmse', 'val_rmse', 'test_rmse', 'test_score'])

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
        case ModelVersion.TRANSFORMER_FEATURES:
            model_kwargs = {
                "sequence_len": sequence_len,
                "feature_num": feature_num,
                "transformer_encoder_head_num": transformer_encoder_head_num,
                "num_layers": transformer_num_layer,
                "fc_layer_dim": fc_layer_dim,
                "fc_dropout": fc_dropout,
            }

            model = TransformerFeatures(**model_kwargs)
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            model_kwargs = {
                "sequence_len": sequence_len,
                "d_model" : sequence_len,
                "feature_num": feature_num,
                "transformer_encoder_head_num": transformer_encoder_head_num,
                "num_layers": transformer_num_layer,
                "fc_layer_dim": fc_layer_dim,
                "fc_dropout": fc_dropout,
            }

            model = TransformerTimeSequence(**model_kwargs)
        case _:
            raise ValueError(f"Model version {model_version} is not supported")

    print(f"Models parameters : {model_kwargs}")

    training_kwargs = {
        "lr": lr,
        "patience": patience,
        "max_epochs": max_epochs,
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
        model=model
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

    trainer.fit(lightning_module, datamodule=scania_data_module)

    callbacks_metrics = trainer.callback_metrics
    train_rmse = callbacks_metrics['train_rmse']
    val_rmse = callbacks_metrics['val_rmse']

    trainer.test(datamodule=scania_data_module, ckpt_path='best')

    callbacks_metrics = trainer.callback_metrics

    test_rmse = callbacks_metrics['test_rmse']
    test_score = callbacks_metrics['test_score']

    # Add the results to the dataframe
    scores.loc[0] = [train_rmse, val_rmse, test_rmse, test_score]

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
        case ModelVersion.TRANSFORMER_FEATURES:
            model_for_reload = TransformerFeatures(**model_kwargs)
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            model_for_reload = TransformerTimeSequence(**model_kwargs)
        case _:
            raise ValueError(f"Model version {model_version} is not supported")

    lightning_module_with_trained_model = BasicLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model=model_for_reload,
    )

    lightning_module_with_trained_model.eval()

    trainer = Trainer(
        accelerator="auto",
    )

    outputs = trainer.predict(lightning_module_with_trained_model, datamodule=scania_data_module)

    predictions = torch.cat([preds for preds, _ in outputs])
    targets = torch.cat([y for _, y in outputs])

    return generate_and_save_model_prediction(
        predictions=predictions,
        targets=targets,
        model_version=model_version.value,
        prediction_type="test",
        results_path=results_path,
    )
