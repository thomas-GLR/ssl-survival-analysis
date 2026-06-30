import os
from datetime import datetime
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning.AutoencoderPretrainingModule import AutoencoderPretrainingModule
from C_MAPSS.lightning.BaselineModule import BaselineModule
from C_MAPSS.lightning.MetricPretrainingModule import MetricPretrainingModule
from dataset.SiamesedDataset import SiameseDataset
from C_MAPSS.utils import utils_cmapss


def train_self_supervised(
        checkpoints_path: str,
        results_path: str,
        model_version: str,  # metric or autoencoder
        # Dataset params
        dataset_root: str,
        sub_dataset: str,
        seed: int | None,
        max_rul: int,
        return_sequence_label: bool,
        norm_type: str,
        cluster_operations: bool,
        norm_by_operations: bool,
        include_cols: list[str] | None,
        exclude_cols: list[str] | None,
        return_id: bool,
        validation_rate: float,
        use_only_final_on_test: bool,
        use_max_rul_on_test: bool,
        use_max_rul_on_valid: bool,
        percent_of_broken_data: float | None,
        percent_of_censored_data: float,

        # Pretrain model parameters
        sequence_len: int,
        pretraining_lr: float,
        dropout: float,
        num_layers: int = 6,
        kernel_size: int = 3,
        base_filters: int = 16,
        latent_dim: int = 64,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        patience: int = 50,
        batch_size_pretraining: int = 64,

        # Baseline parameters
        latent_dim_baseline=64,
        lr_baseline=0.01,
        max_epochs_baseline: int = 100,
        batch_size_baseline: int = 64,

        device: str | None=None,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
) -> tuple[float, float]:
    utils_cmapss.assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )

    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

    broken_percentage = 0. if percent_of_broken_data is None else percent_of_broken_data

    folder_for_current_pretraining = f"pre-trained-model-{model_version}-turbofan-{sub_dataset}-{datetime_for_folders}/censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"
    folder_for_current_training = f"model-baseline-with-{model_version}-turbofan-{sub_dataset}-{datetime_for_folders}/censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"

    pretraining_checkpoints_path = f"{checkpoints_path}/{folder_for_current_pretraining}"
    training_checkpoints_path = f"{checkpoints_path}/{folder_for_current_training}"

    # First we will pretrain the unsupervised model
    dataset_params = {
        'dataset_root': dataset_root,
        'sub_dataset': sub_dataset,
        'seq_len': sequence_len,
        'max_rul': max_rul,
        'norm_type': norm_type,
        'cluster_operations': cluster_operations,
        'norm_by_operations': norm_by_operations,
        'include_cols': include_cols,
        'exclude_cols': exclude_cols,
        'validation_rate': validation_rate,
        'use_only_final_on_test': use_only_final_on_test,
        'use_max_rul_on_test': use_max_rul_on_test,
        'use_max_rul_on_valid': use_max_rul_on_valid,
        'percent_of_broken_data': percent_of_broken_data,
        'percent_of_censored_data': percent_of_censored_data,
        'batch_size': batch_size_pretraining,
        'seed': seed,
    }

    print(f"Creating dataset for siamesed networks with params : {dataset_params}")

    train_pair_loader, val_pair_loader = get_pair_loader_for_pretraining(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        seq_len=sequence_len,
        max_rul=max_rul,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
        validation_rate=validation_rate,
        use_only_final_on_test=use_only_final_on_test,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        batch_size=batch_size_pretraining,
        seed=seed
    )

    baseline_dataset_params = {
        'dataset_root': dataset_root,
        'sub_dataset': sub_dataset,
        'sequence_len': sequence_len,
        'max_rul': max_rul,
        'return_sequence_label': return_sequence_label,
        'norm_type': norm_type,
        'cluster_operations': cluster_operations,
        'norm_by_operations': norm_by_operations,
        'include_cols': include_cols,
        'exclude_cols': exclude_cols,
        'return_id': return_id,
        'validation_rate': validation_rate,
        'use_only_final_on_test': use_only_final_on_test,
        'use_max_rul_on_test': use_max_rul_on_test,
        'use_max_rul_on_valid': use_max_rul_on_valid,
        'percent_of_broken_data': percent_of_broken_data,
        'percent_of_censored_data': percent_of_censored_data,
        'seed': seed,
    }

    print(f"Creating dataset for baseline model with params : {baseline_dataset_params}")

    train_dataset, val_dataset, test_dataset = CMAPSSLoader.get_datasets(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        sequence_len=sequence_len,
        max_rul=max_rul,
        return_sequence_label=return_sequence_label,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
        return_id=return_id,
        validation_rate=validation_rate,
        use_only_final_on_test=use_only_final_on_test,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        seed=seed,
    )

    in_channels = len(train_dataset.feature_cols)

    trainer = build_trainer(
        checkpoints_path=pretraining_checkpoints_path,
        device=device,
        max_epochs=max_epochs,
        patience=patience
    )

    pretraining_model_parameters = {
        'in_channels': in_channels,
        'seq_len': sequence_len,
        'num_layers': num_layers,
        'kernel_size': kernel_size,
        'base_filters': base_filters,
        'latent_dim': latent_dim,
        'dropout': dropout,
        'lr': pretraining_lr,
        'weight_decay': weight_decay,
    }

    print(f"Creating the pre-trained model with parameters : {pretraining_model_parameters}")

    if model_version == 'metric':
        model = MetricPretrainingModule(
            **pretraining_model_parameters
        )
    elif model_version == "autoencoder":
        model = AutoencoderPretrainingModule(
            **pretraining_model_parameters
        )
    else:
        raise ValueError(f"Unrecognized pre-training mode {model_version}.")

    print("Training pre-trained model")

    trainer.fit(model, train_dataloaders=train_pair_loader, val_dataloaders=val_pair_loader)

    print("Creating baseline model...")

    trainer, baseline = build_baseline(
        checkpoints_path=training_checkpoints_path,
        in_channels=in_channels,
        seq_len=sequence_len,
        device=device,
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        latent_dim=latent_dim_baseline,
        base_filters=base_filters,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr_baseline,
        max_epochs=max_epochs_baseline,
        patience=patience,
    )

    train_loader = train_dataset.get_data_loader_without_censored_data(batch_size_baseline, is_model_cnn=True)
    val_loader = val_dataset.get_data_loader_without_censored_data(batch_size_baseline, is_model_cnn=True)
    test_loader = test_dataset.get_data_loader_without_censored_data(batch_size_baseline, is_model_cnn=True)

    print("Training baseline model")

    trainer.fit(baseline, train_dataloaders=train_loader, val_dataloaders=val_loader or test_loader)
    trainer.test(baseline, dataloaders=test_loader)

    baseline_module_with_trained_model = BaselineModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    transformer_lstm_module_with_trained_model = baseline_module_with_trained_model.to(device)
    transformer_lstm_module_with_trained_model.eval()

    predictions = []
    targets = []

    for x, y in test_loader:
        x = x.to(device)
        y_hat = transformer_lstm_module_with_trained_model(x)
        predictions.extend(y_hat.cpu().detach().numpy().flatten())
        targets.extend(y.cpu().detach().numpy().flatten())

    predictions_tensor = torch.Tensor(predictions)
    targets_tensor = torch.Tensor(targets)

    rmse = torch.sqrt(torch.mean((targets_tensor - predictions_tensor) ** 2))
    score = utils_cmapss.cmapss_score(np.array(predictions), np.array(targets))

    return rmse.item(), score

def build_trainer(
        checkpoints_path: str,
        device: str,
        max_epochs: int,
        patience: int
) -> pl.Trainer:
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val/regression_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor='val/regression_loss',
        filename='checkpoint-{epoch:02d}-{val_regression_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    return pl.Trainer(
        default_root_dir=checkpoints_path,
        accelerator=device,
        max_epochs=max_epochs,
        num_sanity_val_steps=2,
        deterministic=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        callbacks=[early_stop_callback, checkpoint_callback],
    )


def get_pair_loader_for_pretraining(
        dataset_root: str,
        sub_dataset: str,
        seq_len: int,
        seed: int | None,
        max_rul: int,
        norm_type: str,
        include_cols: Optional[list[str]],
        exclude_cols: Optional[list[str]],
        percent_of_broken_data: float | None,
        percent_of_censored_data: float,
        cluster_operations: bool,
        norm_by_operations: bool,
        use_only_final_on_test: bool,
        use_max_rul_on_test: bool,
        use_max_rul_on_valid: bool,
        validation_rate: float,
        batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    return SiameseDataset.from_cmapss(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        window_size=seq_len,
        seed=seed,
        num_samples=25000,
        max_rul=max_rul,
        min_distance=1,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        use_only_final_on_test=use_only_final_on_test,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        feature_select=include_cols,
        exclude_cols=exclude_cols,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        validation_rate=validation_rate,
        distance_mode="linear",
        num_val_samples=25000,
        num_workers=0,
        batch_size=batch_size,
    )


def build_baseline(
        checkpoints_path: str,
        in_channels: int,
        seq_len: int,
        checkpoint_path: str,
        device: str,
        latent_dim: int,
        base_filters: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
        lr: float,
        max_epochs: int,
        patience: int,
) -> tuple[pl.Trainer, pl.LightningModule]:
    trainer = build_trainer(
        checkpoints_path=checkpoints_path,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )

    model = BaselineModule(
        in_channels=in_channels,
        seq_len=seq_len,
        latent_dim=latent_dim,
        base_filters=base_filters,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )

    model.load_encoder(checkpoint_path)

    return trainer, model
