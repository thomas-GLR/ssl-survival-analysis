import pytorch_lightning as pl
from torch import accelerator
from torch.utils.data import DataLoader

from C_MAPSS.lightning.MetricPretrainingModule import MetricPretrainingModule
from C_MAPSS.lightning.AutoencoderPretrainingModule import AutoencoderPretrainingModule
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from dataset.SiamesedDataset import SiameseDataset
from C_MAPSS.lightning.BaselineModule import BaselineModule


def train_self_supervised(
        # Dataset parameters
        dataset_root: str,
        sub_dataset: str,
        seq_len: int,
        max_rul: int,
        percent_of_broken_data: float | None,
        percent_of_censored_data: float,
        cluster_operations: bool,
        norm_by_operations: bool,
        validation_rate: float,
        seed: int | None,

        # Pretrain model parameters
        mode: str,  # metric or autoencoder
        in_channels: int,
        lr: float,
        dropout: float,
        num_layers: int = 6,
        kernel_size: int = 3,
        base_filters: int = 16,
        latent_dim: int = 64,
        num_disc_layers: int = 1,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        batch_size_pretraining: int = 64,

        # Baseline parameters
        num_layers_baseline=6,
        kernel_size_baseline=3,
        base_filters_baseline=16,
        latent_dim_baseline=64,
        dropout_baseline=0.1,
        lr_baseline=0.01,
        max_epochs_baseline: int = 100,
        batch_size_baseline: int = 64,

        device: str = "cpu",
):
    # First we will pretrain the unsupervised model
    trainer, model = build_pretraining(
        mode=mode,
        in_channels=in_channels,
        seq_len=seq_len,
        device=device,
        lr=lr,
        dropout=dropout,
        num_layers=num_layers,
        kernel_size=kernel_size,
        base_filters=base_filters,
        latent_dim=latent_dim,
        num_disc_layers=num_disc_layers,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
    )

    train_pair_loader, val_pair_loader, source_loader = get_pair_loader_for_pretraining(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        seq_len=seq_len,
        seed=seed,
        max_rul=max_rul,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        validation_rate=validation_rate,
        batch_size=batch_size_pretraining
    )

    trainer.fit(model, train_dataloaders=train_pair_loader, val_dataloaders=val_pair_loader)
    # if mode == 'metric':
    #     checkpoint = UnsupervisedPretraining.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # elif mode == "autoencoder":
    #     checkpoint = AutoencoderPretraining.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # else:
    #     raise ValueError(f"Unrecognized pre-training mode {mode}.")

    trainer, baseline = build_baseline(
        in_channels=in_channels,
        seq_len=seq_len,
        device=device,
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        num_layers=num_layers_baseline,
        kernel_size=kernel_size_baseline,
        base_filters=base_filters_baseline,
        latent_dim=latent_dim_baseline,
        dropout=dropout_baseline,
        lr=lr_baseline,
        max_epochs=max_epochs_baseline,
    )

    train_dataset, val_dataset, test_dataset = CMAPSSLoader.get_datasets(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        sequence_len=seq_len,
        max_rul=max_rul,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        norm_type="z-score",
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        validation_rate=validation_rate,
        seed=seed,
    )

    train_loader = train_dataset.get_data_loader_without_censored_data(batch_size_baseline, is_model_cnn=True)
    val_loader = val_dataset.get_data_loader_without_censored_data(batch_size_baseline, is_model_cnn=True)
    test_loader = test_dataset.get_data_loader_without_censored_data(batch_size_baseline, is_model_cnn=True)

    trainer.fit(baseline, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(baseline, dataloaders=test_loader)


def build_pretraining(
        mode: str,  # metric or autoencoder,
        in_channels: int,
        seq_len: int,
        lr: float,
        device: str,
        dropout: float,
        num_layers: int = 6,
        kernel_size: int = 3,
        base_filters: int = 16,
        latent_dim: int = 64,
        num_disc_layers: int = 1,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
) -> tuple[pl.Trainer, pl.LightningModule]:
    trainer = build_trainer(mode, device, max_epochs)

    if mode == 'metric':
        model = MetricPretrainingModule(
            in_channels=in_channels,
            seq_len=seq_len,
            num_layers=num_layers,
            kernel_size=kernel_size,
            base_filters=base_filters,
            latent_dim=latent_dim,
            dropout=dropout,
            domain_tradeoff=0.0,
            domain_disc_dim=latent_dim,
            num_disc_layers=num_disc_layers,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif mode == "autoencoder":
        model = AutoencoderPretrainingModule(
            in_channels=in_channels,
            seq_len=seq_len,
            num_layers=num_layers,
            kernel_size=kernel_size,
            base_filters=base_filters,
            latent_dim=latent_dim,
            dropout=dropout,
            domain_tradeoff=0.0,
            domain_disc_dim=latent_dim,
            num_disc_layers=num_disc_layers,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unrecognized pre-training mode {mode}.")

    return trainer, model


def build_trainer(model_name: str, device: str, max_epochs: int) -> pl.Trainer:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'./checkpoints/model-{model_name}-turbofan',
        monitor='val/regression_loss',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    return pl.Trainer(
        num_sanity_val_steps=2,
        max_epochs=max_epochs,
        accelerator=device,
        deterministic=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        callbacks=[checkpoint_callback],
    )


def get_pair_loader_for_pretraining(
        dataset_root: str,
        sub_dataset: str,
        seq_len: int,
        seed: int | None,
        max_rul: int,
        percent_of_broken_data: float | None,
        percent_of_censored_data: float,
        cluster_operations: bool,
        norm_by_operations: bool,
        validation_rate: float,
        batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_pair_loader, val_pair_loader, source_val_loader, test_dataset = SiameseDataset.from_cmapss(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        window_size=seq_len,
        seed=seed,
        num_samples=25000,
        max_rul=max_rul,
        min_distance=1,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        feature_select=None,
        norm_type="z-score",
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        validation_rate=validation_rate,
        distance_mode="linear",
        num_val_samples=25000,
        num_workers=0,
        batch_size=batch_size,
    )

    return train_pair_loader, val_pair_loader, source_val_loader


def build_baseline(
        in_channels: int,
        seq_len: int,
        checkpoint_path: str,
        device: str,
        num_layers=6,
        kernel_size=3,
        base_filters=16,
        latent_dim=64,
        dropout=0.1,
        lr=0.01,
        max_epochs=100,
) -> tuple[pl.Trainer, pl.LightningModule]:
    trainer = build_trainer("baseline", device, max_epochs)

    model = BaselineModule(
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout=dropout,
        lr=lr,
    )

    model.load_encoder(checkpoint_path)

    return trainer, model
