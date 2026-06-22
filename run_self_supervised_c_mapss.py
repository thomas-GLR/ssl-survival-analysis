import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from dataset.SiamesedDataset import SiameseDataset
from models.self_supervised import Baseline
from models.self_supervised.Autoencoder import Autoencoder
from utils.utils import score

C_MAPSS_DIR = "data/C_MAPSS"


def train_one_epoch_siamese_network_autoencoder(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: DataLoader
) -> float:
    model.train()
    sum_loss = 0.

    batch_n = len(data)

    for i, batch in enumerate(data):
        anchors, queries, true_distances, domain_labels = batch

        optimizer.zero_grad()

        # Compute the regression_loss and the domain_loss (enable the model to predict the current domain (failure or censored data))
        combined = torch.cat([anchors, queries])
        embeddings = model.encoder(combined)
        outputs = model.decoder(embeddings)

        loss = model.compute_loss(anchors, outputs, combined)

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / batch_n if batch_n > 0 else 0.


def train_one_epoch_siamese_network_baseline(model: nn.Module, optimizer: torch.optim.Optimizer, data: DataLoader):
    model.train()
    sum_loss = 0.

    batch_n = len(data)

    for i, batch in enumerate(data):
        features, targets = batch

        optimizer.zero_grad()

        predictions = model(features)

        loss = model.compute_loss(predictions, targets)

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / batch_n if batch_n > 0 else 0.


def train_or_get_autoencoder(
        sub_dataset: str,
        seq_len: int,
        max_rul: int,
        percent_of_broken_data: float | None,
        percent_of_censored_data: float,
        cluster_operations: bool,
        norm_by_operations: bool,
        validation_rate: float,
        epochs: int,
        autoencoder_path: str = "",
        force_training: bool = False,
) -> Autoencoder:
    if not force_training and os.path.isfile(autoencoder_path):
        print(f"Loading the autoencoder from \"{autoencoder_path}\"...")

        return torch.load(autoencoder_path, weights_only=False)

    train_pair_loader, val_pair_loader, source_val_loader, test_dataset = SiameseDataset.from_cmapss(
        dataset_root=C_MAPSS_DIR,
        sub_dataset=sub_dataset,
        window_size=seq_len,
        num_samples =25000,
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
        num_workers=0
    )

    autoencoder = Autoencoder(
        in_channels=24,
        seq_len=seq_len,
        num_layers=6,
        kernel_size=3,
        base_filters=16,
        latent_dim=64,
        dropout=0.1
    )

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=0.0)

    print(f"Training autoencoder for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch_siamese_network_autoencoder(autoencoder, optimizer, train_pair_loader)

        running_vloss = 0.0
        autoencoder.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_pair_loader):
                anchors, queries, true_distances, domain_labels = batch

                combined = torch.cat([anchors, queries])
                embeddings = autoencoder.encoder(combined)
                outputs = autoencoder.decoder(embeddings)

                vloss = autoencoder.compute_loss(anchors, outputs, combined)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'Epoch {epoch}/{epochs} - Loss train {avg_loss} valid {avg_vloss}')

    print("Finished Training")
    print("Saving model in file autoencoder.pth...")

    torch.save(autoencoder, "autoencoder.pth")

    return autoencoder


def train_or_get_baseline_regressor_model(
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        baseline_regressor_path: str = "",
        force_training: bool = False,
) -> Baseline:
    if not force_training and os.path.isfile(baseline_regressor_path):
        print(f"Loading the baseline regressor from \"{baseline_regressor_path}\"...")

        return torch.load(baseline_regressor_path, weights_only=False)


    baseline_regressor = Baseline(autoencoder.encoder)

    param_groups = [
        {"params": baseline_regressor.encoder.parameters()},
        {"params": baseline_regressor.regressor.parameters()},
    ]

    regressor_optimizer = torch.optim.Adam(param_groups, lr=0.01)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch_siamese_network_baseline(baseline_regressor, regressor_optimizer, train_loader)

        running_vloss = 0.0
        autoencoder.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                features, targets = batch

                predictions = baseline_regressor(features)

                vloss = baseline_regressor.compute_loss(predictions, targets)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'Epoch {epoch}/{epochs} - Loss train {avg_loss} valid {avg_vloss}')

    print("Finished Training")

    print("Saving model in file baseline_regressor.pth...")

    torch.save(baseline_regressor, "baseline_regressor.pth")

    return baseline_regressor


if __name__ == "__main__":
    seq_len = 30
    max_rul = 125
    percent_of_broken_data = None
    percent_of_censored_data = 0.9
    cluster_operations = True
    norm_by_operations = True
    validation_rate = 0.2
    epochs = 100

    sub_dataset = "FD001"

    autoencoder = train_or_get_autoencoder(
        sub_dataset=sub_dataset,
        seq_len=seq_len,
        max_rul=max_rul,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        epochs=epochs,
        validation_rate=validation_rate,
        autoencoder_path="autoencoder.pth"
    )

    train_dataset, val_dataset, test_dataset = CMAPSSLoader.get_datasets(
        dataset_root=C_MAPSS_DIR,
        sub_dataset=sub_dataset,
        sequence_len=seq_len,
        max_rul=max_rul,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        norm_type="z-score",
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        validation_rate=validation_rate,
    )

    train_loader = train_dataset.get_data_loader_without_censored_data(256, is_model_cnn=True)
    val_loader = val_dataset.get_data_loader_without_censored_data(256, is_model_cnn=True)
    test_loader = test_dataset.get_data_loader_without_censored_data(256, is_model_cnn=True)

    baseline_regressor = train_or_get_baseline_regressor_model(train_loader, val_loader, epochs, "baseline_regressor.pth")

    baseline_regressor.eval()

    test_results = []
    for batch in test_loader:
        x, y = batch
        y_hat = baseline_regressor(x)
        test_results.append((y, y_hat))

    # Calculate RMSE
    rmse = np.sqrt(np.mean([(y - y_hat).pow(2).mean().item() for y, y_hat in test_results]))
    # Calculate score
    y_true = torch.cat([y for y, _ in test_results])
    y_pred = torch.cat([y_hat for _, y_hat in test_results])
    print(f'Test RMSE for {sub_dataset}: {rmse}')
    print(f'Score for {sub_dataset}: {score(y_pred.detach().numpy(), y_true.detach().numpy())}')
