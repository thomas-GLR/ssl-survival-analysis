import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.SiamesedDataset import SiameseDataset
from models.self_supervised.Autoencoder import Autoencoder

C_MAPSS_DIR = "data\\C_MAPSS"


def compute_autoencoder_loss(outputs, inputs, domain_tradeoff: float = 0.0):
    regression_loss = nn.MSELoss()(outputs, inputs)

    if domain_tradeoff > 0:
        batch_size = anchors.shape[0]
        # TODO Check if it used in the original paper and if yes need to replace domain_disc
        # domain_pred = self.domain_disc(embeddings[:batch_size])
        # domain_loss = nn.BCEWithLogitsLoss()(domain_pred, domain_labels)
        domain_loss = 0
    else:
        domain_loss = 0

    return regression_loss + domain_tradeoff * domain_loss


def train_one_epoch_siamese_network_autoencoder(
        model,
        optimizer: torch.optim.Optimizer,
        data: DataLoader,
        domain_tradeoff: float = 0.0
) -> float:
    model.train()
    running_loss = 0.
    last_loss = 0.

    batch_n = len(data)

    for i, batch in enumerate(data):
        anchors, queries, true_distances, domain_labels = batch

        optimizer.zero_grad()

        # Compute the regression_loss and the domain_loss (enable the model to predict the current domain (failure or censored data))
        combined = torch.cat([anchors, queries])
        embeddings = model.encoder(combined)
        outputs = model.decoder(embeddings)

        loss = compute_autoencoder_loss(outputs, combined, domain_tradeoff)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % batch_n == batch_n - 1:
            last_loss = running_loss / batch_n  # loss per batch
            # print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return last_loss


def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(data):
        x, y = batch

        prediction = model(x)

        loss = nn.MSELoss()(prediction, x)


if __name__ == "__main__":
    seq_len = 30

    train_pair_loader, val_pair_loader, source_val_loader, test_dataset = SiameseDataset.from_cmapss(
        dataset_root=C_MAPSS_DIR,
        sub_dataset="FD001",
        num_samples=25000,
        max_rul=125,
        min_distance=1,
        percent_of_broken_data=None,
        percent_of_censored_data=0.9,
        feature_select=None,
        norm_type="z-score",
        cluster_operations=True,
        norm_by_operations=True,
        validation_rate=0.2,
        distance_mode="linear",
        num_val_samples=25000,
        num_workers=0
    )

    for batch_idx, (anchors, queries, true_distances, domain_labels) in enumerate(train_pair_loader):
        print("batch:", batch_idx)
        print("Anchors shape:", anchors.shape)
        print("queries shape:", queries.shape)
        print("true_distances shape:", true_distances.shape)
        print("domain_labels shape:", domain_labels.shape)
        print("Anchors dtype:", anchors.dtype)
        print("queries dtype:", queries.dtype)
        print("true_distances dtype:", true_distances.dtype)
        print("domain_labels dtype:", domain_labels.dtype)
        break

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

    epochs = 100

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch_siamese_network_autoencoder(autoencoder, optimizer, train_pair_loader)

        running_vloss = 0.0
        autoencoder.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_pair_loader):
                combined = torch.cat([anchors, queries])
                embeddings = autoencoder.encoder(combined)
                outputs = autoencoder.decoder(embeddings)
                vloss = compute_autoencoder_loss(outputs, combined)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'Epoch {epoch}/{epochs} - Loss train {avg_loss} valid {avg_vloss}')
