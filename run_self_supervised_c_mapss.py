import torch
import torch.nn as nn

from dataset import CMAPSSDataset, tmp
from models.self_supervised.Autoencoder import Autoencoder

C_MAPSS_DIR = "data\\C_MAPSS"


def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(data):
        x, y = batch

        prediction = model(x)

        loss = nn.MSELoss()(prediction, x)


if __name__ == "__main__":
    seq_len = 30

    train_loader, [val_pairs_loader, source_val_loader] = tmp.get_data_loader_siamese_network(
        dataset_root=C_MAPSS_DIR,
        fd_source=1,
        num_samples=25000,
        batch_size=1,
        window_size=1,
        max_rul=125,
        min_distance=1,
        percent_broken=0.1,
        percent_fail_runs=0.1,
        feature_select=None,
        norm_type="z-score",
        cluster_operations=True,
        norm_by_operations=True,
        validation_rate=0.2,
        truncate_val=True,
        distance_mode="linear",
        num_val_samples=25000,
        num_workers=0
    )

    # for batch_idx, (anchors, queries, true_distances, domain_labels) in enumerate(train_loader):
    #     print("batch:", batch_idx)
    #     print("Anchors shape:", anchors.shape)
    #     print("queries shape:", queries.shape)
    #     print("true_distances shape:", true_distances.shape)
    #     print("domain_labels shape:", domain_labels.shape)
    #     print("Anchors dtype:", anchors.dtype)
    #     print("queries dtype:", queries.dtype)
    #     print("true_distances dtype:", true_distances.dtype)
    #     print("domain_labels dtype:", domain_labels.dtype)
    #     break
    #
    # for batch_idx, (anchors, queries, true_distances, domain_labels) in enumerate(val_pairs_loader):
    #     print("batch:", batch_idx)
    #     print("Anchors shape:", anchors.shape)
    #     print("queries shape:", queries.shape)
    #     print("true_distances shape:", true_distances.shape)
    #     print("domain_labels shape:", domain_labels.shape)
    #     print("Anchors dtype:", anchors.dtype)
    #     print("queries dtype:", queries.dtype)
    #     print("true_distances dtype:", true_distances.dtype)
    #     print("domain_labels dtype:", domain_labels.dtype)
    #     break
    #
    # for batch_idx, (x, y) in enumerate(source_val_loader):
    #     print("batch:", batch_idx)
    #     print("x shape:", x.shape)
    #     print("y shape:", y.shape)
    #     print("x dtype:", x.dtype)
    #     print("y dtype:", y.dtype)
    #
    #     for idx, label in enumerate(y):
    #         print(f"Label n°{idx} : {label}")
    #     break

    # train_dataset, test_dataset, valid_dataset = CMAPSSDataset.get_data_loaders(
    #     dataset_root=C_MAPSS_DIR,
    #     sequence_len=1,
    #     sub_dataset='FD001',
    #     norm_type='z-score',
    #     max_rul=125,
    #     cluster_operations=False,
    #     norm_by_operations=False,
    #     use_max_rul_on_test=True,
    #     validation_rate=0.2,
    #     return_id=False,
    #     use_only_final_on_test=True,
    #     loader_kwargs={'batch_size': 64}
    # )
    #
    # for batch_idx, (x, y) in enumerate(train_dataset):
    #     print("batch:", batch_idx)
    #     print("x shape:", x.shape)
    #     print("y shape:", y.shape)
    #     print("x dtype:", x.dtype)
    #     print("y dtype:", y.dtype)
    #
    #     break

    print(f"Nombre data : {len(source_val_loader)}")
    for batch_idx, (x, y) in enumerate(source_val_loader):
        # print(f"{batch_idx} - {x} -> {y}")
        print(f"{y}")

    # autoencoder = Autoencoder(
    #     in_channels=24,
    #     seq_len=seq_len,
    #     num_layers=6,
    #     kernel_size=3,
    #     base_filters=16,
    #     latent_dim=64,
    #     dropout=0.1
    # )
    #
    # # Conv1d wait for a tensor with shape (batch_size, channels, seq_len) where channels mean features
    # train_dataset.x = train_dataset.x.permute(0, 2, 1)
    #
    # optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=0.0)
    #
    # epochs = 100
    #
    # for epoch in range(1, epochs + 1):
    #     autoencoder.train()
    #     optimizer.zero_grad()
