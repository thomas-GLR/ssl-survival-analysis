import torch

from C_MAPSS.lightning.UnsupervisedPretrainingModule import UnsupervisedPretrainingModule


class MetricPretrainingModule(UnsupervisedPretrainingModule):
    def __init__(
        self,
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout,
        lr,
        weight_decay,
    ):
        super().__init__(
            in_channels=in_channels,
            seq_len=seq_len,
            num_layers=num_layers,
            kernel_size=kernel_size,
            base_filters=base_filters,
            latent_dim=latent_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.hparams["mode"] = "metric"

    def forward(self, anchors, queries):
        anchor_embeddings, query_embeddings = self._get_anchor_query_embeddings(
            anchors, queries
        )
        pred_distances = self._pairwise_distance(anchor_embeddings, query_embeddings)

        return pred_distances

    def _get_losses(self, batch):
        anchors, queries, true_distances = batch
        anchor_embeddings, query_embeddings = self._get_anchor_query_embeddings(
            anchors, queries
        )
        pred_distances = self._pairwise_distance(anchor_embeddings, query_embeddings)
        regression_loss = self.criterion_regression(pred_distances, true_distances)

        return regression_loss

    def _pairwise_distance(self, anchor_embeddings, query_embeddings):
        return torch.pairwise_distance(anchor_embeddings, query_embeddings, eps=1e-8)

    def _get_anchor_query_embeddings(self, anchors, queries):
        batch_size = anchors.shape[0]
        combined = torch.cat([anchors, queries])
        embeddings = self.encoder(combined)
        anchor_embeddings, query_embeddings = torch.split(embeddings, batch_size)

        return anchor_embeddings, query_embeddings
