import torch.nn as nn
import torch

from models.self_supervised.base.Encoder import Encoder


class MetricLearning(nn.Module):
    def __init__(
            self,
            in_channels,
            seq_len,
            num_layers,
            kernel_size,
            base_filters,
            latent_dim,
            dropout):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.encoder = Encoder(
            self.in_channels,
            self.base_filters,
            self.kernel_size,
            self.num_layers,
            self.latent_dim,
            self.seq_len,
            self.dropout,
            norm_outputs=True,
        )

    def forward(self, anchors, queries):
        anchor_embeddings, query_embeddings = self._get_anchor_query_embeddings(
            anchors, queries
        )
        pred_distances = self._pairwise_distance(anchor_embeddings, query_embeddings)

        return pred_distances

    def _pairwise_distance(self, anchor_embeddings, query_embeddings):
        return torch.pairwise_distance(anchor_embeddings, query_embeddings, eps=1e-8)

    def _get_anchor_query_embeddings(self, anchors, queries):
        batch_size = anchors.shape[0]
        combined = torch.cat([anchors, queries])
        embeddings = self.encoder(combined)
        anchor_embeddings, query_embeddings = torch.split(embeddings, batch_size)

        return anchor_embeddings, query_embeddings