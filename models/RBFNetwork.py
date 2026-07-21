from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_centers(
        init_data: torch.Tensor,
        n_centers: int,
        generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Initializes RBF centers by randomly sampling rows from data.

    Args:
        init_data: Reference feature vectors of shape ``(n_samples, in_features)``
            to sample centers from.
        n_centers: Number of centers to sample.
        generator: Random generator controlling which rows are sampled. Passing a
            different generator per committee member is how CoBCReg's "different
            random initialization of centers" diversity source is realized.

    Returns:
        A tensor of shape ``(n_centers, in_features)`` holding the sampled centers.

    Raises:
        ValueError: If ``init_data`` has fewer rows than ``n_centers``.
    """
    if init_data.shape[0] < n_centers:
        raise ValueError(
            f"init_data has {init_data.shape[0]} rows, need at least {n_centers} to sample centers from."
        )
    indices = torch.randperm(init_data.shape[0], generator=generator)[:n_centers]
    return init_data[indices].clone()


def _init_widths_from_centers(
        centers: torch.Tensor,
        distance_order: float,
        width_neighbors: int,
        width_scale: float,
) -> torch.Tensor:
    """Initializes per-center RBF widths from the spread of the centers themselves.

    Uses a Moody-Darken-style heuristic: each center's width is the mean distance
    to its ``width_neighbors`` nearest other centers, computed under the same
    Minkowski order used by the network's basis function, so the width scale stays
    consistent with the distance metric it will be divided against at inference.

    Args:
        centers: RBF centers of shape ``(n_centers, in_features)``.
        distance_order: Minkowski distance order ``p`` (``p=2`` is Euclidean).
        width_neighbors: Number of nearest neighboring centers to average over.
        width_scale: Multiplicative scale applied to the resulting widths.

    Returns:
        A tensor of shape ``(n_centers,)`` with strictly positive initial widths.
    """
    n_centers = centers.shape[0]
    k = min(width_neighbors, n_centers - 1)
    pairwise = torch.cdist(centers, centers, p=distance_order)
    pairwise.fill_diagonal_(float("inf"))
    nearest, _ = torch.topk(pairwise, k=k, largest=False, dim=1)
    widths = nearest.mean(dim=1) * width_scale
    return widths.clamp(min=1e-6)


class RBFNetwork(nn.Module):
    """Radial basis function network regressor for a CoBCReg committee member.

    Reference: Hady, Schwenker & Palm, "Semi-supervised Learning for Regression
    with Co-training by Committee" — CoBCReg's committee members are RBF networks
    whose Gaussian basis is evaluated on a Minkowski distance of order ``p`` instead
    of the plain Euclidean distance. ``p=2`` recovers the standard Gaussian RBF
    exactly. Varying ``p`` across committee members (alongside different bootstrap
    samples and different random center initializations, both handled outside this
    class) is CoBCReg's diversity mechanism.
    """

    def __init__(
            self,
            in_features: int,
            n_centers: int,
            distance_order: float = 2.0,
            trainable_centers: bool = True,
            trainable_widths: bool = True,
            width_neighbors: int = 2,
            width_scale: float = 1.0,
            init_data: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initializes the RBF network.

        Args:
            in_features: Number of input features per sample.
            n_centers: Number of RBF centers (hidden units).
            distance_order: Minkowski distance order ``p`` used inside the Gaussian
                basis (``p=2`` is the standard Euclidean Gaussian RBF).
            trainable_centers: Whether the centers are updated by backprop. If
                ``False`` they stay fixed at their initial (random or data-sampled)
                values.
            trainable_widths: Whether the per-center widths are updated by backprop.
                If ``False`` they stay fixed at their heuristic initial values.
            width_neighbors: Number of nearest neighboring centers averaged over
                when initializing widths.
            width_scale: Multiplicative scale applied to the heuristic initial
                widths.
            init_data: Optional reference feature vectors of shape
                ``(n_samples, in_features)`` used to initialize centers by random
                subset sampling. If ``None``, centers fall back to a standard
                normal init, which is only meant for quick smoke tests.
            generator: Random generator controlling center sampling / fallback
                initialization. Use a different generator per committee member to
                get CoBCReg's random-center-init diversity.
        """
        super().__init__()
        self.in_features = in_features
        self.n_centers = n_centers
        self.distance_order = distance_order

        if init_data is not None:
            centers = _init_centers(init_data, n_centers, generator)
        else:
            centers = torch.randn(n_centers, in_features, generator=generator)

        if trainable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        widths = _init_widths_from_centers(centers, distance_order, width_neighbors, width_scale)
        raw_widths = widths + torch.log(-torch.expm1(-widths))
        if trainable_widths:
            self.raw_widths = nn.Parameter(raw_widths)
        else:
            self.register_buffer("raw_widths", raw_widths)

        self.linear = nn.Linear(n_centers, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the regression output for a batch of feature vectors.

        Args:
            x: Input features of shape ``(batch, in_features)``.

        Returns:
            Predicted regression targets of shape ``(batch, 1)``.
        """
        distances = torch.cdist(x, self.centers, p=self.distance_order)
        widths = F.softplus(self.raw_widths)
        basis = torch.exp(-(distances ** 2) / (2 * widths ** 2 + 1e-12))
        return self.linear(basis)
