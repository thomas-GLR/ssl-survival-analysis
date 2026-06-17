"""
Semi-Supervised Generative M2 Model using Variational Autoencoders (PyTorch)
=============================================================================
Based on:
  Zhang et al., "Semi-Supervised Bearing Fault Diagnosis and Classification
  Using Variational Autoencoder-Based Deep Generative Models",
  IEEE Sensors Journal, Vol. 21, No. 5, March 2021.

And the original deep generative model from:
  Kingma et al., "Semi-supervised Learning with Deep Generative Models",
  NeurIPS 2014.

Architecture
------------
The M2 model jointly trains:
  - An encoder q_phi(z | x)          -> Gaussian latent variable z
  - A classifier q_phi(y | x)        -> Categorical label distribution
  - A decoder p_theta(x | y, z)      -> Reconstructs input given z and y

Loss / Objective
----------------
The combined objective (Eq. 7 in the paper) is:

    J^alpha = sum_{x ~ p_u}  U(x)
            + sum_{(x,y) ~ p_l} [ L(x, y) - alpha * log q_phi(y|x) ]

where:

  U(x) = -ELBO_U
       = - sum_y q_phi(y|x) * [ -L(x,y) ] - H(q_phi(y|x))   (Eq. 5)

  L(x, y) = -ELBO_L
           = - E_{q_phi(z|x)} [ log p_theta(x|y,z)
                               + log p_theta(y)
                               + log p(z)
                               - log q_phi(z|x) ]             (Eq. 6)
           = reconstruction_loss + KL(q_phi(z|x) || p(z)) - log p(y)

  alpha controls the relative weight between generative and discriminative
  learning. The paper sets alpha = 0.1 * N, where N is the number of
  labeled samples.

KL Cost Annealing (beta-VAE, Eq. 8)
------------------------------------
To mitigate "KL vanishing", a weight beta is annealed from 0 -> 1 during
training. The revised ELBO becomes:

    ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))

Network Structure (from Section III.C of the paper)
----------------------------------------------------
Input dimension  : 1024 (one vibration signal segment)
Latent dimension : 128
Encoder q(z|x)   : Conv1d x2  -> FC  (outputs mu_z, log_var_z)
Classifier q(y|x): Conv1d x2 + MaxPool1d x2 + FC -> Softmax
Decoder p(x|y,z) : FC -> ConvTranspose1d x3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from models.vae_m2.Classifier import Classifier
from models.vae_m2.Decoder import Decoder
from models.vae_m2.Encoder import Encoder


# ---------------------------------------------------------------------------
# M2 Model
# ---------------------------------------------------------------------------

class VaeM2(nn.Module):
    """
    Semi-Supervised Generative M2 Model.

    Parameters
    ----------
    input_dim   : length of one vibration signal segment (default 1024)
    latent_dim  : dimension of latent space z (default 128)
    num_classes : number of fault categories (default 10)
    dropout     : dropout rate used in encoder and classifier
    alpha       : weight on the discriminative loss term (Eq. 7).
                  Paper recommends alpha = 0.1 * N_labeled.
                  Pass None to set it at training time via compute_loss().
    beta        : initial KL weight for cost annealing (Eq. 8).
                  Anneal from 0 -> 1 externally by updating model.beta.

    Usage
    -----
    model = VAE_M2(num_classes=10, alpha=0.1 * N_labeled)

    # Training step — labeled batch
    loss, info = model.compute_loss(x_l, y_l, x_u=None)

    # Training step — unlabeled batch
    loss, info = model.compute_loss(x_l=None, y_l=None, x_u=x_u)

    # Training step — mixed batch (both in same call)
    loss, info = model.compute_loss(x_l, y_l, x_u)

    # Inference
    probs = model.classify(x)    # (B, num_classes)
    pred  = probs.argmax(dim=-1) # class indices
    """

    def __init__(
        self,
        input_dim:   int   = 1024,
        latent_dim:  int   = 128,
        num_classes: int   = 10,
        dropout:     float = 0.25,
        alpha:       float = None,
        beta:        float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.alpha       = alpha
        self.beta        = beta   # annealed externally from 0 -> 1

        # Sub-networks
        self.encoder    = Encoder(input_dim, latent_dim, dropout)
        self.classifier = Classifier(input_dim, num_classes, dropout)
        self.decoder    = Decoder(input_dim, latent_dim, num_classes)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z ~ q(z|x) using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at eval time

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (mu_z, log_var_z) for input x."""
        return self.encoder(x)

    def classify(self, x: Tensor) -> Tensor:
        """Return predicted class probabilities (B, num_classes)."""
        return self.classifier(x)

    def decode(self, z: Tensor, y_onehot: Tensor) -> Tensor:
        """Reconstruct x given latent z and one-hot label y."""
        return self.decoder(z, y_onehot)

    # ------------------------------------------------------------------
    # ELBO for a single (x, y) pair — L(x, y)  [Eq. 6]
    # ------------------------------------------------------------------

    def _elbo_labeled(self, x: Tensor, y_onehot: Tensor) -> Tensor:
        """
        Computes -L(x, y) = reconstruction + beta*KL - log p(y)

        Shapes
        ------
        x       : (B, input_dim)
        y_onehot: (B, num_classes)
        Returns : scalar  (mean over batch)
        """
        # Encode
        mu_z, log_var_z = self.encode(x)         # (B, latent_dim)
        z = self.reparameterize(mu_z, log_var_z) # (B, latent_dim)

        # Decode
        x_hat = self.decode(z, y_onehot)         # (B, input_dim)

        # --- Reconstruction loss: -E[log p(x|y,z)] ---
        # MSE is standard for continuous signals (Gaussian decoder)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        # --- KL divergence: KL(q(z|x) || p(z)) ---
        # Analytical form for Gaussian:  0.5 * sum(mu^2 + var - log_var - 1)
        kl_loss = -0.5 * torch.mean(
            1 + log_var_z - mu_z.pow(2) - log_var_z.exp()
        )

        # --- Prior on y: log p(y) = log(1/C) = -log(C) ---
        # Uniform prior over C classes; constant per batch so it regularises
        # the classifier towards balanced predictions.
        log_py = -torch.log(torch.tensor(float(self.num_classes),
                                         device=x.device))

        # -L(x, y)  =  recon + beta*KL - log p(y)
        return recon_loss + self.beta * kl_loss - log_py

    # ------------------------------------------------------------------
    # ELBO for unlabeled data — U(x)  [Eq. 5]
    # ------------------------------------------------------------------

    def _elbo_unlabeled(self, x: Tensor) -> Tensor:
        """
        Computes U(x) = -ELBO_U
                      = sum_y q(y|x) * L(x,y)  -  H(q(y|x))

        We marginalise over all classes by computing L(x, y) for every
        possible y and weighting by q_phi(y|x).

        Returns : scalar (mean over batch)
        """
        # Class probabilities: (B, C)
        q_y = self.classify(x)

        # Build one-hot matrices for all C classes at once
        # y_all shape: (C, B, C)  — y_all[c] is a batch where every row is
        # the one-hot for class c.
        B, C = x.size(0), self.num_classes
        device = x.device

        # Expand x to (C*B, input_dim) and y_onehot to (C*B, C)
        x_expanded = x.unsqueeze(0).expand(C, -1, -1).reshape(C * B, -1)

        eye = torch.eye(C, device=device)           # (C, C)
        y_expanded = eye.unsqueeze(1).expand(-1, B, -1).reshape(C * B, C)

        # Compute -L(x, y) for every class simultaneously
        neg_elbo_all = self._elbo_labeled(x_expanded, y_expanded)  # scalar
        # We need per-sample per-class values; recompute without .mean():
        neg_elbo_per = self._neg_elbo_per_sample(x_expanded, y_expanded)
        # neg_elbo_per: (C*B,) -> (C, B)
        neg_elbo_per = neg_elbo_per.view(C, B)     # (C, B)

        # Weighted sum:  sum_y q(y|x) * L(x,y)
        # q_y: (B, C)  -> transpose to (C, B)
        q_y_T = q_y.t()                            # (C, B)
        weighted = (q_y_T * neg_elbo_per).sum(0)   # (B,)

        # Entropy of classifier: H(q(y|x)) = -sum_y q(y|x) log q(y|x)
        entropy = -(q_y * (q_y + 1e-8).log()).sum(-1)  # (B,)

        # U(x) = weighted - entropy  (note the sign: we minimise U)
        u_x = weighted - entropy                   # (B,)
        return u_x.mean()

    def _neg_elbo_per_sample(self, x: Tensor, y_onehot: Tensor) -> Tensor:
        """
        Per-sample version of _elbo_labeled (no .mean() reduction).
        Used internally by _elbo_unlabeled.

        Returns : (B,) tensor
        """
        mu_z, log_var_z = self.encode(x)
        z = self.reparameterize(mu_z, log_var_z)
        x_hat = self.decode(z, y_onehot)

        # Per-sample MSE: mean over input_dim
        recon = F.mse_loss(x_hat, x, reduction='none').mean(-1)  # (B,)

        # Per-sample KL
        kl = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).mean(-1)

        log_py = -torch.log(torch.tensor(float(self.num_classes),
                                         device=x.device))

        return recon + self.beta * kl - log_py   # (B,)

    # ------------------------------------------------------------------
    # Combined objective  J^alpha  [Eq. 7]
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x_l:   Tensor = None,   # labeled inputs   (B_l, input_dim)
        y_l:   Tensor = None,   # integer labels    (B_l,)
        x_u:   Tensor = None,   # unlabeled inputs  (B_u, input_dim)
        alpha: float  = None,   # override instance alpha if provided
    ) -> Tuple[Tensor, dict]:
        """
        Compute the combined M2 loss J^alpha (Eq. 7).

        At least one of (x_l, y_l) or x_u must be provided.

        Parameters
        ----------
        x_l   : labeled vibration segments, shape (B_l, input_dim)
        y_l   : integer class labels,        shape (B_l,)
        x_u   : unlabeled vibration segments, shape (B_u, input_dim)
        alpha : discriminative weight; falls back to self.alpha if None.

        Returns
        -------
        loss : scalar tensor (to call .backward() on)
        info : dict with individual loss components for logging
        """
        _alpha = alpha if alpha is not None else self.alpha
        if _alpha is None:
            raise ValueError(
                "alpha must be set either at __init__ or passed to compute_loss(). "
                "The paper recommends alpha = 0.1 * N_labeled."
            )

        loss = torch.tensor(0.0, device=(
            x_l.device if x_l is not None else x_u.device))
        info = {}

        # ---- Labeled term: L(x, y) - alpha * log q(y|x) ----
        if x_l is not None and y_l is not None:
            y_onehot = F.one_hot(y_l, self.num_classes).float().to(x_l.device)

            labeled_elbo = self._elbo_labeled(x_l, y_onehot)   # scalar

            # Discriminative loss: -alpha * log q(y|x)
            q_y = self.classify(x_l)                            # (B_l, C)
            log_q_y = (q_y + 1e-8).log()
            # NLL of the true label
            disc_loss = F.nll_loss(log_q_y, y_l, reduction='mean')

            labeled_loss = labeled_elbo + _alpha * disc_loss
            loss = loss + labeled_loss

            info['labeled_elbo']  = labeled_elbo.item()
            info['disc_loss']     = disc_loss.item()
            info['labeled_total'] = labeled_loss.item()

        # ---- Unlabeled term: U(x) ----
        if x_u is not None:
            unlabeled_loss = self._elbo_unlabeled(x_u)
            loss = loss + unlabeled_loss
            info['unlabeled_elbo'] = unlabeled_loss.item()

        info['total_loss'] = loss.item()
        return loss, info

    # ------------------------------------------------------------------
    # Convenience: forward for inference only
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Return class probabilities for inference."""
        return self.classify(x)

    # ------------------------------------------------------------------
    # KL annealing helper
    # ------------------------------------------------------------------

    def anneal_beta(self, epoch: int, total_epochs: int) -> None:
        """
        Linear KL cost annealing from 0 to 1 over the first `total_epochs`
        epochs (Eq. 8 / Section III.C.1 of the paper).

        Call at the start of each epoch:
            model.anneal_beta(epoch, warmup_epochs)
        """
        self.beta = min(1.0, epoch / max(total_epochs, 1))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    INPUT_DIM   = 1024
    LATENT_DIM  = 128
    NUM_CLASSES = 10
    N_LABELED   = 516          # ~4% of 12900 training samples (paper Sec IV.C)
    ALPHA       = 0.1 * N_LABELED   # = 51.6

    model = VaeM2(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        alpha=ALPHA,
        beta=0.0,              # starts at 0; anneal to 1 during warmup
    )

    # Simulate a batch from the CWRU dataset (batch_size=200 per paper)
    B_labeled   = 20
    B_unlabeled = 180

    x_l = torch.randn(B_labeled,   INPUT_DIM)
    y_l = torch.randint(0, NUM_CLASSES, (B_labeled,))
    x_u = torch.randn(B_unlabeled, INPUT_DIM)

    # Anneal beta at epoch 5 of 10 warmup epochs -> beta = 0.5
    model.anneal_beta(epoch=5, total_epochs=10)
    print(f"Beta after annealing: {model.beta:.2f}")

    model.train()
    loss, info = model.compute_loss(x_l=x_l, y_l=y_l, x_u=x_u)
    print("\n--- Training loss (mixed batch) ---")
    for k, v in info.items():
        print(f"  {k:25s}: {v:.4f}")
    loss.backward()
    print("\nBackward pass OK")

    # Inference
    model.eval()
    with torch.no_grad():
        probs = model.classify(x_l)
        preds = probs.argmax(dim=-1)
    print(f"\nPredicted classes (first 5): {preds[:5].tolist()}")
    print(f"Class probs sum to 1: {probs.sum(-1).allclose(torch.ones(B_labeled))}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {n_params:,}")