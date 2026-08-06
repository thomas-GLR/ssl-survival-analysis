"""Lightning wrapper for ordinal-regression RUL models on Scania Component X.

This is the classification counterpart of :class:`~scania.lightning_module.BasicLightningModule`.
Instead of regressing a scalar RUL with MSE, it turns any of the repo's architectures into an
**ordinal logistic model**: the architecture keeps its scalar ``(B, 1)`` head, and
``spacecutter``'s :class:`~spacecutter.models.LogisticCumulativeLink` converts that single
number into a probability over the ``num_classes`` *ordered* failure-urgency classes by
comparing it against ``num_classes - 1`` learned cutpoints. Training minimizes the negative
log-likelihood of the true class (:class:`~spacecutter.losses.CumulativeLinkLoss`).

Two consequences worth knowing:

- ``forward`` returns a ``(B, num_classes)`` **probability** matrix, not a class index. The
  co-training ensemble needs those probabilities to compute conformal p-values, and
  :meth:`predict_class` is provided for the places that want a hard label.
- The cutpoints must stay in ascending order for the link to define a valid distribution.
  ``spacecutter`` enforces this with a ``skorch`` callback, which is unusable here (skorch is
  not a dependency of this project), so :meth:`on_train_batch_end` reimplements the same
  clipping directly as a Lightning hook.

The RUL target standardization of ``BasicLightningModule`` is deliberately **not** carried
over: class indices are categorical positions, not a magnitude to rescale.

The logged metrics come from :mod:`models.classification.ordinal_metrics`, which encodes the
Scania 5-class taxonomy, so ``num_classes`` is expected to match
``ordinal_metrics.NUM_CLASSES``.
"""

import torch
import torch.nn as nn
from lightning import LightningModule
from spacecutter.losses import CumulativeLinkLoss, cumulative_link_loss
from spacecutter.models import OrdinalLogisticModel

from models.classification import ordinal_metrics


class OrdinalLightningModule(LightningModule):
    """Train/evaluate an architecture as an ordinal logistic classifier."""

    def __init__(
            self,
            lr: float,
            model: nn.Module,
            num_classes: int = ordinal_metrics.NUM_CLASSES,
            class_weights: torch.Tensor | None = None,
            cutpoint_margin: float = 0.0,
            cutpoint_min_val: float = -1.0e6,
    ):
        """
        Args:
            lr: Adam learning rate.
            model: The *predictor* — any ``nn.Module`` mapping ``(B, seq_len, feature_num)`` to
                ``(B, 1)`` (every architecture in ``models/classification/`` does). It is
                wrapped here, not modified; note ``OrdinalLogisticModel`` deep-copies it, so the
                instance passed in is never trained.
            num_classes: Number of ordered classes. ``spacecutter`` requires ``> 2``.
            class_weights: Optional ``(num_classes,)`` per-class loss weights, for the heavy
                class imbalance of the urgency labels. ``None`` weights every class equally.
            cutpoint_margin: Minimum gap enforced between two adjacent cutpoints by the
                ascension clipping (``spacecutter``'s ``AscensionCallback.margin``).
            cutpoint_min_val: Floor for the smallest cutpoint (``AscensionCallback.min_val``).
        """
        super().__init__()
        # ``model`` is ignored for the same reason as in BasicLightningModule (PyTorch 2.6
        # weights_only=True unpickling); ``class_weights`` is a tensor and equally unwanted in
        # the hparams blob.
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.num_classes = num_classes
        self.net = OrdinalLogisticModel(model, num_classes=num_classes)
        self.class_weights = class_weights
        self.criterion = CumulativeLinkLoss(class_weights=class_weights)
        self.lr = lr
        self.cutpoint_margin = cutpoint_margin
        self.cutpoint_min_val = cutpoint_min_val

        # Per-epoch buffers. Predictions are stored as hard class indices (not the full
        # probability matrices) so memory stays O(n_samples) rather than O(n_samples * K);
        # the loss is accumulated as a running sum instead of being recomputed at epoch end.
        self.training_step_outputs: list[torch.Tensor] = []
        self.training_step_targets: list[torch.Tensor] = []
        self.validation_step_outputs: list[torch.Tensor] = []
        self.validation_step_targets: list[torch.Tensor] = []
        self._val_loss_sum: float = 0.0
        self._val_sample_count: int = 0

    # ------------------------------------------------------------------ #
    # Forward / prediction
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the class-probability matrix.

        Args:
            x: Input windows, shape ``(B, seq_len, feature_num)``.

        Returns:
            ``(B, num_classes)`` probabilities, each row summing to 1.
        """
        return self.net(x)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Predict hard class indices.

        Args:
            x: Input windows, shape ``(B, seq_len, feature_num)``.

        Returns:
            ``(B, 1)`` ``long`` tensor of the most probable class per sample.
        """
        return self.forward(x).argmax(dim=1, keepdim=True)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_target(y: torch.Tensor) -> torch.Tensor:
        """Coerce a label batch to the ``(B, 1)`` ``long`` layout ``CumulativeLinkLoss`` gathers on.

        Args:
            y: Labels as produced by the Scania datasets — ``(B, 1)`` (or ``(B,)``) ``float``
                class indices.

        Returns:
            ``(B, 1)`` ``long`` tensor.
        """
        return y.reshape(-1, 1).long()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one training step and accumulate predictions for the epoch-end metrics."""
        x, y = batch
        probs = self.net(x)
        target = self._as_target(y)
        loss = self.criterion(probs, target)

        self.training_step_outputs.append(probs.argmax(dim=1).detach())
        self.training_step_targets.append(target.view(-1).detach())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Keep the cumulative-link cutpoints strictly ascending after every optimizer step.

        This is ``spacecutter.callbacks.AscensionCallback`` reimplemented as a Lightning hook:
        that callback subclasses ``skorch.callbacks.Callback``, and skorch is not installed in
        this project, so importing it would fail. Clipping (rather than reparameterizing) is
        what spacecutter does and what the cumulative-link formulation was fitted with.
        """
        self._clip_cutpoints()

    @torch.no_grad()
    def _clip_cutpoints(self) -> None:
        """Clamp each cutpoint below its successor (minus ``cutpoint_margin``)."""
        cutpoints = self.net.link.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(self.cutpoint_min_val, cutpoints[i + 1] - self.cutpoint_margin)

    def on_train_epoch_end(self) -> None:
        """Log train accuracy / macro-F1 over the epoch's accumulated predictions."""
        outputs = torch.cat(self.training_step_outputs)
        targets = torch.cat(self.training_step_targets)

        self.training_step_outputs.clear()
        self.training_step_targets.clear()

        self.log("train_accuracy", ordinal_metrics.accuracy(outputs, targets), prog_bar=True)
        self.log("train_f1_macro", ordinal_metrics.f1_macro(outputs, targets))

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Accumulate the validation loss and predictions for the epoch-end metrics."""
        x, y = batch
        probs = self.net(x)
        target = self._as_target(y)

        # Sum-reduced (the functional form, so the class weights still apply) so batches of
        # unequal size are averaged correctly at epoch end.
        loss = cumulative_link_loss(
            probs, target, reduction="sum", class_weights=self.class_weights)
        self._val_loss_sum += float(loss.detach())
        self._val_sample_count += target.shape[0]

        self.validation_step_outputs.append(probs.argmax(dim=1).detach())
        self.validation_step_targets.append(target.view(-1).detach())

    def on_validation_epoch_end(self) -> None:
        """Log ``val_loss`` plus the ordinal metrics.

        ``val_loss`` is mandatory, not cosmetic: ``run_training_job`` /
        ``run_finetune_job`` hard-code ``monitor="val_loss"`` for both ``EarlyStopping`` and
        ``ModelCheckpoint``, so the whole best-model selection of the co-training loop hangs
        off this one value.
        """
        outputs = torch.cat(self.validation_step_outputs)
        targets = torch.cat(self.validation_step_targets)

        val_loss = self._val_loss_sum / max(self._val_sample_count, 1)

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
        self._val_loss_sum = 0.0
        self._val_sample_count = 0

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", ordinal_metrics.accuracy(outputs, targets), prog_bar=True)
        self.log("val_f1_macro", ordinal_metrics.f1_macro(outputs, targets))
        self.log("val_mae", ordinal_metrics.mae(outputs, targets))
        self.log("val_rmse", ordinal_metrics.rmse(outputs, targets))
        self.log("val_cost", ordinal_metrics.scania_cost(outputs, targets))

    # ------------------------------------------------------------------ #
    # Optimizer / inference
    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return an Adam optimizer over the wrapped model *and* the cutpoints.

        ``self.net.parameters()`` deliberately covers ``net.link.cutpoints`` too — the
        cutpoints are learned jointly with the predictor.
        """
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def predict_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a ``(features, target)`` batch.

        Args:
            batch: ``(features, target)`` from the predict dataloader.
            batch_idx: Batch index (unused, required by Lightning's signature).

        Returns:
            ``(probabilities, targets)``.
        """
        x, y = batch
        return self.net(x), y
