import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics.functional import mean_squared_error


def _cmapss_score(predict: np.ndarray, label: np.ndarray) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)


class BasicLightningModule(LightningModule):
    def __init__(
            self,
            lr,
            model: nn.Module,
            target_mean: float = 0.0,
            target_std: float = 1.0,
            use_cotraining_ensemble_survival_loss_function: bool = False,
            cotraining_survival_loss_lambda: float = 1.0,
    ):
        super(BasicLightningModule, self).__init__()
        # We need to ignore model to prevent UnpicklingError since PyTorch 2.6 with weights_only=True
        self.save_hyperparameters(ignore=['model'])
        self.net = model
        self.lr = lr
        # CoTrainingEnsemble_v2-only (Scania) survival-loss selection for 4-element batches
        # (x, y, is_censored, lower_bound). False/default keeps every other caller (COPROG,
        # CoTrainingEnsemble v1, RSF, C-MAPSS, HPO) on the plain 2-element (x, y) MSE path
        # untouched. use_cotraining_ensemble_survival_loss_function is flipped post-construction
        # by CoTrainingEnsemble_v2's fine-tune job (models/cotraining_gpu_pool.run_finetune_job),
        # mirroring how that same job already overrides self.lr for the fine-tune's reduced
        # learning rate -- the shared module_builder partial always constructs with the default
        # (False), so Initial training and scratch-retrains are never affected by it.
        self.use_cotraining_ensemble_survival_loss_function = use_cotraining_ensemble_survival_loss_function
        self.cotraining_survival_loss_lambda = cotraining_survival_loss_lambda
        # RUL target standardization. The network learns/predicts in normalized
        # target space (loss is computed there), which keeps the MSE gradient on
        # an O(1) scale so the optimizer isn't dominated by the raw target
        # magnitude (mean ~117, std ~84). Predictions are de-normalized back to
        # real RUL units everywhere they are stored / returned, so all reported
        # metrics and saved predictions stay in original units. The defaults
        # (0, 1) make this a no-op for callers that don't pass stats.
        # These are hyperparameters (saved via save_hyperparameters) so they are
        # restored on load_from_checkpoint, which matters because predictions are
        # generated after reloading the best checkpoint.
        self.target_mean = target_mean
        self.target_std = target_std
        # On CMAPSS storing outputs is not a big problem as there is not too much data.
        # But on bigest dataset there is a risk of memory overflow
        self.training_step_outputs = []
        self.training_step_targets = []
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        # forward returns predictions in REAL RUL units (de-normalized). The
        # network itself learns in normalized target space (training_step feeds
        # self.net directly and normalizes y), but external callers of the module
        # -- notably Coprog._predict, which calls module(x) -> forward -- expect
        # real-unit predictions to stay consistent with the raw labels/targets
        # they compare against. The plain Lightning path never calls forward for
        # inference (it uses predict_step / test_step, which de-normalize too).
        return self._denorm(self.net(x))

    def _denorm(self, t: torch.Tensor) -> torch.Tensor:
        """De-normalize a prediction from standardized target space back to real RUL units.

        :param t: tensor of predictions in normalized target space.
        :return: tensor of predictions in original RUL units.
        """
        return t * self.target_std + self.target_mean

    def training_step(self, batch, batch_idx):
        # CoTrainingEnsemble_v2 (Scania only) feeds a 4-element batch (x, y, is_censored,
        # lower_bound) for Initial training with censored data and for cotraining-loss
        # fine-tuning; every other caller keeps the plain (x, y) 2-element batch below.
        if len(batch) == 4:
            x, y, is_censored, lower_bound = batch
            preds = self.net(x)
            y_norm = (y - self.target_mean) / self.target_std
            lower_bound_norm = (lower_bound - self.target_mean) / self.target_std
            if self.use_cotraining_ensemble_survival_loss_function:
                loss = self._cotraining_ensemble_survival_loss_function(
                    preds, y_norm, is_censored, lower_bound_norm)
            else:
                loss = self._survival_loss_function(preds, y_norm, is_censored, lower_bound_norm)
            # Censored rows carry either a dummy placeholder target (Initial training) or an
            # approximate pseudo-label (fine-tuning) -- neither belongs in the ground-truth
            # train_rmse, so only failed rows are tracked for that metric.
            failed_mask = ~is_censored.view(-1).bool()
            self.training_step_outputs.extend(self._denorm(preds).detach()[failed_mask])
            self.training_step_targets.extend(y.detach()[failed_mask])
            return loss

        x, y = batch
        preds = self.net(x)
        # Loss is computed in normalized target space; predictions are stored
        # de-normalized so train_rmse stays in real RUL units.
        y_norm = (y - self.target_mean) / self.target_std
        loss = F.mse_loss(preds, y_norm)
        self.training_step_outputs.extend(self._denorm(preds).detach())
        self.training_step_targets.extend(y.detach())
        return loss

    def _survival_loss_function(
            self,
            preds: torch.Tensor,
            y_norm: torch.Tensor,
            is_censored: torch.Tensor,
            lower_bound_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Survival loss for CoTrainingEnsemble_v2 Initial training with censored data.

        Failed (uncensored) rows contribute their squared error against the true (normalized)
        target. Censored rows have no ground truth yet (``y_norm`` is a dummy placeholder for
        them, never read) -- they only contribute a squared error against their survival lower
        bound, and only when the prediction falls below that bound; a prediction at or above the
        bound is already consistent with the data and the row is dropped.

        Args:
            preds: Model predictions in normalized target space, shape ``(N,)`` or ``(N, 1)``.
            y_norm: Normalized targets, meaningful for failed rows only.
            is_censored: Per-row censoring flag (``True``/``1`` for censored rows).
            lower_bound_norm: Normalized survival lower bound per row, meaningful for censored
                rows only.

        Returns:
            Scalar loss tensor, graph-connected to ``preds`` even when no row contributes
            (all-censored-above-bound batch).
        """
        preds_flat = preds.view(-1)
        y_flat = y_norm.view(-1)
        lb_flat = lower_bound_norm.view(-1)
        censored_mask = is_censored.view(-1).bool()
        failed_mask = ~censored_mask

        failed_se = (preds_flat[failed_mask] - y_flat[failed_mask]) ** 2

        censored_preds = preds_flat[censored_mask]
        censored_lb = lb_flat[censored_mask]
        violates = censored_preds < censored_lb
        censored_se = (censored_preds[violates] - censored_lb[violates]) ** 2

        total_n = failed_se.numel() + censored_se.numel()
        if total_n == 0:
            return (preds_flat * 0.0).sum()
        return (failed_se.sum() + censored_se.sum()) / total_n

    def _cotraining_ensemble_survival_loss_function(
            self,
            preds: torch.Tensor,
            y_norm: torch.Tensor,
            is_censored: torch.Tensor,
            lower_bound_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Survival loss for CoTrainingEnsemble_v2 fine-tuning iterations.

        Unlike Initial training, every censored row here already carries a real pseudo-label
        (assigned by a peer model during unit selection), never a placeholder. Failed rows
        contribute a plain MSE term against their true target. Censored rows contribute a
        ``cotraining_survival_loss_lambda``-weighted MSE term against their pseudo-label, plus an
        extra MSE-against-lower-bound term computed only over the censored rows whose prediction
        falls below their bound -- non-violating censored rows are already fully accounted for by
        the pseudo-label term, so the bound term is averaged over the violating subset only.

        Args:
            preds: Model predictions in normalized target space, shape ``(N,)`` or ``(N, 1)``.
            y_norm: Normalized targets -- true target for failed rows, pseudo-label for censored
                rows.
            is_censored: Per-row censoring flag (``True``/``1`` for censored rows).
            lower_bound_norm: Normalized survival lower bound per row, meaningful for censored
                rows only.

        Returns:
            Scalar loss tensor, graph-connected to ``preds`` even when a term has no rows.
        """
        preds_flat = preds.view(-1)
        y_flat = y_norm.view(-1)
        lb_flat = lower_bound_norm.view(-1)
        censored_mask = is_censored.view(-1).bool()
        failed_mask = ~censored_mask

        loss = (preds_flat * 0.0).sum()

        if failed_mask.any():
            loss = loss + F.mse_loss(preds_flat[failed_mask], y_flat[failed_mask])

        if censored_mask.any():
            censored_preds = preds_flat[censored_mask]
            censored_y = y_flat[censored_mask]
            loss = loss + self.cotraining_survival_loss_lambda * F.mse_loss(censored_preds, censored_y)

            censored_lb = lb_flat[censored_mask]
            violates = censored_preds < censored_lb
            if violates.any():
                loss = loss + F.mse_loss(censored_preds[violates], censored_lb[violates])

        return loss

    def on_train_epoch_end(self):
        outputs = torch.stack(self.training_step_outputs)
        targets = torch.stack(self.training_step_targets)

        rmse = mean_squared_error(outputs, targets, squared=False)

        self.training_step_outputs.clear()
        self.training_step_targets.clear()

        self.log('train_rmse', rmse, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)
        # Store de-normalized predictions so val_rmse / val_loss (monitored by
        # early stopping and checkpointing) stay in real RUL units.
        self.validation_step_outputs.extend(self._denorm(preds).detach())
        self.validation_step_targets.extend(y.detach())

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y = batch
        preds = self._denorm(self.net(x))
        self.test_step_outputs.extend(preds)
        self.test_step_targets.extend(y)

    def on_test_epoch_end(self):
        outputs = torch.tensor(self.test_step_outputs)
        targets = torch.tensor(self.test_step_targets)

        rmse = mean_squared_error(
            outputs,
            targets,
            squared=False
        )

        np_outputs, np_targets = outputs.cpu().numpy(), targets.cpu().numpy()

        score = _cmapss_score(np_outputs, np_targets)

        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        self.log('test_rmse', rmse)
        self.log('test_score', score)

    def on_validation_epoch_end(self):
        outputs = torch.stack(self.validation_step_outputs)
        targets = torch.stack(self.validation_step_targets)

        rmse = mean_squared_error(outputs, targets, squared=False)
        score = _cmapss_score(outputs.cpu().numpy().flatten(), targets.cpu().numpy().flatten())

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

        self.log('val_loss', rmse ** 2, prog_bar=True)
        self.log('val_rmse', rmse)
        self.log('val_score', score)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        """Run inference on a (features, target) batch and return both.

        :param batch: tuple of (features, target) tensors from the predict dataloader.
        :param batch_idx: index of the batch (unused, required by Lightning's signature).
        :return: tuple of (predictions, targets) tensors.
        """
        x, y = batch
        preds = self._denorm(self.net(x))
        return preds, y
