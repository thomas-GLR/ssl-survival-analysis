import copy
import warnings
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
from enum import Enum

from lightning import LightningModule, Trainer
from torch.utils.data import TensorDataset, DataLoader


class SelectionMode(Enum):
    VOTING = 1
    EVIDENCE = 2


class CoTrainingEnsemble_v2:
    """
    This is the second version of the co training ensemble. This version have a higher computational cost.
    """

    def __init__(
            self,
            models: list[nn.Module],
            weights: list[float] | None = None,
            verbose: int = 0,
            fine_tune_lr_factor: float = 0.1,
            forgetting_warning_tolerance: float = 0.0,
    ):
        """
        :param models: list[nn.Module]
            The models that will be used in the co-training ensemble.
        :param weights: list[float] | None
            Optional pre-defined weights for each model.
        :param verbose: int
            Verbosity level. 0 = silent, 1 = key decisions, 2 = full per-candidate details.
        :param fine_tune_lr_factor: float
            Multiplier applied to a model's own learning rate while it is being
            fine-tuned (see ``_fine_tune_fun``), so an already-trained model is
            nudged rather than overridden by the newly added censored data.
        :param forgetting_warning_tolerance: float
            Relative tolerance (e.g. ``0.05`` = 5%) allowed for the validation MSE
            to increase after a fine-tuning call before a forgetting warning is
            raised. ``0.0`` (default) warns on any increase.
        """
        if weights is not None and len(models) != len(weights):
            raise ValueError("The number of weights must be the same as the number of models.")

        if not (0 < fine_tune_lr_factor <= 1):
            raise ValueError("fine_tune_lr_factor must be in the range (0, 1].")

        if forgetting_warning_tolerance < 0:
            raise ValueError("forgetting_warning_tolerance must be >= 0.")

        self.models = models
        self.number_of_models = len(self.models)
        self.lightning_modules = None
        self.trainer_factories = None
        self.batchs_size = None
        self.shuffle_dataloaders = None
        self.fine_tune_trainable_params = None
        self.weights = weights
        self.verbose = verbose
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.forgetting_warning_tolerance = forgetting_warning_tolerance

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message)

    def setup_training(
            self,
            lightning_modules: list[LightningModule],
            trainer_factories: list[Callable[[], Trainer]],
            batchs_size: list[int],
            shuffle_dataloaders: list[bool],
            fine_tune_trainable_params: list[Callable[[LightningModule], list[nn.Parameter]] | None] | None = None,
    ) -> None:
        r"""Setup training for the models.

        Args:
            lightning_modules (list[LightningModule]): The lightning modules that will be used to train
                the models. Each lightning module will be used to train one model. You need to keep
                the same order for each model.
            trainer_factories (list[Callable[[], Trainer]]): The trainer factories that will be used
                to construct the trainer to train the models. Each trainer will be used to train one model.
                You need to keep the same order for each model.

                Example:
                    trainer_factories: list[Callable[[], Trainer]] = [
                        lambda: Trainer(max_epochs=10, accelerator="gpu"),
                        lambda: Trainer(max_epochs=10, accelerator="gpu"),
                    ]
            batchs_size (list[int]): The batch size that will be used to train the models.
                Each batch size will be used to train one model.
            shuffle_dataloaders (list[bool]): The shuffle dataloader that will be used to train
                the models. Each shuffle dataloader will be used to train one model.
            fine_tune_trainable_params (list[Callable[[LightningModule], list[nn.Parameter]] | None] | None):
                Optional, one entry per model. Each entry is either ``None`` (fine-tune all
                parameters, the default) or a callable that, given the model being fine-tuned,
                returns the subset of its parameters that should stay trainable — every other
                parameter is frozen (``requires_grad=False``) for the duration of that
                fine-tuning call. Only affects ``_fine_tune_fun``; ``_train_fun`` always trains
                every parameter. Only used when the ensemble is actually fine-tuned (see
                ``train``'s ``is_fine_tuning_during_finding_best_suspension_data`` /
                ``is_fine_tuning_for_last_step``).

                Example, freezing everything but the regression head of a ``CNN1D``-backed
                ``TransformerLstmModule``:
                    fine_tune_trainable_params = [
                        lambda lm: list(lm.net.regressor.parameters()),
                    ] * models_number
        """
        if (len(lightning_modules) != len(self.models) or
                len(trainer_factories) != len(self.models) or
                len(shuffle_dataloaders) != len(self.models) or
                len(batchs_size) != len(self.models)):
            raise ValueError(
                f"The number of lightning modules (size={len(lightning_modules)}), trainers (size={len(trainer_factories)}), shuffle_dataloaders (size={len(shuffle_dataloaders)}), batchs_size (size={len(batchs_size)}) must be the same as the number of models.")

        if fine_tune_trainable_params is not None and len(fine_tune_trainable_params) != len(self.models):
            raise ValueError(
                f"The number of fine_tune_trainable_params (size={len(fine_tune_trainable_params)}) must be the same as the number of models.")

        self.lightning_modules = lightning_modules
        self.trainer_factories = trainer_factories
        self.batchs_size = batchs_size
        self.shuffle_dataloaders = shuffle_dataloaders
        self.fine_tune_trainable_params = fine_tune_trainable_params

    def train(
            self,
            is_fine_tuning_during_finding_best_suspension_data: bool,
            is_fine_tuning_for_last_step: bool,
            selection_mode: SelectionMode,
            train_with_censored_data: bool,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: int,
            val_data: torch.Tensor | None = None,
            val_label: torch.Tensor | None = None,
    ) -> None:
        r"""The train algorithm for co-training ensemble v2

        Args:
            is_fine_tuning_during_finding_best_suspension_data:
                - True : when iterating over censored data the models will be fine-tuned with the new censored data
                    instead of training from scratch.
                - False : the models will be trained from scratch with the new censored data.
            is_fine_tuning_for_last_step:
                - True : the models will be fine-tuned after censored data is selected
                    instead of training from scratch.
                - False : the models will be trained from scratch after censored data is selected.
            selection_mode:
                - VOTING :
                - EVIDENCE :
            train_with_censored_data:
                - True : the training of the models will be done with censored data. The model need to be able to handle censored data.
                - False : the training of the models will be done only with failure data.
            failure_data:
                The features of failure data.
            failure_label:
                The target of failure data.
            suspension_data:
                The suspension data.
            suspension_ids:
                The ids of each different individual who are censored.
            iterations:
                The number of iteration for training models on suspension data
            suspension_pool_size:
                The number of suspension data selected for each iteration
            val_data:
                Optional validation features used for early stopping / best-checkpoint
                selection during every training call.
            val_label:
                Optional validation labels associated with ``val_data``. Must be provided
                together with ``val_data``.
        """
        self._check_if_training_is_possible()

        if (val_data is None) != (val_label is None):
            raise ValueError("val_data and val_label must both be provided or both be None.")

        total_suspension_units = len(torch.unique(suspension_ids))
        self._log(1, f"[CoTraining] Starting training | models: {self.number_of_models} | "
                     f"failure samples: {len(failure_data)} | "
                     f"censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | pool size: {suspension_pool_size} | "
                     f"selection: {selection_mode.name}")

        models_datasets = []
        h: list[LightningModule] = []

        for j in range(self.number_of_models):
            x_i, y_i = failure_data, failure_label

            # TODO need to see how to deel with survloss and data
            # if train_with_censored_data:
            #     x_i = torch.cat([x_i, suspension_data], dim=0)
            #     y_i = torch.cat([y_i, ], dim=0)

            models_datasets.append((x_i, y_i))

            self._log(1, f"[CoTraining] Initial training of model {j} on {len(x_i)} failure samples...")
            h_j = self._train_fun(
                model=copy.deepcopy(self.lightning_modules[j]),
                model_index=j,
                x=x_i,
                y=y_i,
                val_x=val_data,
                val_y=val_label,
            )

            h.append(h_j)

        self._log(1, f"[CoTraining] Initial training done.")

        remaining_suspension_ids = torch.unique(suspension_ids)

        for i in range(iterations):
            if len(remaining_suspension_ids) == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i}: no remaining censored units.")
                break

            pool_size = min(suspension_pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
            pool_ids = shuffled_ids[:pool_size]  # U'

            self._log(1, f"[CoTraining] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            # Phase 1 — for each model j, predict pseudo-labels and compute delta for every
            # unit in the pool. Results are stored in an OrderedDict (sorted by delta desc)
            # so the best candidate is always first.
            # Structure: all_preds[j] = OrderedDict{ unit_id_int -> (unit_id, xu, lu_p, delta) }
            all_preds: dict[int, OrderedDict] = {}

            for j in range(self.number_of_models):
                hj = h[j]
                xj, yj = models_datasets[j]
                candidates = []

                self._log(2, f"[CoTraining]   Model {j}: evaluating {pool_size} candidates...")

                for unit_idx, unit_id in enumerate(pool_ids):
                    mask = (suspension_ids == unit_id)
                    xu = suspension_data[mask]

                    lu_p = self._predict(hj, xu)
                    lu_p_reshaped = lu_p.view(-1, yj.shape[1]) if yj.dim() > 1 else lu_p.view(-1)

                    if is_fine_tuning_during_finding_best_suspension_data:
                        # Fine-tuning warm-starts from hj, so only the new candidate's
                        # data needs to be passed in — the whole point of fine-tuning
                        # here is to avoid re-training on the full accumulated dataset.
                        hj_prime = self._fine_tune_fun(
                            model=copy.deepcopy(hj),
                            model_index=j,
                            x=xu,
                            y=lu_p_reshaped,
                            val_x=val_data,
                            val_y=val_label,
                        )
                    else:
                        x_augmented = torch.cat([xj, xu], dim=0)
                        y_augmented = torch.cat([yj, lu_p_reshaped], dim=0)
                        hj_prime = self._train_fun(
                            model=copy.deepcopy(self.lightning_modules[j]),
                            model_index=j,
                            x=x_augmented,
                            y=y_augmented,
                            val_x=val_data,
                            val_y=val_label,
                        )

                    delta = self._confidence_measure(xj, yj, hj, hj_prime)
                    candidates.append((unit_id, xu, lu_p, delta))
                    self._log(2, f"[CoTraining]     candidate {unit_idx + 1}/{len(pool_ids)} unit {unit_id.item()}: delta = {delta:.4f}")

                candidates.sort(key=lambda e: e[3], reverse=True)
                all_preds[j] = OrderedDict(
                    (uid.item(), (uid, xu, lu_p, delta))
                    for uid, xu, lu_p, delta in candidates
                )

                if self.verbose >= 2:
                    ranking = [(uid.item(), d) for uid, _, _, d in candidates]
                    self._log(2, f"[CoTraining]   Model {j} candidate ranking (best first): "
                                 f"{[(uid, round(d, 4)) for uid, d in ranking]}")

            # Phase 2 — for each model k, pick the best available candidate using the
            # selection mode. When a candidate is chosen it is removed from every model's
            # map so no other model k can reuse the same suspension unit.
            censored_data_selected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
                None for _ in range(self.number_of_models)
            ]

            for k in range(self.number_of_models):
                match selection_mode:
                    case SelectionMode.VOTING:
                        censored_data_selected[k] = self._voting_censored_data_selection(
                            all_preds=all_preds,
                            model_index_to_exclude=k,
                        )
                    case SelectionMode.EVIDENCE:
                        censored_data_selected[k] = self._evidential_censored_data_selection()
                    case _:
                        raise ValueError(f"Unknown selection mode: {selection_mode.name}")

                if censored_data_selected[k] is not None:
                    selected_id = censored_data_selected[k][0].item()
                    self._log(1, f"[CoTraining]   Model {k}: selected unit {selected_id} "
                                 f"(mode: {selection_mode.name})")

                    # Remove the selected unit from every model's candidate map so it
                    # cannot be assigned to another model in this iteration.
                    for j in range(self.number_of_models):
                        all_preds[j].pop(selected_id, None)

                    remaining_suspension_ids = remaining_suspension_ids[
                        remaining_suspension_ids != censored_data_selected[k][0]
                        ]
                else:
                    self._log(1, f"[CoTraining]   Model {k}: no unit selected (no positive delta found).")

            if all(x is None for x in censored_data_selected):
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: "
                             f"no model found a beneficial censored unit.")
                break

            for j in range(self.number_of_models):
                if censored_data_selected[j] is not None:
                    _, xu, lu = censored_data_selected[j]
                    xj, yj = models_datasets[j]

                    lu2_reshaped = lu.view(-1, yj.shape[1]) if yj.dim() > 1 else lu.view(-1)

                    xj = torch.cat([xj, xu], dim=0)
                    yj = torch.cat([yj, lu2_reshaped], dim=0)

                    models_datasets[j] = (xj, yj)

                    self._log(1, f"[CoTraining]   Retraining model {j} | "
                                 f"dataset size: {len(xj)} samples "
                                 f"({'fine-tune' if is_fine_tuning_for_last_step else 'from scratch'})")

                    if is_fine_tuning_for_last_step:
                        # Fine-tuning warm-starts from h[j], so only the newly selected
                        # unit's data is passed in, not the full accumulated (xj, yj) —
                        # that's what makes fine-tuning cheaper than retraining from scratch.
                        h[j] = self._fine_tune_fun(
                            model=copy.deepcopy(h[j]),
                            model_index=j,
                            x=xu,
                            y=lu2_reshaped,
                            val_x=val_data,
                            val_y=val_label,
                        )
                    else:
                        h[j] = self._train_fun(
                            model=copy.deepcopy(self.lightning_modules[j]),
                            model_index=j,
                            x=xj,
                            y=yj,
                            val_x=val_data,
                            val_y=val_label,
                        )

        self._log(1, f"[CoTraining] Training complete.")
        self.lightning_modules = h

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction: L^P = w1*h1(x) + w2*h2(x) + ...

        :param x: Shape (N, *feature_dims) for multiple samples, or (*feature_dims,) for a single sample.

        :return Tensor of shape (N, *output_dims) for multiple samples, or (*output_dims,) for a single sample.
        """
        if self.weights is None:
            raise ValueError(
                "Weights are not provided. Provide them when instanciate the class or train the ensemble first.")

        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)

        weighted_preds = []
        for j, model in enumerate(self.lightning_modules):
            pred = self._predict(model, x)  # shape (N, *output_dims)
            weighted_preds.append(pred * self.weights[j])

        result = torch.stack(weighted_preds, dim=0).sum(dim=0)  # shape (N, *output_dims)

        if single_sample:
            result = result.squeeze(0)

        return result

    def _train_fun(
            self,
            model: LightningModule,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None = None,
            val_y: torch.Tensor | None = None,
    ) -> LightningModule:
        """
        Train the lightning module for the given index with the given data.

        If a validation set is provided it is used for early stopping / checkpointing.
        After ``trainer.fit`` the best checkpoint (as tracked by the trainer's
        ``ModelCheckpoint`` callback) is reloaded so we never keep the potentially
        worse last-epoch weights.

        :param model: LightningModule
            The lightning module to train.
        :param model_index: int
            The index of the model to train.
        :param x: torch.Tensor
            The data features.
        :param y: torch.Tensor
            The data labels.
        :param val_x: torch.Tensor | None
            Optional validation features used for early stopping / best-checkpoint selection.
        :param val_y: torch.Tensor | None
            Optional validation labels associated with ``val_x``.
        :return: LightningModule
            The trained lightning module.
        """
        return self._fit_and_reload_best(model, model_index, x, y, val_x, val_y)

    def _fit_and_reload_best(
            self,
            model: LightningModule,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None,
            val_y: torch.Tensor | None,
    ) -> LightningModule:
        """
        Fit ``model`` (whatever its current weights are) on ``(x, y)`` and reload the
        best checkpoint (as tracked by the trainer's ``ModelCheckpoint`` callback) so
        we never keep the potentially worse last-epoch weights. Shared by
        ``_train_fun`` (fresh model) and ``_fine_tune_fun`` (warm-started model).
        """
        # Trainer can save some state then we create a new one
        trainer: Trainer = self.trainer_factories[model_index]()

        batch_size: int = self.batchs_size[model_index]

        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=self.shuffle_dataloaders[model_index])

        val_loader = None
        if val_x is not None and val_y is not None:
            val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Reload the best checkpoint (based on the monitored validation metric) so we
        # use the best model after training instead of the last-epoch weights.
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", "") if checkpoint_callback else ""
        if best_model_path:
            self._log(2, f"[CoTraining]     Reloading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=model.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            self._log(2, f"[CoTraining]     No best model found, the model with last epoch is used")

        return model

    def _fine_tune_fun(
            self,
            model: LightningModule,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None = None,
            val_y: torch.Tensor | None = None,
    ) -> LightningModule:
        """
        Fine-tune the lightning module for the given index with the given data.

        Unlike ``_train_fun``, ``model`` is expected to already carry the weights
        from a previous training/fine-tuning call (the co-training loop passes in a
        deep copy of the current model, not a fresh one). This continues training
        those weights instead of reinitializing them, using a reduced learning rate
        (``self.fine_tune_lr_factor``) so the update is a nudge rather than a full
        override.

        :param model: LightningModule
            The already-trained lightning module to fine-tune in place.
        :param model_index: int
            The index of the model to fine-tune.
        :param x: torch.Tensor
            The data features.
        :param y: torch.Tensor
            The data labels.
        :param val_x: torch.Tensor | None
            Optional validation features used for early stopping / best-checkpoint
            selection, and to check for forgetting (see below).
        :param val_y: torch.Tensor | None
            Optional validation labels associated with ``val_x``.
        :return: LightningModule
            The fine-tuned lightning module.

        Note on freezing: if ``setup_training`` was given a
        ``fine_tune_trainable_params`` entry for this model index, every
        parameter *not* returned by that callable is frozen
        (``requires_grad=False``) for the duration of this call — e.g. to keep
        a backbone fixed and only adapt the head when fine-tuning on a single
        new data point. Frozen parameters get no gradient, so the optimizer
        built by ``configure_optimizers()`` simply skips them; nothing about
        the ``LightningModule`` itself needs to change.
        """
        original_lr = getattr(model, "lr", None)
        if original_lr is not None:
            model.lr = original_lr * self.fine_tune_lr_factor

        trainable_params_fn = (
            self.fine_tune_trainable_params[model_index]
            if self.fine_tune_trainable_params is not None
            else None
        )
        original_requires_grad = self._apply_freeze(model, trainable_params_fn)

        # Only the new unit is used for training (see the call sites in `train`),
        # so there is no rehearsal of previously-seen data to protect against
        # catastrophic forgetting beyond the reduced LR, the optional freezing
        # above, and the best-checkpoint reload below. Snapshot the pre-fine-tune
        # validation MSE so we can warn if the fine-tuned model ends up
        # generalizing worse than before.
        has_val_data = val_x is not None and val_y is not None
        val_mse_before = self._mse_on(model, val_x, val_y) if has_val_data else None

        try:
            model = self._fit_and_reload_best(model, model_index, x, y, val_x, val_y)
        finally:
            if original_lr is not None:
                model.lr = original_lr
            self._restore_freeze(original_requires_grad)

        if has_val_data:
            val_mse_after = self._mse_on(model, val_x, val_y)
            if val_mse_after > val_mse_before * (1 + self.forgetting_warning_tolerance):
                message = (
                    f"[CoTraining] Possible forgetting detected for model {model_index}: "
                    f"validation MSE went from {val_mse_before:.4f} to {val_mse_after:.4f} "
                    f"after fine-tuning on {len(x)} new sample(s)."
                )
                warnings.warn(message)
                self._log(1, message)

        return model

    def _apply_freeze(
            self,
            model: LightningModule,
            trainable_params_fn: Callable[[LightningModule], list[nn.Parameter]] | None,
    ) -> dict[nn.Parameter, bool] | None:
        """
        Freeze every parameter of ``model`` except the ones returned by
        ``trainable_params_fn``. Returns the pre-freeze ``requires_grad`` state
        (to be restored via ``_restore_freeze``), or ``None`` if
        ``trainable_params_fn`` is ``None`` (no freezing).
        """
        if trainable_params_fn is None:
            return None

        trainable_params = set(trainable_params_fn(model))
        original_requires_grad = {p: p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad = p in trainable_params

        return original_requires_grad

    def _restore_freeze(self, original_requires_grad: dict[nn.Parameter, bool] | None) -> None:
        """Undo ``_apply_freeze``, restoring each parameter's original ``requires_grad``."""
        if original_requires_grad is None:
            return

        for p, requires_grad in original_requires_grad.items():
            p.requires_grad = requires_grad

    def _mse_on(self, model: LightningModule, x: torch.Tensor, y: torch.Tensor) -> float:
        """Mean squared error of ``model`` on ``(x, y)``, used for the forgetting check."""
        y_flat = y.view(-1)
        pred_flat = self._predict(model, x).view(-1)
        return ((y_flat - pred_flat) ** 2).mean().item()

    def _predict(
            self,
            model: LightningModule,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the output for the given model index and input data.

        :param model: LightningModule
            The lightning module to use for prediction.
        :param x: torch.Tensor
            The input data features.
        :return: torch.Tensor
            The predicted output.
        """
        model.eval()

        with torch.no_grad():
            x = x.to(next(model.parameters()).device)

            predictions = model(x)

        return predictions

    def _confidence_measure(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            original_model: LightningModule,
            augmented_model: LightningModule,
    ) -> float:
        r"""
        Δ_{j, X_u} from line 9:
            Σ(L_i − h_j(x_i))²  −  Σ(L_i − h'_j(x_i))²

        A positive value means the augmented model is *better* on the labelled
        set, so adding X_u was beneficial.

        :param x:               Labelled failure inputs.
        :param y:               Corresponding true labels.
        :param original_model:  h_j  – model before adding the suspension sample.
        :param augmented_model: h'_j – model after  adding the suspension sample.

        :return Scalar float (positive ⟹ improvement).
        """
        y_flat = y.view(-1)

        pred_orig = self._predict(original_model, x).view(-1)
        pred_aug = self._predict(augmented_model, x).view(-1)

        mse_orig = ((y_flat - pred_orig) ** 2).sum().item()
        mse_aug = ((y_flat - pred_aug) ** 2).sum().item()

        return mse_orig - mse_aug  # > 0 means augmented model is better

    def _voting_censored_data_selection(
            self,
            all_preds: dict[int, OrderedDict],
            model_index_to_exclude: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        r"""Select the censored unit that best improves the ensemble on average.

        For each candidate unit u, the average delta is computed over all models
        j ≠ k (including negative deltas, which penalise units that hurt some
        models). The unit with the highest average delta is selected, provided
        that average is strictly positive.

        The pseudo-label assigned to model k is the prediction made by the model
        j ≠ k that achieved the highest individual delta for the selected unit
        (i.e. the most confident predictor for that specific unit).

        Args:
            all_preds: mapping from model index j to an OrderedDict of
                ``{unit_id_int: (unit_id_tensor, xu, lu_p, delta)}``.
            model_index_to_exclude: index k of the model being updated — its
                own predictions are excluded from both the average and the
                pseudo-label selection.

        Returns:
            ``(unit_id, xu, lu_p)`` for the selected unit, or ``None`` if no
            unit has a strictly positive average delta.
        """
        # Collect every unit_id that at least one non-excluded model has scored.
        all_unit_ids: set[int] = {
            uid
            for j, preds in all_preds.items()
            if j != model_index_to_exclude
            for uid in preds
        }

        best_avg_delta = 0.0  # strictly > 0 required to accept a candidate
        best_unit_id_int: int | None = None

        for unit_id_int in all_unit_ids:
            deltas = [
                all_preds[j][unit_id_int][3]
                for j in all_preds
                if j != model_index_to_exclude and unit_id_int in all_preds[j]
            ]
            if not deltas:
                continue
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta > best_avg_delta:
                best_avg_delta = avg_delta
                best_unit_id_int = unit_id_int

        if best_unit_id_int is None:
            return None

        # Among models j ≠ k, pick the one with the highest individual delta
        # for this unit — its prediction is the most reliable pseudo-label.
        best_j = max(
            (j for j in all_preds if j != model_index_to_exclude and best_unit_id_int in all_preds[j]),
            key=lambda j: all_preds[j][best_unit_id_int][3],
        )

        unit_id, xu, lu_p, _ = all_preds[best_j][best_unit_id_int]
        return unit_id, xu, lu_p

    def _evidential_censored_data_selection(self):
        raise NotImplementedError("Evidential censored data selection is not implemented yet.")

    def _check_if_training_is_possible(self):
        if (self.lightning_modules is None
                or self.trainer_factories is None
                or self.batchs_size is None
                or self.shuffle_dataloaders is None):
            raise ValueError("You need to call setup_training before calling train.")

    def calculate_weights(
            self,
            x_test: torch.Tensor,
            target: torch.Tensor,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float],
            mode: str,
    ):
        """

        Args:
            x_test:
            target:
            criteria_callback:
            mode: value can be "min" or "max".
                "min" mean that more the score is little more the model is good.
                "max" mean that more the score is high more the model is good.

        Returns:

        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")

        scores = []

        # Flatten both sides so the criteria callback compares aligned (N,) vectors
        # instead of broadcasting (N, 1) against (N,) into an (N, N) matrix.
        target_flat = target.view(-1).float()

        for model in self.lightning_modules:
            pred = self._predict(model, x_test).view(-1).to(target_flat.device)
            scores.append(criteria_callback(pred, target_flat))

        self._log(1, f"[CoTraining] Calculating weights (mode={mode}) | "
                     f"scores per model: {[round(s, 4) for s in scores]}")

        if mode == "min":
            if any(s == 0 for s in scores):
                raise ValueError(
                    "At least one model has a score of zero in 'min' mode, inverse weighting is undefined.")
            inv_scores = [1.0 / s for s in scores]
            total = sum(inv_scores)
            self.weights = [inv_s / total for inv_s in inv_scores]
        else:
            total = sum(scores)
            if total == 0:
                raise ValueError(f"The sum of scores from all models is zero, cannot calculate weights : {scores}")
            self.weights = [s / total for s in scores]

        self._log(1, f"[CoTraining] Weights assigned: "
                     f"{[f'model {j}={round(w, 4)}' for j, w in enumerate(self.weights)]}")
