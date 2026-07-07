import copy
from typing import Callable

import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


class Coprog:
    """
    Co-training-based PROGnostics (COPROG) algorithm.

    Reference: "A co-training-based approach for prediction of remaining useful
    life utilizing both failure and suspension data." Chao Hu, Byeng D. Youn, Taejin Kim, Pingfeng Wang.

    Two models are trained on complementary views of the failure data. At each
    iteration, each model attempts to self-label the most informative suspension
    sample and passes it to the *other* model (cross-training). Training stops
    when neither model finds a beneficial sample or after `T` iterations.

    Training is delegated to PyTorch Lightning. Each of the two models is wrapped
    in a :class:`~lightning.LightningModule` and trained with a
    :class:`~lightning.Trainer` produced by a factory (same pattern as
    ``CoTrainingEnsemble_v2``). Every call to :meth:`_train_fun` reloads the best
    checkpoint (based on the validation metric monitored by the trainer's
    ``ModelCheckpoint`` callback) so we never keep the last-epoch weights.

    Use :meth:`setup_training` to provide the Lightning modules, trainer
    factories, batch sizes and shuffle flags before calling :meth:`train`.

    :param first_model:  A torch.nn.Module for view 1.
    :param second_model: A torch.nn.Module for view 2.
    :param verbose: Verbosity level. 0 = silent, 1 = key decisions, 2 = full per-candidate details.
    """

    def __init__(
            self,
            first_model: nn.Module,
            second_model: nn.Module,
            verbose: int = 0,
    ):
        # Keep the originals so external code can still reference them.
        self.first_model = first_model
        self.second_model = second_model
        self.models = [first_model, second_model]

        self.w1 = None
        self.w2 = None

        self.verbose = verbose

        # Set through setup_training()
        self.lightning_modules: list[LightningModule] | None = None  # pristine templates
        self.trainer_factories: list[Callable[[], Trainer]] | None = None
        self.batch_sizes: list[int] | None = None
        self.shuffle_dataloaders: list[bool] | None = None

        # Trained Lightning modules (set after calling .train())
        self._h1: LightningModule | None = None
        self._h2: LightningModule | None = None

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message)

    def setup_training(
            self,
            lightning_modules: list[LightningModule],
            trainer_factories: list[Callable[[], Trainer]],
            batch_sizes: list[int],
            shuffle_dataloaders: list[bool],
    ) -> None:
        r"""Setup training for the two models.

        Args:
            lightning_modules (list[LightningModule]): The lightning modules used to train the
                models. ``lightning_modules[0]`` wraps ``first_model`` and ``lightning_modules[1]``
                wraps ``second_model``. These instances are used as *pristine templates*: every
                training call trains a fresh deep copy, so they are never mutated.
            trainer_factories (list[Callable[[], Trainer]]): Factories that build a fresh Trainer
                for each training call. To benefit from best-model reload, the produced Trainer
                should include a ``ModelCheckpoint`` callback (typically monitoring ``val_loss``).

                Example:
                    trainer_factories = [
                        lambda: Trainer(max_epochs=200, accelerator="auto",
                                        callbacks=[EarlyStopping(monitor="val_loss", patience=50),
                                                   ModelCheckpoint(monitor="val_loss", save_top_k=1)]),
                        ...
                    ]
            batch_sizes (list[int]): Batch size used to train each model.
            shuffle_dataloaders (list[bool]): Whether to shuffle the training DataLoader of each model.
        """
        if (len(lightning_modules) != len(self.models) or
                len(trainer_factories) != len(self.models) or
                len(shuffle_dataloaders) != len(self.models) or
                len(batch_sizes) != len(self.models)):
            raise ValueError(
                f"The number of lightning modules (size={len(lightning_modules)}), "
                f"trainer factories (size={len(trainer_factories)}), "
                f"shuffle_dataloaders (size={len(shuffle_dataloaders)}), "
                f"batch_sizes (size={len(batch_sizes)}) must be the same as the number of models "
                f"(size={len(self.models)})."
            )

        self.lightning_modules = lightning_modules
        self.trainer_factories = trainer_factories
        self.batch_sizes = batch_sizes
        self.shuffle_dataloaders = shuffle_dataloaders

    def _check_if_training_is_possible(self) -> None:
        if (self.lightning_modules is None
                or self.trainer_factories is None
                or self.batch_sizes is None
                or self.shuffle_dataloaders is None):
            raise ValueError("You need to call setup_training before calling train.")

    def train(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: int,
            val_data: torch.Tensor | None = None,
            val_label: torch.Tensor | None = None,
    ) -> None:
        """
        Full COPROG training procedure (Algorithm 1 in the paper).

        :param failure_data:        Shape (N, *feature_dims) – labelled failure set L.
        :param failure_label:       Shape (N,) or (N, 1)     – RUL labels for L.
        :param suspension_data:     Shape (M, *feature_dims) – unlabelled suspension set U.
        :param suspension_ids:      Shape (M,) – unit id of each suspension sequence.
        :param iterations:          Maximum number of co-training rounds T.
        :param suspension_pool_size: Size u of the random sub-pool U' drawn each round. If -1 then all censored are selected
        :param val_data:            Optional validation features used for early stopping /
                                    best-checkpoint selection during every training call.
        :param val_label:           Optional validation labels associated with ``val_data``.
                                    Must be provided together with ``val_data``.
        """
        self._check_if_training_is_possible()

        if (val_data is None) != (val_label is None):
            raise ValueError("val_data and val_label must both be provided or both be None.")

        total_suspension_units = len(torch.unique(suspension_ids))
        self._log(1, f"[Coprog] Starting training | failure samples: {len(failure_data)} | "
                     f"censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | pool size: {suspension_pool_size} | "
                     f"validation: {'yes' if val_data is not None else 'no'}")

        # Line 1 – L1 = L2 = L  (we split L into two views)
        x1, y1 = failure_data, failure_label
        x2, y2 = failure_data, failure_label

        # Line 2 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
        self._log(1, f"[Coprog] Initial training of h1 on {len(x1)} failure samples...")
        h1 = self._train_fun(
            copy.deepcopy(self.lightning_modules[0]),
            0,
            x1,
            y1,
            val_data,
            val_label,
        )
        self._log(1, f"[Coprog] Initial training of h2 on {len(x2)} failure samples...")
        h2 = self._train_fun(
            copy.deepcopy(self.lightning_modules[1]),
            1,
            x2,
            y2,
            val_data,
            val_label,
        )
        self._log(1, f"[Coprog] Initial training done.")

        remaining_suspension_ids = torch.unique(suspension_ids)

        # Line 3 – Repeat for T times
        for i in range(iterations):

            # Line 4 – Create pool U' of u suspension units
            if len(remaining_suspension_ids) == 0:
                self._log(1, f"[Coprog] Early stop at iteration {i}: no remaining censored units.")
                break

            if suspension_pool_size == -1:
                pool_size = len(remaining_suspension_ids)
            else:
                pool_size = min(suspension_pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
            pool_ids = shuffled_ids[:pool_size]  # U'

            self._log(1, f"[Coprog] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            pi = [None, None]  # π1, π2

            # Line 5 – for j = 1 to 2
            for j, (hj, xj, yj) in enumerate([(h1, x1, y1), (h2, x2, y2)]):

                # Line 6-9 – compute Δ for every X_u ⊂ U'
                best_delta = 0.0
                best_unit_id: torch.Tensor | None = None
                best_xu: torch.Tensor | None = None
                best_label: torch.Tensor | None = None

                self._log(2, f"[Coprog]   Model h{j + 1}: evaluating {len(pool_ids)} candidates...")

                for candidate_idx, unit_id in enumerate(pool_ids):
                    # Extract all sequences for this specific unit
                    mask = (suspension_ids == unit_id)
                    xu = suspension_data[mask]  # Shape: (N_u, *dims)

                    # Line 7 – pseudo-label from current model j
                    lu_p = self._predict(hj, xu)  # L^P_u

                    # Line 8 – train a temporary model h'j from scratch on the augmented set
                    x_aug = torch.cat([xj, xu], dim=0)

                    lu_p_reshaped = lu_p.view(-1, yj.shape[1]) if yj.dim() > 1 else lu_p.view(-1)
                    y_aug = torch.cat([yj, lu_p_reshaped], dim=0)

                    hj_prime = self._train_fun(
                        copy.deepcopy(self.lightning_modules[j]),
                        j,
                        x_aug,
                        y_aug,
                        val_data,
                        val_label,
                    )

                    # Line 9 – Δ_{j, X_u} = MSE(hj, L) – MSE(h'j, L)
                    delta = self._confidence_measure(xj, yj, hj, hj_prime)

                    self._log(2, f"[Coprog]     candidate {candidate_idx + 1}/{len(pool_ids)} "
                                 f"unit {unit_id.item()}: delta = {delta:.4f}")

                    if delta > best_delta:
                        best_delta = delta
                        best_unit_id = unit_id
                        best_xu = xu
                        best_label = lu_p

                # Lines 11-15 – select best candidate or set π = ∅
                if best_unit_id is not None and best_delta > 0:
                    # Line 12 – X*_j = argmax Δ;  L*_j = h_j(X*_j)
                    # Line 13 – π_j = {(X*_j, L*_j)};  U' = U' \ π_j
                    pi[j] = (best_unit_id, best_xu, best_label)
                    # Remove selected sample from U'
                    pool_ids = pool_ids[pool_ids != best_unit_id]
                    self._log(1, f"[Coprog]   Model h{j + 1}: selected unit {best_unit_id.item()} "
                                 f"(delta = {best_delta:.4f})")
                else:
                    pi[j] = None  # Line 15
                    self._log(1, f"[Coprog]   Model h{j + 1}: no unit selected (no positive delta found).")

            # Line 17 – end for j

            # Line 18 – if π1 == ∅ && π2 == ∅  exit
            if pi[0] is None and pi[1] is None:
                self._log(1, f"[Coprog] Early stop at iteration {i + 1}: "
                             f"no model found a beneficial censored unit.")
                break

            # Line 19 – L1 = L1 ∪ π2;  L2 = L2 ∪ π1   (cross-labelling)
            if pi[1] is not None:
                _, xu2, lu2 = pi[1]

                lu2_reshaped = lu2.view(-1, y1.shape[1]) if y1.dim() > 1 else lu2.view(-1)

                x1 = torch.cat([x1, xu2], dim=0)
                y1 = torch.cat([y1, lu2_reshaped], dim=0)

            if pi[0] is not None:
                _, xu1, lu1 = pi[0]

                lu1_reshaped = lu1.view(-1, y2.shape[1]) if y2.dim() > 1 else lu1.view(-1)

                x2 = torch.cat([x2, xu1], dim=0)
                y2 = torch.cat([y2, lu1_reshaped], dim=0)

            # Remove newly labelled samples from U (global pool)
            selected_ids = []
            if pi[0] is not None: selected_ids.append(pi[0][0].item())
            if pi[1] is not None: selected_ids.append(pi[1][0].item())

            for s_id in set(selected_ids):
                remaining_suspension_ids = remaining_suspension_ids[remaining_suspension_ids != s_id]

            # Line 20 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
            self._log(1, f"[Coprog]   Retraining h1 | dataset size: {len(x1)} samples")
            h1 = self._train_fun(
                copy.deepcopy(self.lightning_modules[0]),
                0,
                x1,
                y1,
                val_data,
                val_label,
            )
            self._log(1, f"[Coprog]   Retraining h2 | dataset size: {len(x2)} samples")
            h2 = self._train_fun(
                copy.deepcopy(self.lightning_modules[1]),
                1,
                x2,
                y2,
                val_data,
                val_label,
            )

        self._log(1, f"[Coprog] Training complete.")

        # Save final trained models
        self._h1 = h1
        self._h2 = h2

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction (line 22): L^P = w1*h1(x) + w2*h2(x).

        :param x: Shape (N, *feature_dims).

        :return Tensor of shape (N,) with predicted RUL values.
        """
        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .predict().")

        if self.w1 is None or self.w2 is None:
            raise RuntimeError("Call .calculate_weights() before .predict().")

        p1 = self._predict(self._h1, x).view(-1)
        p2 = self._predict(self._h2, x).view(-1)
        return self.w1 * p1 + self.w2 * p2

    def prediction_for_first_model(self, x: torch.Tensor) -> torch.Tensor:
        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .prediction_for_first_model().")
        return self._predict(self._h1, x).view(-1)

    def prediction_for_second_model(self, x: torch.Tensor) -> torch.Tensor:
        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .prediction_for_second_model().")
        return self._predict(self._h2, x).view(-1)

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
        TrainFun: fit ``model`` on (x, y) with Lightning and return it.
        Equivalent to lines 2 / 8 / 20 in the pseudo-code.

        A fresh Trainer is built from the factory for this model index. If a
        validation set is provided it is used for early stopping / checkpointing.
        After ``trainer.fit`` the best checkpoint (as tracked by the trainer's
        ``ModelCheckpoint`` callback) is reloaded so we never keep the potentially
        worse last-epoch weights.
        """
        # A Trainer keeps internal state, so we always build a fresh one.
        trainer = self.trainer_factories[model_index]()
        batch_size = self.batch_sizes[model_index]

        train_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=self.shuffle_dataloaders[model_index],
        )

        val_loader = None
        if val_x is not None and val_y is not None:
            val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Reload the best checkpoint (based on the monitored validation metric) so we
        # use the best model after training instead of the last-epoch weights.
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", "") if checkpoint_callback else ""
        if best_model_path:
            self._log(2, f"[Coprog]     Reloading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=model.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            self._log(2, f"[Coprog]     No best model find the model with last epoch is used")

        return model

    def _predict(self, model: LightningModule, x: torch.Tensor) -> torch.Tensor:
        """
        Return model predictions (shape (N, *output_dims)) without tracking gradients.
        """
        model.eval()
        with torch.no_grad():
            x = x.to(next(model.parameters()).device)
            return model(x)

    def _confidence_measure(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            original_model: LightningModule,
            augmented_model: LightningModule,
    ) -> float:
        """
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
        y_flat = y.view(-1).to(next(original_model.parameters()).device)

        pred_orig = self._predict(original_model, x).view(-1)
        pred_aug = self._predict(augmented_model, x).view(-1)

        mse_orig = ((y_flat - pred_orig) ** 2).sum().item()
        mse_aug = ((y_flat - pred_aug) ** 2).sum().item()

        return mse_orig - mse_aug  # > 0 means augmented model is better

    def calculate_weights(
            self,
            x_test: torch.Tensor,
            target: torch.Tensor,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float],
            mode: str,
    ):
        """
        Compute the ensemble weights from each model's score on a held-out set.

        Args:
            x_test: Features of the held-out (ideally validation) set.
            target: Labels of the held-out set.
            criteria_callback: Callable returning a scalar score from (prediction, target).
            mode: value can be "min" or "max".
                "min" mean that more the score is little more the model is good.
                "max" mean that more the score is high more the model is good.
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")

        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .calculate_weights().")

        scores = []

        # Flatten both sides so the criteria callback compares aligned (N,) vectors
        # instead of broadcasting (N, 1) against (N,) into an (N, N) matrix.
        target_flat = target.view(-1).float()

        pred_h1 = self._predict(self._h1, x_test).view(-1).to(target_flat.device)
        scores.append(criteria_callback(pred_h1, target_flat))

        pred_h2 = self._predict(self._h2, x_test).view(-1).to(target_flat.device)
        scores.append(criteria_callback(pred_h2, target_flat))

        self._log(1, f"[Coprog] Calculating weights (mode={mode}) | "
                     f"scores per model: {[round(s, 4) for s in scores]}")

        if mode == "min":
            if any(s == 0 for s in scores):
                raise ValueError(
                    "At least one model has a score of zero in 'min' mode, inverse weighting is undefined.")
            inv_scores = [1.0 / s for s in scores]
            total = sum(inv_scores)
            weights = [inv_s / total for inv_s in inv_scores]
        else:
            total = sum(scores)
            if total == 0:
                raise ValueError(f"The sum of scores from all models is zero, cannot calculate weights : {scores}")
            weights = [s / total for s in scores]

        self.w1, self.w2 = weights

        self._log(1, f"[Coprog] Weights assigned: h1={round(self.w1, 4)}, h2={round(self.w2, 4)}")