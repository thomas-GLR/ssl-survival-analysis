import copy
from typing import Callable

import torch
import torch.nn as nn
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

    :param first_model:  A torch.nn.Module for view 1.
    :param second_model: A torch.nn.Module for view 2.
    :param w1: Weight of first model in the final ensemble prediction (default 0.5).
    :param w2: Weight of second model in the final ensemble prediction (default 0.5).
    :param lr: Learning rate used when fine-tuning with a new labelled suspension sample (default 1e-3).
    :param epochs: Number of epochs for each TrainFun call (default 20).
    :param batch_size: Mini-batch size for TrainFun (default 32).
    :param device: torch.device to run on (default: cuda if available, else cpu).
    :param shuffle_dataloader: if set to True it will shuffle the data during the training otherwise no. Default = False
    """

    def __init__(
            self,
            first_model: nn.Module,
            second_model: nn.Module,
            lr_first_model: float = 1e-3,
            lr_second_model: float = 1e-3,
            epochs_first_model: int = 20,
            epochs_second_model: int = 20,
            batch_size_first_model: int = 32,
            batch_size_second_model: int = 32,
            device: str | None = None,
            shuffle_dataloader: bool = False,
            verbose: int = 0,
            first_and_second_model_already_trained: bool = False
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep the originals so we can deep-copy them cheaply
        self.first_model = first_model.to(self.device)
        self.second_model = second_model.to(self.device)

        self.w1 = None
        self.w2 = None

        self.lr_first_model = lr_first_model
        self.lr_second_model = lr_second_model
        self.epochs_first_model = epochs_first_model
        self.epochs_second_model = epochs_second_model
        self.batch_size_first_model = batch_size_first_model
        self.batch_size_second_model = batch_size_second_model

        self.shuffle_dataloader = shuffle_dataloader
        self.verbose = verbose

        # Trained models (set after calling .train())
        self._h1: nn.Module | None = self.first_model if first_and_second_model_already_trained else None
        self._h2: nn.Module | None = self.second_model if first_and_second_model_already_trained else None

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message)

    def train(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: int
    ) -> None:
        """
        Full COPROG training procedure (Algorithm 1 in the paper).

        :param failure_data:        Shape (N, *feature_dims) – labelled failure set L.
        :param failure_label:       Shape (N,) or (N, 1)     – RUL labels for L.
        :param suspension_data:     Shape (M, *feature_dims) – unlabelled suspension set U.
        :param iterations:          Maximum number of co-training rounds T.
        :param suspension_pool_size: Size u of the random sub-pool U' drawn each round.
        """
        failure_data = failure_data.to(self.device)
        failure_label = failure_label.to(self.device).float()
        suspension_data = suspension_data.to(self.device)
        suspension_ids = suspension_ids.to(self.device)

        total_suspension_units = len(torch.unique(suspension_ids))
        self._log(1, f"[Coprog] Starting training | failure samples: {len(failure_data)} | "
                     f"censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | pool size: {suspension_pool_size}")

        # Line 1 – L1 = L2 = L  (we split L into two views)
        x1, y1 = failure_data, failure_label
        x2, y2 = failure_data, failure_label

        # Line 2 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
        self._log(1, f"[Coprog] Initial training of h1 on {len(x1)} failure samples...")
        h1 = self._train_fun(
            copy.deepcopy(self.first_model),
            x1,
            y1,
            self.lr_first_model,
            self.batch_size_first_model,
            self.epochs_first_model,
            "h1",
        )
        self._log(1, f"[Coprog] Initial training of h2 on {len(x2)} failure samples...")
        h2 = self._train_fun(
            copy.deepcopy(self.second_model),
            x2,
            y2,
            self.lr_second_model,
            self.batch_size_second_model,
            self.epochs_second_model,
            "h2"
        )
        self._log(1, f"[Coprog] Initial training done.")

        remaining_suspension_ids = torch.unique(suspension_ids)# remaining_suspension = suspension_data.clone()

        number_iterations = 0

        # Line 3 – Repeat for T times
        for i in range(iterations):

            # Line 4 – Create pool U' of u suspension units
            if len(remaining_suspension_ids) == 0:
                self._log(1, f"[Coprog] Early stop at iteration {i}: no remaining censored units.")
                break

            pool_size = min(suspension_pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
            pool_ids = shuffled_ids[:pool_size] # U'

            self._log(1, f"[Coprog] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            pi = [None, None]  # π1, π2

            training_params = [
                {
                    "lr": self.lr_first_model,
                    "batch_size": self.batch_size_first_model,
                    "epochs": self.epochs_first_model,
                },
                {
                    "lr": self.lr_second_model,
                    "batch_size": self.batch_size_second_model,
                    "epochs": self.epochs_second_model,
                }
            ]

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

                    # Line 8 – train a temporary model h'j
                    model_j = copy.deepcopy(self.first_model if j == 0 else self.second_model)
                    x_aug = torch.cat([xj, xu], dim=0)

                    lu_p_reshaped = lu_p.view(-1, yj.shape[1]) if yj.dim() > 1 else lu_p.view(-1)
                    y_aug = torch.cat([yj, lu_p_reshaped], dim=0)

                    hj_prime = self._train_fun(
                        copy.deepcopy(model_j),
                        x_aug,
                        y_aug,
                        training_params[j]["lr"],
                        training_params[j]["batch_size"],
                        training_params[j]["epochs"],
                        f"h{j + 1}_prime",
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
                copy.deepcopy(self.first_model),
                x1,
                y1,
                self.lr_first_model,
                self.batch_size_first_model,
                self.epochs_first_model,
                "h1",
            )
            self._log(1, f"[Coprog]   Retraining h2 | dataset size: {len(x2)} samples")
            h2 = self._train_fun(
                copy.deepcopy(self.second_model),
                x2,
                y2,
                self.lr_second_model,
                self.batch_size_second_model,
                self.epochs_second_model,
                "h2"
            )
            number_iterations += 1

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

        x = x.to(self.device)
        p1 = self._predict(self._h1, x)
        p2 = self._predict(self._h2, x)
        return self.w1 * p1 + self.w2 * p2

    def _train_fun(
            self,
            model: nn.Module,
            x: torch.Tensor,
            y: torch.Tensor,
            lr: float,
            batch_size: int,
            epochs: int,
            model_name: str = ""
    ) -> nn.Module:
        """
        TrainFun: fit `model` on (x, y) with MSE loss and return it.
        Equivalent to lines 2 / 8 / 20 in the pseudo-code.
        """
        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle_dataloader)

        self._log(2, f"[Coprog]     Training {model_name} | samples: {len(x)} | "
                     f"epochs: {epochs} | batch_size: {batch_size} | lr: {lr}")

        best_loss = 1_000_000
        avg_epochs_loss = 0.

        for epoch in range(epochs):
            avg_loss = 0.

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()

            if avg_loss < best_loss:
                best_loss = (avg_loss / len(loader))

            avg_epochs_loss += (avg_loss / len(loader))

        self._log(2, f"[Coprog]     {model_name} done | avg loss: {avg_epochs_loss / epochs:.4f} | "
                     f"best epoch loss: {best_loss:.4f}")

        return model

    @torch.no_grad()
    def _predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Return flattened predictions (shape (N,)) without tracking gradients.
        """
        model.eval()
        return model(x).view(-1)

    def _confidence_measure(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            original_model: nn.Module,
            augmented_model: nn.Module,
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
        y_flat = y.view(-1)

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

        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .calculate_weights().")

        scores = []

        x_test = x_test.to(self.device)
        target = target.to(self.device).float()

        pred_h1 = self._predict(self._h1, x_test)
        scores.append(criteria_callback(pred_h1, target))

        pred_h2 = self._predict(self._h2, x_test)
        scores.append(criteria_callback(pred_h2, target))

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