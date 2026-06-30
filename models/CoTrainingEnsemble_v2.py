import copy
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
    ):
        """
        :param models: list[nn.Module]
            The models that will be used in the co-training ensemble.
        """
        self.models = models
        self.number_of_models = len(self.models)
        self.lightning_modules = None
        self.trainer_factories = None
        self.batchs_size = None
        self.shuffle_dataloaders = None

    def setup_training(
            self,
            lightning_modules: list[LightningModule],
            trainer_factories: list[Callable[[], Trainer]],
            batchs_size: list[int],
            shuffle_dataloaders: list[bool],
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
        """
        if (len(lightning_modules) != len(self.models) or
                len(trainer_factories) != len(self.models) or
                len(shuffle_dataloaders) != len(self.models) or
                len(batchs_size) != len(self.models)):
            raise ValueError("The number of lightning modules and trainers must be the same as the number of models.")

        self.lightning_modules = lightning_modules
        self.trainer_factories = trainer_factories
        self.batchs_size = batchs_size
        self.shuffle_dataloaders = shuffle_dataloaders

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
        """
        self._check_if_training_is_possible()

        models_datasets = []
        h = []

        for j in range(self.number_of_models):
            x_i, y_i = failure_data, failure_label

            # TODO need to see how to deel with survloss and data
            # if train_with_censored_data:
            #     x_i = torch.cat([x_i, suspension_data], dim=0)
            #     y_i = torch.cat([y_i, ], dim=0)

            models_datasets.append((x_i, y_i))

            h_j = self._train_fun(
                model=copy.deepcopy(self.lightning_modules[j]),
                model_index=j,
                x=x_i,
                y=y_i,
            )

            h.append(h_j)

        remaining_suspension_ids = torch.unique(suspension_ids)

        for i in range(iterations):
            if len(remaining_suspension_ids) == 0:
                break

            pool_size = min(suspension_pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
            pool_ids = shuffled_ids[:pool_size]  # U'

            # Phase 1 — for each model j, predict pseudo-labels and compute delta for every
            # unit in the pool. Results are stored in an OrderedDict (sorted by delta desc)
            # so the best candidate is always first.
            # Structure: all_preds[j] = OrderedDict{ unit_id_int -> (unit_id, xu, lu_p, delta) }
            all_preds: dict[int, OrderedDict] = {}

            for j in range(self.number_of_models):
                hj = h[j]
                xj, yj = models_datasets[j]
                candidates = []

                for unit_id in pool_ids:
                    mask = (suspension_ids == unit_id)
                    xu = suspension_data[mask]

                    lu_p = self._predict(hj, xu)

                    x_augmented = torch.cat([xj, xu], dim=0)
                    lu_p_reshaped = lu_p.view(-1, yj.shape[1]) if yj.dim() > 1 else lu_p.view(-1)
                    y_augmented = torch.cat([yj, lu_p_reshaped], dim=0)

                    if is_fine_tuning_during_finding_best_suspension_data:
                        hj_prime = self._fine_tune_fun(
                            model=copy.deepcopy(hj),
                            model_index=j,
                            x=x_augmented,
                            y=y_augmented,
                        )
                    else:
                        hj_prime = self._train_fun(
                            model=copy.deepcopy(self.lightning_modules[j]),
                            model_index=j,
                            x=x_augmented,
                            y=y_augmented,
                        )

                    delta = self._confidence_measure(xj, yj, hj, hj_prime)
                    candidates.append((unit_id, xu, lu_p, delta))

                candidates.sort(key=lambda e: e[3], reverse=True)
                all_preds[j] = OrderedDict(
                    (uid.item(), (uid, xu, lu_p, delta))
                    for uid, xu, lu_p, delta in candidates
                )

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

                    # Remove the selected unit from every model's candidate map so it
                    # cannot be assigned to another model in this iteration.
                    for j in range(self.number_of_models):
                        all_preds[j].pop(selected_id, None)

                    remaining_suspension_ids = remaining_suspension_ids[
                        remaining_suspension_ids != censored_data_selected[k][0]
                    ]

            if all(x is None for x in censored_data_selected):
                break

            for j in range(self.number_of_models):
                if censored_data_selected[j] is not None:
                    _, xu, lu = censored_data_selected[j]
                    xj, yj = models_datasets[j]

                    lu2_reshaped = lu.view(-1, yj.shape[1]) if yj.dim() > 1 else lu.view(-1)

                    xj = torch.cat([xj, xu], dim=0)
                    yj = torch.cat([yj, lu2_reshaped], dim=0)

                    models_datasets[j] = (xj, yj)

                    if is_fine_tuning_for_last_step:
                        h[j] = self._fine_tune_fun(
                            model=copy.deepcopy(h[j]),
                            model_index=j,
                            x=xj,
                            y=yj,
                        )
                    else:
                        h[j] = self._train_fun(
                            model=copy.deepcopy(self.lightning_modules[j]),
                            model_index=j,
                            x=xj,
                            y=yj,
                        )

    def _train_fun(
            self,
            model: LightningModule,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> LightningModule:
        """
        Train the lightning module for the given index with the given data.

        :param model: LightningModule
            The lightning module to train.
        :param model_index: int
            The index of the model to train.
        :param x: torch.Tensor
            The data features.
        :param y: torch.Tensor
            The data labels.
        :return: LightningModule
            The trained lightning module.
        """
        # Trainer can save some state then we create a new one
        trainer: Trainer = self.trainer_factories[model_index]()

        batch_size: int = self.batchs_size[model_index]

        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=self.shuffle_dataloaders[model_index])

        trainer.fit(model, train_dataloaders=train_loader)

        return model

    def _fine_tune_fun(
            self,
            model: LightningModule,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> LightningModule:
        """
        Fine-tune the lightning module for the given index with the given data.

        :param model: LightningModule
            The lightning module to fine-tune.
        :param model_index: int
            The index of the model to fine-tune.
        :param x: torch.Tensor
            The data features.
        :param y: torch.Tensor
            The data labels.
        :return: LightningModule
            The fine-tuned lightning module.
        """
        # TODO implement the fine-tuning logic
        raise NotImplementedError("Fine-tuning is not implemented yet.")

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
