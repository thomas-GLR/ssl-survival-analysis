import copy
import csv
import os
from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from enum import Enum

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator
from lightning import LightningModule, Trainer
from torch.utils.data import TensorDataset, DataLoader


class _TorchRegressorAdapter:
    """Minimal sklearn-style learner exposing only ``predict`` for ``crepes.WrapRegressor``.

    ``WrapRegressor``/``DifficultyEstimator`` only ever call ``learner.predict(X)`` (never
    ``.fit``) and work with 2-D numpy arrays, whereas the wrapped torch model expects a
    ``(N, seq_len, n_features)`` tensor. This adapter bridges the two: it reshapes the
    flattened features back to the model layout, runs the ensemble's ``_predict`` and
    returns a 1-D numpy array aligned with the numpy labels crepes uses for residuals.
    """

    def __init__(self, predict_fn, model, seq_len: int, n_features: int):
        self._predict_fn = predict_fn
        self._model = model
        self._seq_len = seq_len
        self._n_features = n_features

    def predict(self, X):
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        x = x.view(-1, self._seq_len, self._n_features)
        preds = self._predict_fn(self._model, x)
        return preds.detach().cpu().numpy().reshape(-1)


class CoTrainingEnsemble_v3:
    """
    Confidence-based co-training ensemble (v3) — owner-based selection with a confidence-weighted
    pseudo-label and an optional confidence threshold.

    Like v2, each already-trained model is wrapped in a normalized conformal regressor (calibrated
    on the validation set with a kNN ``DifficultyEstimator`` so widths vary per unit) and every
    remaining censored unit is scored by the *width* of its conformal prediction interval — a
    narrower interval means the model is more confident about that unit.

    Where v2 gives each model the unit its *peers* are most confident about, v3 turns it around:

    * **Owner-based selection.** Each model *k* picks the unit **it** is most confident about
      (smallest interval width). Because each model calibrates its own conformal regressor, widths
      are not comparable across models, so they are first **normalized per model** (divided by that
      model's median width over the remaining units).
    * **Confidence-weighted pseudo-label.** For the owned unit the pseudo-RUL is a weighted average
      of *all* models' per-window predictions, weighted by inverse normalized width (narrower =
      more confident = larger weight; the weights sum to 1).
    * **Owner-excluded assignment.** The unit and its computed RUL are handed to **every other
      model but not the owner** — the confident model teaches its peers rather than re-learning its
      own guess — and the unit is then removed from the pool for good.
    * **Threshold.** ``width_threshold`` (on the normalized width) gates two things: a model may
      only *own* a unit when its own normalized width is under the threshold, and only models under
      the threshold for a unit contribute to that unit's weighted RUL. A model whose prediction is
      too wide (not confident enough) is ignored both as an owner and as a contributor.

    Models are retrained from scratch each iteration with their newly pseudo-labelled units.
    """

    def __init__(
            self,
            models: list[nn.Module],
            weights: list[float] | None = None,
            verbose: int = 0,
            confidence: float = 0.95,
            width_threshold: float | None = None,
    ):
        """
        :param models: list[nn.Module]
            The models that will be used in the co-training ensemble.
        :param weights: list[float] | None
            Optional pre-defined weights for each model.
        :param verbose: int
            Verbosity level. 0 = silent, 1 = key decisions, 2 = full per-candidate details.
        :param confidence: float
            Confidence level in ``(0, 1)`` passed to ``crepes`` when building the conformal
            prediction intervals used to score censored units. Higher confidence produces
            wider intervals. Defaults to ``0.95``.
        :param width_threshold: float | None
            Optional cutoff on the **normalized** interval width (raw width divided by the
            model's median width over the remaining units, so a value around ``1.0`` is a typical
            width for that model, ``< 1`` is more confident than typical). A model may only *own*
            a unit whose normalized width is ``<= width_threshold``, and only models under the
            threshold for a unit contribute to that unit's weighted RUL. ``None`` (default)
            disables the filter: every model can own and every model contributes.
        """
        if weights is not None and len(models) != len(weights):
            raise ValueError("The number of weights must be the same as the number of models.")

        if not (0 < confidence < 1):
            raise ValueError("confidence must be in the range (0, 1).")

        if width_threshold is not None and width_threshold <= 0:
            raise ValueError("width_threshold must be > 0 when provided.")

        self.models = models
        self.number_of_models = len(self.models)
        self.lightning_modules = None
        self.trainer_factories = None
        self.batchs_size = None
        self.shuffle_dataloaders = None
        self.weights = weights
        self.verbose = verbose
        self.confidence = confidence
        self.width_threshold = width_threshold
        # Optional path to a .txt log file (set by ``train``). When not None, every
        # ``_log`` message is appended to it regardless of ``verbose``.
        self._log_file_path: str | None = None

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message)
        # When a log file is configured, capture every message regardless of level;
        # append-per-call keeps it crash-safe and needs no file-handle lifecycle.
        if self._log_file_path is not None:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

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
            raise ValueError(
                f"The number of lightning modules (size={len(lightning_modules)}), trainers (size={len(trainer_factories)}), shuffle_dataloaders (size={len(shuffle_dataloaders)}), batchs_size (size={len(batchs_size)}) must be the same as the number of models.")

        self.lightning_modules = lightning_modules
        self.trainer_factories = trainer_factories
        self.batchs_size = batchs_size
        self.shuffle_dataloaders = shuffle_dataloaders

    def train(
            self,
            train_with_censored_data: bool,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            test_data: torch.Tensor | None = None,
            test_label: torch.Tensor | None = None,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
            weight_mode: str = "min",
            metrics_file: str | None = None,
            log_file: str | None = None,
    ) -> None:
        r"""The train algorithm for co-training ensemble v3.

        Each iteration scores **all** remaining censored units with a conformal prediction
        interval per model (calibrated on the validation set) and normalizes the widths per model.
        Then, greedily over the models, each model *owns* the remaining unit it is most confident
        about (smallest normalized width, and — when ``width_threshold`` is set — only if that
        width is under the threshold). The owned unit's pseudo-RUL is a weighted average of every
        contributing model's per-window predictions (inverse-normalized-width weights), and the
        unit is assigned to **all other models but the owner** before being removed from the pool.
        Finally every model with new data is retrained from scratch.

        Args:
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
            val_data:
                Validation features. Required: used both for early stopping / best-checkpoint
                selection during every training call and as the calibration set for the
                conformal regressors.
            val_label:
                Validation labels associated with ``val_data``. Required.
            test_data:
                Optional test features used only for the per-stage metrics logged to
                ``metrics_file``. Never used for training or model selection.
            test_label:
                Optional test labels associated with ``test_data``.
            criteria_callback:
                Score used both per model (test score) and to compute the ensemble
                weights on the validation set for the "with weights" metrics. Same
                callback the caller passes to ``calculate_weights`` (lower is better
                for ``weight_mode="min"``). Required when metrics logging is enabled.
            weight_mode:
                "min" or "max", passed to ``_compute_weights`` when deriving the
                per-stage ensemble weights (defaults to "min").
            metrics_file:
                Optional path to a ``.csv`` file. When given (together with the test
                set, the validation set and ``criteria_callback``), per-stage metrics
                are appended: one row after the initial training, one after each
                iteration that retrains, and one final row. Enables metrics logging.
            log_file:
                Optional path to a ``.txt`` file. When given, every log message is
                appended to it regardless of ``verbose`` (stdout still follows
                ``verbose``). The path stays active on the instance after ``train``
                returns, so logs from a subsequent ``calculate_weights`` call are
                captured too.
        """
        self._log_file_path = log_file

        self._check_if_training_is_possible()

        # The validation set is used to calibrate every conformal regressor, so it is
        # mandatory in v3 (it is also reused for early stopping and the weighted metrics).
        if val_data is None or val_label is None:
            raise ValueError(
                "val_data and val_label are required in v3 (used to calibrate the conformal regressors).")

        # Per-stage metrics need the test set, the criteria callback and a destination file.
        # Enable only when all are present; if the caller asked for metrics (test_data given)
        # but left something out, fail loudly.
        metrics_enabled = test_data is not None
        if metrics_enabled:
            if test_label is None:
                raise ValueError("test_label must be provided together with test_data.")
            if criteria_callback is None:
                raise ValueError("criteria_callback is required to log per-stage metrics (per-model test score).")
            if metrics_file is None:
                raise ValueError("metrics_file is required to log per-stage metrics.")

        total_suspension_units = len(torch.unique(suspension_ids))
        self._log(1, f"[CoTraining] Starting training | models: {self.number_of_models} | "
                     f"failure samples: {len(failure_data)} | "
                     f"censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | confidence: {self.confidence} | "
                     f"width_threshold: {self.width_threshold} | ")

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

        if metrics_enabled:
            self._log_stage_metrics(
                stage="initial",
                h=h,
                models_datasets=models_datasets,
                test_data=test_data,
                test_label=test_label,
                val_data=val_data,
                val_label=val_label,
                criteria_callback=criteria_callback,
                weight_mode=weight_mode,
                metrics_file=metrics_file,
            )

        remaining_suspension_ids = torch.unique(suspension_ids)

        for i in range(iterations):
            if len(remaining_suspension_ids) == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i}: no remaining censored units.")
                break

            self._log(1, f"[CoTraining] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} ---")

            # Phase 1 — for each model j, build a conformal regressor (calibrated on the
            # validation set) and score every remaining censored unit by the width of the
            # prediction interval at the unit's last window (narrower = more confident).
            # Results are stored in an OrderedDict (sorted by width ascending) so the most
            # confident candidate is always first.
            # Structure:
            #   all_preds[j] = OrderedDict{ unit_id_int -> (unit_id, xu, lu_p, lower, upper, width) }
            all_preds: dict[int, OrderedDict] = {}

            for j in range(self.number_of_models):
                hj = h[j]
                xj, yj = models_datasets[j]

                self._log(2, f"[CoTraining]   Model {j}: calibrating conformal regressor and "
                             f"scoring {len(remaining_suspension_ids)} censored units...")

                wrapper = self._build_calibrated_regressor(hj, xj, val_data, val_label)

                # Collect each unit's rows, its per-window pseudo-labels and its last window
                # (the interval of that final window is the unit's "range").
                unit_ids = list(remaining_suspension_ids)
                xus: list[torch.Tensor] = []
                lu_ps: list[torch.Tensor] = []
                last_windows: list[torch.Tensor] = []
                for unit_id in unit_ids:
                    mask = (suspension_ids == unit_id)
                    xu = suspension_data[mask]
                    xus.append(xu)
                    lu_ps.append(self._predict(hj, xu))
                    last_windows.append(xu[-1])

                # One batched conformal call for all units' last windows -> (U, 2) [lower, upper].
                last_windows_tensor = torch.stack(last_windows, dim=0)
                intervals = wrapper.predict_int(
                    self._flatten(last_windows_tensor), confidence=self.confidence)

                candidates = []
                for idx, unit_id in enumerate(unit_ids):
                    lower = float(intervals[idx, 0])
                    upper = float(intervals[idx, 1])
                    width = upper - lower
                    candidates.append((unit_id, xus[idx], lu_ps[idx], lower, upper, width))
                    self._log(2, f"[CoTraining]     unit {unit_id.item()}: "
                                 f"range [{lower:.2f}, {upper:.2f}] width = {width:.4f}")

                # Smaller width = more confident, so sort ascending (best candidate first).
                candidates.sort(key=lambda e: e[5])
                all_preds[j] = OrderedDict(
                    (uid.item(), (uid, xu, lu_p, lower, upper, width))
                    for uid, xu, lu_p, lower, upper, width in candidates
                )

                if self.verbose >= 2:
                    ranking = [(uid.item(), round(w, 4)) for uid, _, _, _, _, w in candidates]
                    self._log(2, f"[CoTraining]   Model {j} candidate ranking (most confident first): "
                                 f"{ranking}")

            # Widths are not comparable across models (each model calibrates its own conformal
            # regressor), so normalize each model's widths by its own median before comparing
            # them or applying the threshold.
            norm_width = self._normalized_widths(all_preds)

            # Phase 2 — greedily, for each model k (the "owner"), pick the remaining unit k is
            # most confident about, compute a confidence-weighted RUL from every contributing
            # model, and assign it to all other models. The owned unit is then removed from the
            # pool so no other model can reuse it and it never returns to the owner.
            available_ids: set[int] = {uid for preds in all_preds.values() for uid in preds}

            # pending[j] accumulates the (xu, rul) units assigned to model j this iteration; a
            # model can receive up to (number_of_models - 1) units (one per other owner).
            pending: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
                [] for _ in range(self.number_of_models)
            ]

            any_selected = False
            for k in range(self.number_of_models):
                owner_uid = self._select_owner_unit(norm_width[k], available_ids)

                if owner_uid is None:
                    self._log(1, f"[CoTraining]   Model {k}: owns no unit "
                                 f"(none available or none under the width threshold).")
                    continue

                any_selected = True

                rul, contributors, contrib_weights = self._compute_weighted_rul(
                    all_preds, norm_width, owner_uid)
                xu = all_preds[k][owner_uid][1]

                receivers = [j for j in range(self.number_of_models) if j != k]
                contrib_log = [(j, round(w, 4)) for j, w in zip(contributors, contrib_weights)]
                self._log(1, f"[CoTraining]   Model {k}: owns unit {owner_uid} | "
                             f"RUL contributors (model, weight): {contrib_log} | "
                             f"assigned to models {receivers}")

                for j in receivers:
                    pending[j].append((xu, rul))

                # Remove the owned unit everywhere so it is never reused this iteration nor
                # returned to any model in a later one.
                available_ids.discard(owner_uid)
                for j in range(self.number_of_models):
                    all_preds[j].pop(owner_uid, None)
                    norm_width[j].pop(owner_uid, None)
                remaining_suspension_ids = remaining_suspension_ids[
                    remaining_suspension_ids != owner_uid
                    ]

            if not any_selected:
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: "
                             f"no unit owned by any model.")
                break

            for j in range(self.number_of_models):
                if pending[j]:
                    xj, yj = models_datasets[j]

                    for xu, lu in pending[j]:
                        lu_reshaped = lu.view(-1, yj.shape[1]) if yj.dim() > 1 else lu.view(-1)
                        xj = torch.cat([xj, xu], dim=0)
                        yj = torch.cat([yj, lu_reshaped], dim=0)

                    models_datasets[j] = (xj, yj)

                    self._log(1, f"[CoTraining]   Retraining model {j} from scratch | "
                                 f"received {len(pending[j])} new unit(s) | "
                                 f"dataset size: {len(xj)} samples")

                    h[j] = self._train_fun(
                        model=copy.deepcopy(self.lightning_modules[j]),
                        model_index=j,
                        x=xj,
                        y=yj,
                        val_x=val_data,
                        val_y=val_label,
                    )

            if metrics_enabled:
                self._log_stage_metrics(
                    stage=f"iteration_{i + 1}",
                    h=h,
                    models_datasets=models_datasets,
                    test_data=test_data,
                    test_label=test_label,
                    val_data=val_data,
                    val_label=val_label,
                    criteria_callback=criteria_callback,
                    weight_mode=weight_mode,
                    metrics_file=metrics_file,
                )

        if metrics_enabled:
            self._log_stage_metrics(
                stage="final",
                h=h,
                models_datasets=models_datasets,
                test_data=test_data,
                test_label=test_label,
                val_data=val_data,
                val_label=val_label,
                criteria_callback=criteria_callback,
                weight_mode=weight_mode,
                metrics_file=metrics_file,
            )

        self._log(1, f"[CoTraining] Training complete.")
        self.lightning_modules = h

    def _log_stage_metrics(
            self,
            stage: str,
            h: list[LightningModule],
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]],
            test_data: torch.Tensor,
            test_label: torch.Tensor,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float],
            weight_mode: str,
            metrics_file: str,
    ) -> None:
        """
        Compute and append one row of per-stage metrics to ``metrics_file``.

        For the current models ``h`` this records, per model, the train RMSE (on that
        model's own accumulated ``models_datasets`` split), the validation RMSE, the test
        RMSE and the test score; then the averages of the per-model test RMSE / test score
        (arithmetic mean, ignoring weights); and finally the test RMSE / test score of the
        weighted-ensemble prediction, whose weights are computed on the validation set via
        ``_compute_weights`` (so ``self.weights`` is left untouched).

        Args:
            stage: label for the row ("initial", "iteration_<k>" or "final").
            h: the current best model per index.
            models_datasets: per-model ``(x, y)`` accumulated training split.
            test_data, test_label: test set used only for the metrics.
            val_data, val_label: validation set (used for val RMSE and the weights).
            criteria_callback: score used for per-model test score and the weights.
            weight_mode: "min"/"max" passed to ``_compute_weights``.
            metrics_file: destination CSV; header written only when it does not yet exist.
        """
        test_label_flat = test_label.view(-1).float()

        train_rmses: list[float] = []
        val_rmses: list[float] = []
        test_rmses: list[float] = []
        test_scores: list[float] = []
        test_preds: list[torch.Tensor] = []

        for j, model in enumerate(h):
            xj, yj = models_datasets[j]
            train_rmses.append(self._mse_on(model, xj, yj) ** 0.5)
            val_rmses.append(self._mse_on(model, val_data, val_label) ** 0.5)

            pred_j = self._predict(model, test_data).view(-1).to(test_label_flat.device)
            test_preds.append(pred_j)
            test_rmses.append((((test_label_flat - pred_j) ** 2).mean().item()) ** 0.5)
            test_scores.append(criteria_callback(pred_j, test_label_flat))

        n = len(h)
        avg_test_rmse = sum(test_rmses) / n
        avg_test_score = sum(test_scores) / n

        # Weights come from the validation set (no test leakage) and do NOT mutate
        # self.weights — they exist only to report the weighted-ensemble metrics.
        # Pass ``h`` explicitly: self.lightning_modules still holds the untrained template
        # modules at this point (they are only replaced by ``h`` when train() finishes),
        # so weighting against it would give weights unrelated to the trained models —
        # and would not match the caller's post-train ``calculate_weights``.
        weights = self._compute_weights(val_data, val_label, criteria_callback, weight_mode, models=h)
        weighted_pred = torch.stack(
            [w * pred for w, pred in zip(weights, test_preds)], dim=0
        ).sum(dim=0).view(-1)
        weighted_test_rmse = (((test_label_flat - weighted_pred) ** 2).mean().item()) ** 0.5
        weighted_test_score = criteria_callback(weighted_pred, test_label_flat)

        header = ["stage"]
        for j in range(n):
            header += [f"train_rmse_{j}", f"val_rmse_{j}", f"test_rmse_{j}", f"test_score_{j}"]
        header += ["avg_test_rmse", "avg_test_score", "weighted_test_rmse", "weighted_test_score"]
        for j in range(n):
            header += [f"weight_{j}"]

        row = [stage]
        for j in range(n):
            row += [train_rmses[j], val_rmses[j], test_rmses[j], test_scores[j]]
        row += [avg_test_rmse, avg_test_score, weighted_test_rmse, weighted_test_score]
        for j in range(n):
            row += [weights[j]]

        # Append per call (crash-safe, no file-handle lifecycle), writing the header only
        # the first time the file is created — mirrors the append style of ``_log``.
        write_header = not os.path.exists(metrics_file)
        with open(metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        self._log(1, f"[CoTraining] Metrics [{stage}] | "
                     f"avg test RMSE: {avg_test_rmse:.4f} | avg test score: {avg_test_score:.4f} | "
                     f"weighted test RMSE: {weighted_test_rmse:.4f} | "
                     f"weighted test score: {weighted_test_score:.4f}")

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
        Fit ``model`` on ``(x, y)`` and reload the best checkpoint (as tracked by the
        trainer's ``ModelCheckpoint`` callback) so we never keep the potentially worse
        last-epoch weights. Used by ``_train_fun``.
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

    def _flatten(self, x: torch.Tensor) -> np.ndarray:
        """Flatten ``(N, seq_len, n_features)`` features to a 2-D ``(N, seq_len*n_features)``
        float32 numpy array, as required by ``crepes`` and its kNN ``DifficultyEstimator``."""
        return x.reshape(x.shape[0], -1).detach().cpu().numpy().astype(np.float32)

    def _build_calibrated_regressor(
            self,
            model: LightningModule,
            train_x: torch.Tensor,
            val_x: torch.Tensor,
            val_y: torch.Tensor,
    ) -> WrapRegressor:
        """
        Wrap an already-trained ``model`` in a normalized conformal regressor.

        A ``DifficultyEstimator`` is fitted on the model's (flattened) training features so
        the conformal intervals are *normalized* — their width varies per instance with how
        far the instance is from the training data. Without it a standard conformal regressor
        would return the same width for every unit, giving nothing to rank on. The regressor
        is calibrated on the validation set. crepes never calls ``.fit`` on the learner, so
        wrapping the pre-trained model via ``_TorchRegressorAdapter`` is sufficient.
        """
        seq_len, n_features = train_x.shape[1], train_x.shape[2]

        de = DifficultyEstimator()
        de.fit(X=self._flatten(train_x))

        wrapper = WrapRegressor(_TorchRegressorAdapter(self._predict, model, seq_len, n_features))
        wrapper.calibrate(
            X=self._flatten(val_x),
            y=val_y.view(-1).detach().cpu().numpy().astype(np.float32),
            de=de,
        )
        return wrapper

    def _mse_on(self, model: LightningModule, x: torch.Tensor, y: torch.Tensor) -> float:
        """Mean squared error of ``model`` on ``(x, y)``, used for the per-stage metrics."""
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

    def _normalized_widths(
            self,
            all_preds: dict[int, OrderedDict],
    ) -> dict[int, dict[int, float]]:
        r"""Normalize each model's interval widths by that model's median width.

        Each model calibrates its own conformal regressor, so raw widths live on different
        scales and cannot be compared across models directly. Dividing every model's widths by
        that model's median over the remaining units puts them on a common, model-relative scale
        (a normalized width of ~1.0 is a typical width for that model, ``< 1`` more confident than
        typical). This normalized width drives both the owner selection and the ``width_threshold``.

        Args:
            all_preds: mapping from model index j to an OrderedDict of
                ``{unit_id_int: (unit_id, xu, lu_p, lower, upper, width)}`` (index 5 = raw width).

        Returns:
            ``{model_index: {unit_id_int: normalized_width}}``. If a model's widths are degenerate
            (median ``<= 0``, e.g. all zero), every unit gets a normalized width of ``1.0`` so the
            ranking simply has nothing to discriminate on rather than dividing by zero.
        """
        norm: dict[int, dict[int, float]] = {}
        for j, preds in all_preds.items():
            widths = [tup[5] for tup in preds.values()]
            if not widths:
                norm[j] = {}
                continue
            median = float(np.median(widths))
            if median <= 0:
                norm[j] = {uid: 1.0 for uid in preds}
            else:
                norm[j] = {uid: tup[5] / median for uid, tup in preds.items()}
        return norm

    def _select_owner_unit(
            self,
            norm_width_k: dict[int, float],
            available_ids: set[int],
    ) -> int | None:
        r"""Pick the unit model k is most confident about among the still-available units.

        The owned unit is the available one with the smallest *normalized* width for model k.
        When ``self.width_threshold`` is set, the model owns nothing unless that smallest width is
        ``<= width_threshold`` (its best prediction is still not confident enough).

        Args:
            norm_width_k: ``{unit_id_int: normalized_width}`` for model k.
            available_ids: unit ids not yet owned by another model this iteration.

        Returns:
            The owned unit id, or ``None`` if none is available / none passes the threshold.
        """
        candidates = [(uid, nw) for uid, nw in norm_width_k.items() if uid in available_ids]
        if not candidates:
            return None

        best_uid, best_nw = min(candidates, key=lambda e: e[1])

        if self.width_threshold is not None and best_nw > self.width_threshold:
            return None

        return best_uid

    def _compute_weighted_rul(
            self,
            all_preds: dict[int, OrderedDict],
            norm_width: dict[int, dict[int, float]],
            unit_id_int: int,
    ) -> tuple[torch.Tensor, list[int], list[float]]:
        r"""Confidence-weighted per-window RUL for ``unit_id_int``.

        Contributors are every model that scored the unit and — when ``self.width_threshold`` is
        set — whose normalized width is ``<= width_threshold``. Each contributor's weight is its
        inverse normalized width, normalized so the weights sum to 1 (narrower interval = more
        confident = larger weight). The RUL is the weighted sum of the contributors' per-window
        point predictions ``lu_p`` (all contributors share the unit's window count).

        The owner always qualifies (its width passed the same threshold during selection), so the
        contributor list is never empty.

        Returns:
            ``(rul, contributors, weights)`` where ``rul`` has the shape of a single model's
            ``lu_p`` and ``contributors``/``weights`` are aligned lists (for logging).
        """
        eps = 1e-9

        contributors: list[int] = []
        for j, preds in all_preds.items():
            if unit_id_int not in preds:
                continue
            nw = norm_width[j][unit_id_int]
            if self.width_threshold is not None and nw > self.width_threshold:
                continue
            contributors.append(j)

        inv = [1.0 / max(norm_width[j][unit_id_int], eps) for j in contributors]
        total = sum(inv)
        weights = [v / total for v in inv]

        rul: torch.Tensor | None = None
        for j, w in zip(contributors, weights):
            lu_p = all_preds[j][unit_id_int][2]
            contribution = w * lu_p
            rul = contribution if rul is None else rul + contribution

        return rul, contributors, weights

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
        self.weights = self._compute_weights(x_test, target, criteria_callback, mode)

        self._log(1, f"[CoTraining] Weights assigned: "
                     f"{[f'model {j}={round(w, 4)}' for j, w in enumerate(self.weights)]}")

    def _compute_weights(
            self,
            x_test: torch.Tensor,
            target: torch.Tensor,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float],
            mode: str,
            models: list[LightningModule] | None = None,
    ) -> list[float]:
        """
        Compute the ensemble weights for the current models and return them **without**
        mutating ``self.weights``.

        This holds the scoring→weight math shared by ``calculate_weights`` (which stores
        the result) and the per-stage metrics logging in ``train`` (which needs weights
        for the "with weights" test metrics without changing the ensemble's state).

        Args:
            x_test: features to score the models on (the validation set is used at the
                call sites, to avoid leaking test information into the weighting).
            target: labels associated with ``x_test``.
            criteria_callback: per-model score, lower-is-better for ``mode="min"``.
            mode: "min" (lower score is better → inverse weighting) or "max".
            models: the models to weight. Defaults to ``self.lightning_modules``. During
                ``train`` the trained models live in a local list (``h``) and are only
                assigned to ``self.lightning_modules`` at the very end, so the per-stage
                metrics must pass that list explicitly — otherwise the weights would be
                derived from the still-untrained template modules.

        Returns:
            list[float]: one normalized weight per model, summing to 1.
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")

        if models is None:
            models = self.lightning_modules

        scores = []

        # Flatten both sides so the criteria callback compares aligned (N,) vectors
        # instead of broadcasting (N, 1) against (N,) into an (N, N) matrix.
        target_flat = target.view(-1).float()

        for model in models:
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
            return [inv_s / total for inv_s in inv_scores]

        total = sum(scores)
        if total == 0:
            raise ValueError(f"The sum of scores from all models is zero, cannot calculate weights : {scores}")
        return [s / total for s in scores]
