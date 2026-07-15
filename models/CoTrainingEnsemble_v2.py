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

from models.coprog_gpu_pool import TrainingSpec, run_training_job
from models.cotraining_gpu_pool import (
    CoTrainingGpuPool,
    ConformalScoreSpec,
    _TorchRegressorAdapter,
    run_conformal_score_job,
)


class CoTrainingEnsemble_v2:
    """
    Confidence-based co-training ensemble (v2).

    Instead of measuring, for every candidate censored unit, how much a retrained/fine-tuned
    model improves on the labelled set (the expensive "delta" of ``CoTrainingEnsemble``), this
    version ranks censored units by the *width* of a conformal prediction interval produced with
    ``crepes``. Each already-trained model is wrapped in a normalized conformal regressor
    (calibrated on the validation set with a kNN ``DifficultyEstimator`` so widths vary per unit);
    a narrower interval means the model is more confident about that unit. Models are then
    retrained from scratch with the newly pseudo-labelled units.
    """

    def __init__(
            self,
            models: list[nn.Module],
            weights: list[float] | None = None,
            verbose: int = 0,
            confidence: float = 0.95,
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
        """
        if weights is not None and len(models) != len(weights):
            raise ValueError("The number of weights must be the same as the number of models.")

        if not (0 < confidence < 1):
            raise ValueError("confidence must be in the range (0, 1).")

        self.models = models
        self.number_of_models = len(self.models)
        self.lightning_modules = None
        self.trainer_factories = None
        self.batchs_size = None
        self.shuffle_dataloaders = None
        self.weights = weights
        self.verbose = verbose
        self.confidence = confidence

        # Builder-style config (set through setup_training_builder), required for multi-GPU
        # parallel training. Mirrors models.Coprog / CoTrainingEnsemble (v1).
        self.module_builders: list[Callable[[], LightningModule]] | None = None
        self.max_epochs: list[int] | None = None
        self.patiences: list[int] | None = None
        self.gpu_ids: list[int] | None = None
        self._initial_state_dicts: list[dict[str, torch.Tensor]] | None = None
        self._use_builders: bool = False
        self._parallel: bool = False
        self._inline_accelerator: str = "auto"
        self._inline_devices = None
        self._configured: bool = False

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

        # Legacy style: sequential training in the current process.
        self._use_builders = False
        self._parallel = False
        self._configured = True

    def setup_training_builder(
            self,
            module_builders: list[Callable[[], LightningModule]],
            max_epochs: list[int],
            patiences: list[int],
            batchs_size: list[int],
            shuffle_dataloaders: list[bool],
            gpu_ids: list[int] | None = None,
    ) -> None:
        r"""Setup **builder-style** training, the style required for multi-GPU parallel training.

        A fresh module is built from a *picklable* builder, its initial weights are pinned to a
        one-time snapshot, and the ``Trainer`` is built internally. Each list has one entry per
        model (same order as ``models``). v2 always trains from scratch (no fine-tuning), so —
        unlike v1 — there are no fine-tune epoch/patience or freezing arguments.

        Args:
            module_builders: Picklable callables (module-level functions or ``functools.partial``
                — no lambdas/closures) each returning a *fresh* ``LightningModule``.
            max_epochs: Max training epochs per model.
            patiences: ``EarlyStopping`` patience per model.
            batchs_size: Batch size used to train each model.
            shuffle_dataloaders: Whether to shuffle each training ``DataLoader``.
            gpu_ids: Physical GPU ids to train on. ``None`` → auto (one GPU), sequential;
                ``[g]`` → pin to GPU ``g``, sequential; ``[g0, g1, ...]`` (>=2) → parallel:
                every independent job (per-model training, per-model conformal scoring) is
                distributed round-robin across all listed GPUs.

        Raises:
            ValueError: If a list does not have one entry per model.
        """
        model_number = len(self.models)
        if (len(module_builders) != model_number or len(max_epochs) != model_number
                or len(patiences) != model_number or len(batchs_size) != model_number
                or len(shuffle_dataloaders) != model_number):
            raise ValueError(
                f"module_builders, max_epochs, patiences, batchs_size and shuffle_dataloaders "
                f"must all have length {model_number}.")

        self.module_builders = module_builders
        self.max_epochs = max_epochs
        self.patiences = patiences
        self.batchs_size = batchs_size
        self.shuffle_dataloaders = shuffle_dataloaders
        self.gpu_ids = list(gpu_ids) if gpu_ids else None

        # Snapshot one initial weight set per model so every from-scratch training starts from
        # identical weights and workers can reproduce that init across process boundaries.
        self._initial_state_dicts = []
        for builder in module_builders:
            template = builder()
            self._initial_state_dicts.append(
                {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
            )

        if self.gpu_ids is None:
            self._parallel = False
            self._inline_accelerator = "auto"
            self._inline_devices = None
        elif len(self.gpu_ids) == 1:
            self._parallel = False
            self._inline_accelerator = "gpu"
            self._inline_devices = [self.gpu_ids[0]]
        else:
            self._parallel = True
            self._inline_accelerator = "gpu"
            self._inline_devices = [self.gpu_ids[0]]

        self._use_builders = True
        self._configured = True

    def train(
            self,
            train_with_censored_data: bool,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: float,
            add_ratio: float,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            test_data: torch.Tensor | None = None,
            test_label: torch.Tensor | None = None,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
            weight_mode: str = "min",
            metrics_file: str | None = None,
            log_file: str | None = None,
    ) -> None:
        r"""The train algorithm for co-training ensemble v2.

        Each iteration scores **all** remaining censored units with a conformal prediction
        interval per model (calibrated on the validation set), normalizes each model's widths by
        its own median (widths aren't comparable across models, since each calibrates its own
        regressor), selects for every model the unit whose average normalized width across the
        *other* models is smallest (most confident), and retrains all models from scratch on
        their newly pseudo-labelled data.

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
            suspension_pool_size:
                Fraction (in ``(0, 1]``) of the total censored units to draw at random as the
                candidate pool each iteration. ``>= 1.0`` means the whole remaining set is used.
                The pool is re-sampled every iteration so the same units are not always scored
                (also avoids scoring every remaining unit's conformal interval each iteration).
            add_ratio:
                Fraction (in ``(0, 1]``) of the sampled pool to actually add per iteration. The
                per-iteration add count is ``max(1, round(add_ratio * pool_size))`` units, shared
                across all models and handed out round-robin (with a rotating starting model) so no
                single model consistently gets first pick of the most confident censored units.
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
        # mandatory in v2 (it is also reused for early stopping and the weighted metrics).
        if val_data is None or val_label is None:
            raise ValueError(
                "val_data and val_label are required in v2 (used to calibrate the conformal regressors).")

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

        if not (0 < suspension_pool_size <= 1):
            raise ValueError("suspension_pool_size must be a fraction in (0, 1].")
        if not (0 < add_ratio <= 1):
            raise ValueError("add_ratio must be a fraction in (0, 1].")

        total_suspension_units = len(torch.unique(suspension_ids))
        # Candidate pool size (a count of units) derived once from the fraction; the actual pool is
        # re-sampled at random from the remaining units every iteration.
        if suspension_pool_size >= 1.0:
            pool_size = total_suspension_units
        else:
            pool_size = max(1, round(suspension_pool_size * total_suspension_units))
        self._log(1, f"[CoTraining] Starting training | models: {self.number_of_models} | "
                     f"failure samples: {len(failure_data)} | "
                     f"censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | confidence: {self.confidence} | "
                     f"pool fraction: {suspension_pool_size} (size: {pool_size}) | "
                     f"add ratio: {add_ratio} | "
                     f"mode: {'parallel(' + str(self.gpu_ids) + ')' if self._parallel else 'sequential'}")

        if self._parallel:
            self._train_parallel(
                failure_data=failure_data,
                failure_label=failure_label,
                suspension_data=suspension_data,
                suspension_ids=suspension_ids,
                iterations=iterations,
                pool_size=pool_size,
                add_ratio=add_ratio,
                val_data=val_data,
                val_label=val_label,
                metrics_enabled=metrics_enabled,
                test_data=test_data,
                test_label=test_label,
                criteria_callback=criteria_callback,
                weight_mode=weight_mode,
                metrics_file=metrics_file,
            )
            return

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
            h_j = self._fit_from_scratch(j, x_i, y_i, val_data, val_label)

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

            # Draw a random candidate pool from the remaining units. Re-sampling every iteration
            # means the same units are not always scored, and (unlike scoring all remaining units)
            # it bounds the per-iteration conformal cost.
            pool_size_iter = min(pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
            pool_ids = shuffled_ids[:pool_size_iter]

            self._log(1, f"[CoTraining] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            # Phase 1 — for each model j, build a conformal regressor (calibrated on the
            # validation set) and score every pooled censored unit by the width of the
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
                             f"scoring {len(pool_ids)} pooled censored units...")

                wrapper = self._build_calibrated_regressor(hj, xj, val_data, val_label)

                # Collect each unit's rows, its per-window pseudo-labels and its last window
                # (the interval of that final window is the unit's "range").
                unit_ids = list(pool_ids)
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
            # regressor on its own accumulated data), so normalize each model's widths by its
            # own median before comparing them across models.
            norm_width = self._normalized_widths(all_preds)

            # Phase 2 — hand out a shared per-iteration budget of censored units round-robin
            # over the models, rotating the starting model each iteration so no single model
            # consistently gets first pick of the most confident units. Each model still selects
            # from its peers' scores (co-training); a chosen unit is removed from every model's
            # map so it cannot be reused this iteration. A model may receive more than one unit.
            n_add = max(1, round(add_ratio * len(pool_ids)))
            start = i % self.number_of_models
            selected_per_model, remaining_suspension_ids, added = self._assign_units_round_robin(
                all_preds=all_preds,
                norm_width=norm_width,
                n_add=n_add,
                start=start,
                remaining_suspension_ids=remaining_suspension_ids,
            )

            if added == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: "
                             f"no censored unit available for any model.")
                break

            for j in range(self.number_of_models):
                if selected_per_model[j]:
                    xj, yj = models_datasets[j]

                    new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)

                    xj = torch.cat([xj, new_xu], dim=0)
                    yj = torch.cat([yj, new_lu], dim=0)

                    models_datasets[j] = (xj, yj)

                    self._log(1, f"[CoTraining]   Retraining model {j} from scratch | "
                                 f"added {len(selected_per_model[j])} unit(s) | "
                                 f"dataset size: {len(xj)} samples")

                    h[j] = self._fit_from_scratch(j, xj, yj, val_data, val_label)

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

    # ------------------------------------------------------------------ #
    # Shared phase helpers (used by both the sequential and parallel paths)
    # ------------------------------------------------------------------ #

    def _assign_units_round_robin(
            self,
            all_preds: dict[int, OrderedDict],
            norm_width: dict[int, dict[int, float]],
            n_add: int,
            start: int,
            remaining_suspension_ids: torch.Tensor,
    ) -> tuple[list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], torch.Tensor, int]:
        """Phase 2: hand out a shared per-iteration budget of censored units round-robin.

        Rotating the starting model each iteration so no single model consistently gets first
        pick of the most confident units. A chosen unit is removed from every model's candidate
        and normalized-width maps so it cannot be reused this iteration.

        Returns ``(selected_per_model, remaining_suspension_ids, added)``.
        """
        selected_per_model: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.number_of_models)
        ]
        added = 0
        slot = 0
        consecutive_failures = 0
        self._log(1, f"[CoTraining]   Adding up to {n_add} unit(s) this iteration "
                     f"(round-robin from model {start}).")
        while added < n_add and consecutive_failures < self.number_of_models:
            k = (start + slot) % self.number_of_models
            slot += 1

            picked = self._voting_censored_data_selection(
                all_preds=all_preds,
                norm_width=norm_width,
                model_index_to_exclude=k,
            )

            if picked is None:
                # No censored unit currently available for model k.
                consecutive_failures += 1
                continue

            consecutive_failures = 0
            selected_id = picked[0].item()
            selected_per_model[k].append(picked)
            added += 1
            self._log(1, f"[CoTraining]   Model {k}: selected unit {selected_id} ")

            for j in range(self.number_of_models):
                all_preds[j].pop(selected_id, None)
                norm_width[j].pop(selected_id, None)

            remaining_suspension_ids = remaining_suspension_ids[
                remaining_suspension_ids != picked[0]
                ]

        return selected_per_model, remaining_suspension_ids, added

    @staticmethod
    def _concat_selected_units(
            selected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            yj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate every ``(unit_id, xu, lu)`` newly assigned to a model into ``(new_xu, new_lu)``."""
        new_xu = torch.cat([xu for _, xu, _ in selected], dim=0)
        new_lu = torch.cat(
            [lu.view(-1, yj.shape[1]) if yj.dim() > 1 else lu.view(-1) for _, _, lu in selected],
            dim=0,
        )
        return new_xu, new_lu

    # ------------------------------------------------------------------ #
    # Training dispatcher + builder-style spec helpers
    # ------------------------------------------------------------------ #

    def _fit_from_scratch(
            self,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None,
            val_y: torch.Tensor | None,
    ) -> LightningModule:
        """Train one model from scratch (inline, this process).

        Uses the builder path (:func:`run_training_job` + rebuild from the returned CPU state
        dict) when configured via :meth:`setup_training_builder`, otherwise the legacy path
        (:meth:`_train_fun` on a deep-copied template).
        """
        if self._use_builders:
            spec = self._make_fit_spec(model_index, x, y, self._cpu_pair(val_x, val_y))
            spec.accelerator = self._inline_accelerator
            spec.devices = self._inline_devices
            result = run_training_job(spec)
            return self._rebuild_module(model_index, result["state_dict"])
        return self._train_fun(
            copy.deepcopy(self.lightning_modules[model_index]), model_index, x, y, val_x, val_y)

    def _make_fit_spec(
            self,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> TrainingSpec:
        """Build a picklable :class:`TrainingSpec` for a from-scratch training that returns state."""
        return TrainingSpec(
            module_builder=self.module_builders[model_index],
            initial_state_dict=self._initial_state_dicts[model_index],
            max_epochs=self.max_epochs[model_index],
            patience=self.patiences[model_index],
            batch_size=self.batchs_size[model_index],
            shuffle=self.shuffle_dataloaders[model_index],
            train_x=x.detach().cpu(),
            train_y=y.detach().cpu(),
            val_x=val_cpu[0],
            val_y=val_cpu[1],
            return_state=True,
            accelerator="gpu",
            devices=1,
        )

    def _rebuild_module(self, model_index: int, state_dict: dict[str, torch.Tensor]) -> LightningModule:
        """Rebuild a model in this (main) process from a CPU state dict, for inference only."""
        module = self.module_builders[model_index]()
        module.load_state_dict(state_dict)
        return module

    @staticmethod
    def _cpu_pair(
            a: torch.Tensor | None,
            b: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Detach-and-move an optional tensor pair to CPU (for picklable specs)."""
        a_cpu = a.detach().cpu() if a is not None else None
        b_cpu = b.detach().cpu() if b is not None else None
        return a_cpu, b_cpu

    # ------------------------------------------------------------------ #
    # Parallel training path
    # ------------------------------------------------------------------ #

    def _train_parallel(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            pool_size: int,
            add_ratio: float,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            metrics_enabled: bool,
            test_data: torch.Tensor | None,
            test_label: torch.Tensor | None,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float] | None,
            weight_mode: str,
            metrics_file: str | None,
    ) -> None:
        """Multi-GPU parallel version of :meth:`train`'s algorithm.

        The three per-model phases — initial from-scratch training, conformal scoring
        (prediction on the censored units + ``crepes`` interval computation), and the
        end-of-iteration from-scratch retrain — each run concurrently across all ``gpu_ids``
        via a :class:`~models.cotraining_gpu_pool.CoTrainingGpuPool`, distributed round-robin.
        Width normalization and the round-robin voting selection stay in the main process.
        Only CPU state dicts / small per-unit results cross the process boundary.
        """
        n = self.number_of_models
        val_cpu = self._cpu_pair(val_data, val_label)

        pool = CoTrainingGpuPool(self.gpu_ids)
        pool.start()
        try:
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]] = [
                (failure_data, failure_label) for _ in range(n)
            ]

            # --- Initial training: one from-scratch job per model, round-robin. ---
            self._log(1, f"[CoTraining] Initial parallel training of {n} models...")
            job_ids = {
                j: pool.submit_job(pool.round_robin_gpu(j), self._make_fit_spec(j, failure_data, failure_label, val_cpu))
                for j in range(n)
            }
            results = pool.gather(list(job_ids.values()))
            h = [self._rebuild_module(j, results[job_ids[j]]["state_dict"]) for j in range(n)]
            self._log(1, f"[CoTraining] Initial training done.")

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="initial", h=h, models_datasets=models_datasets,
                    test_data=test_data, test_label=test_label, val_data=val_data,
                    val_label=val_label, criteria_callback=criteria_callback,
                    weight_mode=weight_mode, metrics_file=metrics_file,
                )

            remaining_suspension_ids = torch.unique(suspension_ids)

            for i in range(iterations):
                if len(remaining_suspension_ids) == 0:
                    self._log(1, f"[CoTraining] Early stop at iteration {i}: no remaining censored units.")
                    break

                pool_size_iter = min(pool_size, len(remaining_suspension_ids))
                shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
                pool_ids = shuffled_ids[:pool_size_iter]

                self._log(1, f"[CoTraining] --- Iteration {i + 1}/{iterations} | "
                             f"remaining censored units: {len(remaining_suspension_ids)} | "
                             f"pool: {pool_ids.tolist()} ---")

                # --- Phase 1: per-model conformal scoring in parallel (one job per model). ---
                unit_ids_int = [int(uid.item()) for uid in pool_ids]
                # xu is identical across models for a given unit, so build the per-unit
                # sequences once and reuse them for every model's job and for all_preds.
                unit_x = [suspension_data[suspension_ids == uid].detach().cpu() for uid in pool_ids]

                score_jobs: dict[int, int] = {}
                for j in range(n):
                    xj, yj = models_datasets[j]
                    self._log(2, f"[CoTraining]   Model {j}: submitting conformal scoring of "
                                 f"{len(pool_ids)} pooled units on GPU "
                                 f"{pool.round_robin_gpu(j)}...")
                    spec = ConformalScoreSpec(
                        module_builder=self.module_builders[j],
                        state_dict={k: v.detach().cpu().clone() for k, v in h[j].state_dict().items()},
                        train_x=xj.detach().cpu(),
                        val_x=val_cpu[0],
                        val_y=val_cpu[1],
                        unit_ids=unit_ids_int,
                        unit_x=unit_x,
                        confidence=self.confidence,
                        accelerator="gpu",
                        devices=1,
                    )
                    score_jobs[j] = pool.submit_conformal(pool.round_robin_gpu(j), spec)

                results = pool.gather(list(score_jobs.values()))

                # Rebuild all_preds[j] from each worker's per-unit (unit_id, lu_p, lower, upper, width).
                all_preds: dict[int, OrderedDict] = {}
                xu_by_unit = {uid: xu for uid, xu in zip(unit_ids_int, unit_x)}
                for j in range(n):
                    units = results[score_jobs[j]]["units"]
                    candidates = []
                    for unit_id_int, lu_p, lower, upper, width in units:
                        uid_tensor = torch.tensor(unit_id_int)
                        candidates.append(
                            (uid_tensor, xu_by_unit[unit_id_int], lu_p, lower, upper, width))
                    candidates.sort(key=lambda e: e[5])
                    all_preds[j] = OrderedDict(
                        (uid.item(), (uid, xu, lu_p, lower, upper, width))
                        for uid, xu, lu_p, lower, upper, width in candidates
                    )
                    if self.verbose >= 2:
                        ranking = [(uid.item(), round(w, 4)) for uid, _, _, _, _, w in candidates]
                        self._log(2, f"[CoTraining]   Model {j} candidate ranking "
                                     f"(most confident first): {ranking}")

                norm_width = self._normalized_widths(all_preds)

                # --- Phase 2: selection (main process). ---
                n_add = max(1, round(add_ratio * len(pool_ids)))
                start = i % n
                selected_per_model, remaining_suspension_ids, added = self._assign_units_round_robin(
                    all_preds=all_preds,
                    norm_width=norm_width,
                    n_add=n_add,
                    start=start,
                    remaining_suspension_ids=remaining_suspension_ids,
                )

                if added == 0:
                    self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: "
                                 f"no censored unit available for any model.")
                    break

                # --- Phase 3: retrain the updated models from scratch, in parallel. ---
                retrain_jobs: dict[int, int] = {}
                for j in range(n):
                    if selected_per_model[j]:
                        xj, yj = models_datasets[j]
                        new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)
                        xj = torch.cat([xj, new_xu], dim=0)
                        yj = torch.cat([yj, new_lu], dim=0)
                        models_datasets[j] = (xj, yj)

                        self._log(1, f"[CoTraining]   Retraining model {j} from scratch | "
                                     f"added {len(selected_per_model[j])} unit(s) | "
                                     f"dataset size: {len(xj)} samples")
                        retrain_jobs[j] = pool.submit_job(
                            pool.round_robin_gpu(j), self._make_fit_spec(j, xj, yj, val_cpu))

                results = pool.gather(list(retrain_jobs.values()))
                for j, jid in retrain_jobs.items():
                    h[j] = self._rebuild_module(j, results[jid]["state_dict"])

                if metrics_enabled:
                    self._log_stage_metrics(
                        stage=f"iteration_{i + 1}", h=h, models_datasets=models_datasets,
                        test_data=test_data, test_label=test_label, val_data=val_data,
                        val_label=val_label, criteria_callback=criteria_callback,
                        weight_mode=weight_mode, metrics_file=metrics_file,
                    )

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="final", h=h, models_datasets=models_datasets,
                    test_data=test_data, test_label=test_label, val_data=val_data,
                    val_label=val_label, criteria_callback=criteria_callback,
                    weight_mode=weight_mode, metrics_file=metrics_file,
                )

            self._log(1, f"[CoTraining] Training complete.")
            self.lightning_modules = h
        finally:
            pool.shutdown()

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

    def predict_per_model(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Per-model (unweighted) predictions, one tensor per model in the ensemble.

        Unlike :meth:`predict`, no weighting is applied — this is used to report each model's
        own test metrics (e.g. the per-model test RMSE columns of the summary CSV).

        :param x: torch.Tensor
            Input features of shape ``(N, *feature_dims)``.
        :return: list[torch.Tensor]
            One prediction tensor per model, each of shape ``(N, *output_dims)``.
        """
        return [self._predict(model, x) for model in self.lightning_modules]

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
        typical). This normalized width drives both the consensus average and the most-confident
        peer selection in ``_voting_censored_data_selection``.

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

    def _voting_censored_data_selection(
            self,
            all_preds: dict[int, OrderedDict],
            norm_width: dict[int, dict[int, float]],
            model_index_to_exclude: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        r"""Select the censored unit the *other* models are most confident about.

        For each candidate unit u, the average **normalized** interval width (see
        ``_normalized_widths``) is computed over all models j ≠ k. Normalizing first matters
        because each model calibrates its own conformal regressor independently, so raw widths
        aren't on a comparable scale — a systematically noisier model would otherwise dominate
        the average regardless of which unit is actually easiest for it. The unit with the
        *smallest* average normalized width (i.e. the tightest, most confident consensus among
        the other models) is selected.

        The pseudo-label assigned to model k comes from the single model j ≠ k with the smallest
        *normalized* interval width for that unit — the most confident peer — using its
        per-window RUL predictions. Model k's own predictions are ignored so it genuinely learns
        from its peers (the co-training principle).

        Args:
            all_preds: mapping from model index j to an OrderedDict of
                ``{unit_id_int: (unit_id_tensor, xu, lu_p, lower, upper, width)}``.
            norm_width: mapping from model index j to ``{unit_id_int: normalized_width}``, as
                returned by ``_normalized_widths``.
            model_index_to_exclude: index k of the model being updated — its own
                predictions are excluded from both the width average and the pseudo-label.

        Returns:
            ``(unit_id, xu, lu_p)`` for the selected unit, or ``None`` if no unit is
            available.
        """
        # Collect every unit_id that at least one non-excluded model has scored.
        all_unit_ids: set[int] = {
            uid
            for j, preds in all_preds.items()
            if j != model_index_to_exclude
            for uid in preds
        }

        best_avg_width: float | None = None
        best_unit_id_int: int | None = None

        for unit_id_int in all_unit_ids:
            widths = [
                norm_width[j][unit_id_int]
                for j in all_preds
                if j != model_index_to_exclude and unit_id_int in all_preds[j]
            ]
            if not widths:
                continue
            avg_width = sum(widths) / len(widths)
            if best_avg_width is None or avg_width < best_avg_width:
                best_avg_width = avg_width
                best_unit_id_int = unit_id_int

        if best_unit_id_int is None:
            return None

        # The pseudo-label comes from the most confident peer (smallest normalized interval
        # width among models j ≠ k) — not an average — as specified for v2.
        contributors = [
            j for j in all_preds
            if j != model_index_to_exclude and best_unit_id_int in all_preds[j]
        ]
        most_confident_j = min(contributors, key=lambda j: norm_width[j][best_unit_id_int])

        unit_id, xu, lu_p, _, _, _ = all_preds[most_confident_j][best_unit_id_int]

        self._log(1, f"[CoTraining]     Unit {best_unit_id_int} selected for model {model_index_to_exclude} | "
                     f"best avg normalized width = {best_avg_width:.4f}")

        num_sequences = lu_p.view(-1).shape[0]

        self._log(1, f"[CoTraining]     Unit {best_unit_id_int} width per peer model (raw / normalized):")
        for j in contributors:
            self._log(1, f"[CoTraining]     \tmodel {j}: {all_preds[j][best_unit_id_int][5]:.4f} / "
                         f"{norm_width[j][best_unit_id_int]:.4f}")

        self._log(1, f"[CoTraining]     Unit {best_unit_id_int} RUL predicted by all peer models "
                     f"({num_sequences} sequences for this unit):")
        for j in contributors:
            self._log(1, f"[CoTraining]     \tmodel {j}: "
                         f"{[round(v, 4) for v in all_preds[j][best_unit_id_int][2].view(-1).tolist()]}")

        self._log(1, f"[CoTraining]     Unit {best_unit_id_int} chosen RUL ({num_sequences} sequences for this "
                     f"unit, from most confident peer, model {most_confident_j}): "
                     f"{[round(v, 4) for v in lu_p.view(-1).tolist()]}")

        return unit_id, xu, lu_p

    def _check_if_training_is_possible(self):
        if not self._configured:
            raise ValueError("You need to call setup_training or setup_training_builder before calling train.")

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
