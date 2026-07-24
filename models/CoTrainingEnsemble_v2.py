import copy
import csv
import gc
import os
from collections import OrderedDict
from typing import Callable

import numpy as np
import sklearn
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
    FineTuneSpec,
    _TorchRegressorAdapter,
    _monotone_project,
    run_conformal_score_job,
    run_finetune_job,
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
            inference_batch_size: int | None = None,
            use_monotone_projection: bool = False,
            monotone_residual_weight: float = 1.0,
            use_fine_tuning: bool = False,
            fine_tune_lr_factor: float = 0.1,
            fine_tune_max_epochs: int = 20,
            fine_tune_patience: int = 5,
            peer_weighted_pseudo_label: bool = False,
            keep_best_model: bool = False,
            isotonic_time_weighting: bool = False,
            bagging_failure_data: bool = False,
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
        :param inference_batch_size: int | None
            If set, ``_predict`` runs forward passes in chunks of this many samples instead of
            feeding the whole tensor to the model at once. This caps peak (host/GPU) memory to
            ``O(batch)`` during the conformal scoring and per-stage metrics, which is what keeps
            the run within a small (e.g. Colab T4, 12.7 GB RAM) budget. ``None`` keeps the
            legacy single-shot behavior. Numerically identical either way.
        :param use_monotone_projection: bool
            When ``True``, each censored unit's per-window pseudo-labels are projected onto the
            closest non-increasing sequence (isotonic regression), optionally clipped up to the
            per-window survival lower bound (when ``suspension_lower_bounds`` is passed to
            ``train``). The projected sequence becomes the injected pseudo-label, and the
            projection *residual* (how far the raw predictions had to move to become physically
            valid) is blended into the unit-selection score. ``False`` (default) keeps the
            legacy width-only scoring and injects the raw predictions unchanged.
        :param monotone_residual_weight: float
            Weight ``lambda`` of the residual term in the blended selection score
            ``width_norm + lambda * residual_norm`` (both terms per-model normalized). Only used
            when ``use_monotone_projection`` is ``True``; ``0`` disables the residual term (the
            projection still cleans the injected labels but selection stays width-only).
        :param use_fine_tuning: bool
            When ``True``, each receiver model is **warm-started from its current weights and
            fine-tuned** on its grown dataset every iteration instead of being retrained from
            scratch (mirrors ``CoTrainingEnsemble_v3``). Requires the builder path
            (``setup_training_builder``) and the sequential path (not multi-GPU parallel).
            ``False`` (default) keeps the legacy from-scratch retraining.
        :param fine_tune_lr_factor: float
            Multiplier applied to each model's learning rate during a fine-tune (e.g. ``0.1``).
            Only used when ``use_fine_tuning`` is ``True``. Must be ``> 0``.
        :param fine_tune_max_epochs: int
            Max epochs per fine-tuning call. Only used when ``use_fine_tuning`` is ``True``.
        :param fine_tune_patience: int
            ``EarlyStopping`` patience (monitors ``val_loss``) per fine-tuning call. Only used
            when ``use_fine_tuning`` is ``True``.
        :param peer_weighted_pseudo_label: bool
            When ``True``, a selected unit's pseudo-label is the **confidence-weighted
            average** of the per-window RUL predictions of *all* peer models ``j != k`` that
            scored the unit (weight ``prop 1 / norm_width_j**2`` — each peer's own per-unit,
            median-normalized conformal-interval score, the same one used to rank candidates;
            a tighter/more confident peer contributes more), instead of taking the single
            most-confident peer's prediction. Unit selection (tightest average normalized
            width) is unchanged. ``False`` (default) keeps the single most-confident-peer label.
        :param keep_best_model: bool
            When ``True``, after each iteration a receiver's fine-tuned/retrained candidate is
            **kept only if its validation RMSE strictly improves** on that model's best so far;
            otherwise the previous model, its dataset and its best RMSE are retained and the
            iteration's added units are dropped for good (mirrors ``CoTrainingEnsemble_v3``).
            ``False`` (default) always accepts the candidate (legacy behavior).
        :param isotonic_time_weighting: bool
            When ``True``, the monotone (isotonic) projection of each unit's pseudo-labels is
            fitted with per-window ``sample_weight`` proportional to the local time gap
            ``Delta t`` (central gap), so temporally isolated windows are treated as more
            independent. Requires ``use_monotone_projection=True`` and a ``suspension_time_steps``
            tensor passed to ``train``. ``False`` (default) uses the unweighted projection.
        :param bagging_failure_data: bool
            When ``True``, each model's *initial* training set (before any censored unit is
            added) is an independent bootstrap resample of ``failure_data``/``failure_label``
            (``N`` draws with replacement from the ``N`` failure rows, classic bagging), instead
            of every model sharing the exact same failure dataset. Only affects the initial
            dataset; units added during co-training are unaffected. ``False`` (default) keeps
            the legacy behavior of all models starting from the identical failure dataset.
        """
        if weights is not None and len(models) != len(weights):
            raise ValueError("The number of weights must be the same as the number of models.")

        if not (0 < confidence < 1):
            raise ValueError("confidence must be in the range (0, 1).")

        if monotone_residual_weight < 0:
            raise ValueError("monotone_residual_weight must be non-negative.")

        if fine_tune_lr_factor <= 0:
            raise ValueError("fine_tune_lr_factor must be positive.")

        self.models = models
        self.number_of_models = len(self.models)
        self.lightning_modules = None
        self.trainer_factories = None
        self.batchs_size = None
        self.shuffle_dataloaders = None
        self.weights = weights
        self.verbose = verbose
        self.confidence = confidence
        self._inference_batch_size = inference_batch_size
        self.use_monotone_projection = use_monotone_projection
        self.monotone_residual_weight = monotone_residual_weight
        self.use_fine_tuning = use_fine_tuning
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.fine_tune_max_epochs = fine_tune_max_epochs
        self.fine_tune_patience = fine_tune_patience
        self.peer_weighted_pseudo_label = peer_weighted_pseudo_label
        self.keep_best_model = keep_best_model
        self.isotonic_time_weighting = isotonic_time_weighting
        self.bagging_failure_data = bagging_failure_data

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
            calib_data: torch.Tensor | None = None,
            calib_label: torch.Tensor | None = None,
            suspension_lower_bounds: torch.Tensor | None = None,
            suspension_time_steps: torch.Tensor | None = None,
            test_data: torch.Tensor | None = None,
            test_label: torch.Tensor | None = None,
            score_callback: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
            weight_callback: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
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
                Validation features. Required: used for early stopping / best-checkpoint
                selection during every training call, for the per-stage ensemble weighting/
                reporting, and as the fallback calibration set for the conformal regressors
                when ``calib_data`` is not given.
            val_label:
                Validation labels associated with ``val_data``. Required.
            calib_data:
                Optional features for a calibration set kept separate from ``val_data``. When
                given (together with ``calib_label``), the conformal regressors are calibrated
                on this set instead of ``val_data``, avoiding the anti-conservative (too-narrow)
                intervals that result from calibrating on the same data used for early-stopping
                model selection. Falls back to ``val_data`` when ``None`` (legacy behavior).
            calib_label:
                Calibration labels associated with ``calib_data``. Required together with
                ``calib_data``.
            suspension_lower_bounds:
                Optional per-window survival lower bounds for the censored data, row-aligned
                with ``suspension_data`` / ``suspension_ids`` (shape ``(N,)`` or ``(N, 1)``;
                value = time observed until end of study, so a model's predicted RUL should be
                ``>=`` it). Only used when ``use_monotone_projection`` is enabled, to clip each
                unit's projected pseudo-labels up to its bound. If projection is enabled but
                this is ``None``, projection falls back to monotonicity only (no censoring clip).
            suspension_time_steps:
                Optional per-window time steps for the censored data, row-aligned with
                ``suspension_data`` / ``suspension_ids`` (shape ``(N,)`` or ``(N, 1)``, ordered
                oldest -> newest within each unit). Only used when ``isotonic_time_weighting`` is
                enabled: the local time gap ``Delta t`` between a unit's windows becomes the
                ``sample_weight`` of its isotonic projection. Required when
                ``isotonic_time_weighting`` is ``True``.
            test_data:
                Optional test features used only for the per-stage metrics logged to
                ``metrics_file``. Never used for training or model selection.
            test_label:
                Optional test labels associated with ``test_data``.
            score_callback:
                Score reported per model, averaged and for the weighted ensemble in the
                ``test_score`` columns of the metrics file (e.g. the Scania score). Only
                used for reporting, never for weighting. Required when metrics logging is
                enabled.
            weight_callback:
                Score used to compute the per-stage ensemble weights on the validation
                set (e.g. RMSE). Should match the callback the caller passes to
                ``calculate_weights`` (lower is better for ``weight_mode="min"``).
                Required when metrics logging is enabled.
            weight_mode:
                "min" or "max", passed to ``_compute_weights`` when deriving the
                per-stage ensemble weights (defaults to "min").
            metrics_file:
                Optional path to a ``.csv`` file. When given (together with the test
                set, the validation set, ``score_callback`` and ``weight_callback``),
                per-stage metrics
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

        # The four opt-in levers (fine-tuning, peer-weighted pseudo-labels, keep-best-model,
        # time-weighted isotonic) are only wired into the sequential path; refuse to run them
        # silently under multi-GPU parallel where they would be ignored.
        new_features_on = (
            self.use_fine_tuning or self.peer_weighted_pseudo_label
            or self.keep_best_model or self.isotonic_time_weighting
        )
        if self._parallel and new_features_on:
            raise ValueError(
                "use_fine_tuning, peer_weighted_pseudo_label, keep_best_model and "
                "isotonic_time_weighting are only supported on the sequential path; they cannot "
                "be combined with multi-GPU parallel (gpu_ids with >= 2 GPUs).")

        # Fine-tuning warm-starts from the builder-rebuilt module and reuses the builder's
        # batch-size / shuffle config, so it requires setup_training_builder (like v3).
        if self.use_fine_tuning and not self._use_builders:
            raise ValueError(
                "use_fine_tuning requires setup_training_builder (the builder path).")

        # Time-weighted isotonic only has an effect inside the monotone projection and needs the
        # per-window time steps to derive the sample weights.
        if self.isotonic_time_weighting:
            if not self.use_monotone_projection:
                raise ValueError(
                    "isotonic_time_weighting requires use_monotone_projection=True.")
            if suspension_time_steps is None:
                raise ValueError(
                    "isotonic_time_weighting requires suspension_time_steps to be provided.")

        # val_data is mandatory in v2 (early stopping + weighted metrics, and the fallback
        # calibration set when calib_data isn't given).
        if val_data is None or val_label is None:
            raise ValueError(
                "val_data and val_label are required in v2 (used for early stopping / "
                "best-checkpoint selection).")
        if (calib_data is None) != (calib_label is None):
            raise ValueError("calib_data and calib_label must be provided together.")
        # Fall back to the validation set when no dedicated calibration set is given
        # (legacy behavior — see the calib_data/calib_label docstring for the tradeoff).
        calib_data_eff = calib_data if calib_data is not None else val_data
        calib_label_eff = calib_label if calib_label is not None else val_label

        # Per-stage metrics need the test set, the criteria callback and a destination file.
        # Enable only when all are present; if the caller asked for metrics (test_data given)
        # but left something out, fail loudly.
        metrics_enabled = test_data is not None
        if metrics_enabled:
            if test_label is None:
                raise ValueError("test_label must be provided together with test_data.")
            if score_callback is None:
                raise ValueError("score_callback is required to log per-stage metrics (per-model test score).")
            if weight_callback is None:
                raise ValueError("weight_callback is required to log per-stage metrics (per-stage ensemble weights).")
            if metrics_file is None:
                raise ValueError("metrics_file is required to log per-stage metrics.")

        if not (0 < suspension_pool_size <= 1):
            raise ValueError("suspension_pool_size must be a fraction in (0, 1].")
        if not (0 < add_ratio <= 1):
            raise ValueError("add_ratio must be a fraction in (0, 1].")

        # Monotone projection (v2 opt-in): warn once if the censoring clip was requested but no
        # lower bounds are available, then fall back to monotonicity-only. ``clip_bounds`` gates
        # the per-unit lower-bound slicing in both the sequential and parallel scoring paths.
        if self.use_monotone_projection and suspension_lower_bounds is None:
            self._log(1, "[CoTraining] use_monotone_projection is on but suspension_lower_bounds "
                         "was not provided; using monotone projection without the censoring clip.")
        clip_bounds = self.use_monotone_projection and suspension_lower_bounds is not None
        # Whether the isotonic projection is fitted with Delta-t sample weights (guarded above:
        # only reachable with use_monotone_projection and suspension_time_steps present).
        time_weight = self.isotonic_time_weighting and suspension_time_steps is not None

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
                suspension_lower_bounds=suspension_lower_bounds if clip_bounds else None,
                iterations=iterations,
                pool_size=pool_size,
                add_ratio=add_ratio,
                val_data=val_data,
                val_label=val_label,
                calib_data=calib_data_eff,
                calib_label=calib_label_eff,
                metrics_enabled=metrics_enabled,
                test_data=test_data,
                test_label=test_label,
                score_callback=score_callback,
                weight_callback=weight_callback,
                weight_mode=weight_mode,
                metrics_file=metrics_file,
            )
            return

        models_datasets = []
        h: list[LightningModule] = []
        # Per-model validation RMSE, maintained only when keep-best-model needs it to
        # accept/reject a candidate. None otherwise (peer-weighted pseudo-labels now use each
        # peer's per-unit confidence score instead, so they don't need this), so the
        # all-defaults path pays no extra forward pass.
        track_val_rmse = self.keep_best_model
        val_rmses: list[float] | None = [] if track_val_rmse else None

        for j in range(self.number_of_models):
            if self.bagging_failure_data:
                x_i, y_i = self._bootstrap_sample(failure_data, failure_label)
            else:
                x_i, y_i = failure_data, failure_label

            # TODO need to see how to deel with survloss and data
            # if train_with_censored_data:
            #     x_i = torch.cat([x_i, suspension_data], dim=0)
            #     y_i = torch.cat([y_i, ], dim=0)

            models_datasets.append((x_i, y_i))

            self._log(1, f"[CoTraining] Initial training of model {j} on {len(x_i)} failure samples...")
            h_j = self._fit_from_scratch(j, x_i, y_i, val_data, val_label)

            h.append(h_j)

            if track_val_rmse:
                val_rmses.append(self._mse_on(h_j, val_data, val_label) ** 0.5)

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
                score_callback=score_callback,
                weight_callback=weight_callback,
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
            #   all_preds[j] = OrderedDict{ unit_id_int -> (unit_id, xu, lu_p, lower, upper, width, residual, raw_lu_p) }
            # (lu_p is the monotone-projected label when projection is on; residual is 0.0 otherwise;
            #  raw_lu_p is the pre-projection prediction, kept for effectiveness logging)
            all_preds: dict[int, OrderedDict] = {}

            for j in range(self.number_of_models):
                hj = h[j]
                xj, yj = models_datasets[j]

                self._log(2, f"[CoTraining]   Model {j}: calibrating conformal regressor and "
                             f"scoring {len(pool_ids)} pooled censored units...")

                wrapper = self._build_calibrated_regressor(hj, xj, calib_data_eff, calib_label_eff)

                # Collect each unit's rows, its per-window pseudo-labels, its (optional) per-window
                # lower bounds and its last window (the interval of that final window is the unit's
                # "range").
                unit_ids = list(pool_ids)
                xus: list[torch.Tensor] = []
                lu_ps: list[torch.Tensor] = []
                lb_us: list[torch.Tensor | None] = []
                sw_us: list[np.ndarray | None] = []
                last_windows: list[torch.Tensor] = []
                for unit_id in unit_ids:
                    mask = (suspension_ids == unit_id)
                    xu = suspension_data[mask]
                    xus.append(xu)
                    lu_ps.append(self._predict(hj, xu))
                    lb_us.append(suspension_lower_bounds[mask] if clip_bounds else None)
                    sw_us.append(
                        self._time_gap_sample_weight(suspension_time_steps[mask])
                        if time_weight else None)
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
                    # When enabled, project this unit's per-window predictions onto a
                    # non-increasing (and lower-bound-clipped) sequence; the projected labels
                    # replace the raw predictions and the residual feeds the selection score.
                    if self.use_monotone_projection:
                        label, residual = _monotone_project(
                            lu_ps[idx], lb_us[idx], sample_weight=sw_us[idx])
                    else:
                        label, residual = lu_ps[idx], 0.0
                    # Keep the raw (pre-projection) prediction as the last field for logging.
                    candidates.append(
                        (unit_id, xus[idx], label, lower, upper, width, residual, lu_ps[idx]))

                # Smaller width = more confident, so sort ascending (best candidate first).
                candidates.sort(key=lambda e: e[5])
                all_preds[j] = OrderedDict(
                    (uid.item(), (uid, xu, lu_p, lower, upper, width, residual, raw_lu_p))
                    for uid, xu, lu_p, lower, upper, width, residual, raw_lu_p in candidates
                )

                if self.verbose >= 2:
                    ranking = [(uid.item(), round(w, 4)) for uid, _, _, _, _, w, _, _ in candidates]
                    self._log(2, f"[CoTraining]   Model {j} candidate ranking (most confident first): "
                                 f"{ranking}")

                # Release this model's calibrated regressor (and the fitted kNN
                # DifficultyEstimator + train-set copies it holds) before the next model
                # builds its own, so at most one is alive at a time instead of all four.
                del wrapper, xus, lu_ps, lb_us, sw_us, last_windows, last_windows_tensor, intervals, candidates
                gc.collect()

            # Widths are not comparable across models (each model calibrates its own conformal
            # regressor on its own accumulated data), so normalize each model's widths by its
            # own median before comparing them across models. When monotone projection is on,
            # the median-normalized projection residual is blended in (see _selection_scores).
            norm_width = self._selection_scores(all_preds)

            # Log each model's full unit ranking (most confident first) so it can be checked
            # whether the models agree on which censored units are confident or not.
            self._log_confidence_ranking(all_preds, norm_width)

            # Phase 2 — hand out a shared per-iteration budget of censored units round-robin
            # over the models, rotating the starting model each iteration so no single model
            # consistently gets first pick of the most confident units. Each model still selects
            # from its peers' scores (co-training); a chosen unit is removed from every model's
            # map so it cannot be reused this iteration. A model may receive more than one unit.
            n_add = max(1, round(add_ratio * len(pool_ids)))
            start = i % self.number_of_models
            # Pure width term (over the full pool) passed only so selection logging can show the
            # score before vs after the residual blend; None when projection is off.
            width_norm = self._normalized_widths(all_preds) if self.use_monotone_projection else None
            selected_per_model, remaining_suspension_ids, added = self._assign_units_round_robin(
                all_preds=all_preds,
                norm_width=norm_width,
                n_add=n_add,
                start=start,
                remaining_suspension_ids=remaining_suspension_ids,
                width_norm=width_norm,
            )

            if added == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: "
                             f"no censored unit available for any model.")
                break

            for j in range(self.number_of_models):
                if selected_per_model[j]:
                    xj, yj = models_datasets[j]

                    new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)

                    # Build the candidate (accumulated + newly assigned units) without committing
                    # it yet, so keep-best-model can reject it below.
                    candidate_x = torch.cat([xj, new_xu], dim=0)
                    candidate_y = torch.cat([yj, new_lu], dim=0)
                    n_added = len(selected_per_model[j])

                    if self.use_fine_tuning:
                        self._log(1, f"[CoTraining]   Fine-tuning model {j} (warm start) | "
                                     f"added {n_added} unit(s) | "
                                     f"dataset size: {len(candidate_x)} samples")
                        candidate = self._fine_tune(
                            j, self._cpu_state_dict(h[j]), candidate_x, candidate_y,
                            self._cpu_pair(val_data, val_label))
                    else:
                        self._log(1, f"[CoTraining]   Retraining model {j} from scratch | "
                                     f"added {n_added} unit(s) | "
                                     f"dataset size: {len(candidate_x)} samples")
                        candidate = self._fit_from_scratch(
                            j, candidate_x, candidate_y, val_data, val_label)

                    if not self.keep_best_model:
                        # Legacy behavior: always accept the candidate.
                        h[j] = candidate
                        models_datasets[j] = (candidate_x, candidate_y)
                        if track_val_rmse:
                            val_rmses[j] = self._mse_on(candidate, val_data, val_label) ** 0.5
                    else:
                        candidate_rmse = self._mse_on(candidate, val_data, val_label) ** 0.5
                        if candidate_rmse < val_rmses[j]:
                            self._log(1, f"[CoTraining]   Model {j}: kept (val_rmse "
                                         f"{candidate_rmse:.4f} < best {val_rmses[j]:.4f}).")
                            h[j] = candidate
                            models_datasets[j] = (candidate_x, candidate_y)
                            val_rmses[j] = candidate_rmse
                        else:
                            # Reject: keep the previous model, dataset and best RMSE untouched.
                            # The added units were already removed from remaining_suspension_ids
                            # during assignment, so a rejection discards them for good (matches v3).
                            self._log(1, f"[CoTraining]   Model {j}: iteration {i + 1} rejected "
                                         f"(val_rmse {candidate_rmse:.4f} >= best {val_rmses[j]:.4f}); "
                                         f"reverted and dropped {n_added} censored sample(s).")

            if metrics_enabled:
                self._log_stage_metrics(
                    stage=f"iteration_{i + 1}",
                    h=h,
                    models_datasets=models_datasets,
                    test_data=test_data,
                    test_label=test_label,
                    val_data=val_data,
                    val_label=val_label,
                    score_callback=score_callback,
                    weight_callback=weight_callback,
                    weight_mode=weight_mode,
                    metrics_file=metrics_file,
                )

            # Drop this iteration's scoring structures (each holds full per-unit tensors)
            # before the next iteration allocates its own, so peak RAM does not accumulate
            # across the 20 iterations.
            del all_preds, norm_width, selected_per_model
            gc.collect()

        if metrics_enabled:
            self._log_stage_metrics(
                stage="final",
                h=h,
                models_datasets=models_datasets,
                test_data=test_data,
                test_label=test_label,
                val_data=val_data,
                val_label=val_label,
                score_callback=score_callback,
                weight_callback=weight_callback,
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
            width_norm: dict[int, dict[int, float]] | None = None,
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
                width_norm=width_norm,
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

    @staticmethod
    def _bootstrap_sample(
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a bootstrap resample of ``(x, y)``: ``N`` draws with replacement from ``N`` rows.

        Args:
            x: Features, shape ``(N, ...)``.
            y: Labels aligned with ``x``, shape ``(N, ...)``.

        Returns:
            ``(x_resampled, y_resampled)``, each with the same length ``N`` as the input.
        """
        idx = torch.randint(0, len(x), (len(x),))
        return x[idx], y[idx]

    @staticmethod
    def _time_gap_sample_weight(time_steps: torch.Tensor) -> np.ndarray:
        """Per-window isotonic ``sample_weight`` proportional to the local time gap ``Delta t``.

        Uses the *central* local gap: for an interior window ``i`` the weight is
        ``(t_{i+1} - t_{i-1}) / 2``; the two endpoints use their one-sided gap. A larger gap means
        the window is more temporally isolated, so it is treated as more independent (weighted up)
        in the pooled-adjacent-violators fit. Non-positive gaps (duplicate/unordered time steps)
        are floored to a small positive value so every window keeps a strictly positive weight.

        Args:
            time_steps: Per-window time steps for a single unit, shape ``(m,)`` or ``(m, 1)``,
                ordered oldest -> newest.

        Returns:
            A ``float64`` array of length ``m`` of strictly positive weights.
        """
        t = time_steps.detach().cpu().reshape(-1).numpy().astype(np.float64)
        m = t.shape[0]
        if m == 1:
            return np.ones(1, dtype=np.float64)
        w = np.empty(m, dtype=np.float64)
        w[0] = t[1] - t[0]
        w[-1] = t[-1] - t[-2]
        if m > 2:
            w[1:-1] = (t[2:] - t[:-2]) / 2.0
        # Guard against non-positive gaps (duplicate/unordered time steps) with a small floor
        # relative to the mean absolute gap so no window collapses to zero weight.
        positive = w[w > 0]
        floor = (positive.mean() * 1e-6) if positive.size > 0 else 1e-6
        return np.maximum(w, floor)

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

    @staticmethod
    def _cpu_state_dict(module: LightningModule) -> dict[str, torch.Tensor]:
        """Detach-and-clone a module's ``state_dict`` to CPU (a picklable warm-start snapshot)."""
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _fine_tune(
            self,
            model_index: int,
            current_state_dict: dict[str, torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> LightningModule:
        """Warm-start fine-tune one model from ``current_state_dict`` on ``(x, y)``.

        Loads the current (trained) weights, scales the learning rate by
        ``self.fine_tune_lr_factor`` and trains all parameters for up to
        ``self.fine_tune_max_epochs`` with ``EarlyStopping`` patience
        ``self.fine_tune_patience`` (monitoring ``val_loss`` on ``val_cpu``). Runs inline on the
        resolved single device. Requires the builder path.

        Args:
            model_index: Index of the model being fine-tuned.
            current_state_dict: CPU ``state_dict`` warm start (the model's current weights).
            x: Training features (full accumulated dataset for this model).
            y: Training targets aligned with ``x``.
            val_cpu: ``(val_x, val_y)`` on CPU for early stopping / best-checkpoint selection.

        Returns:
            A fresh ``LightningModule`` rebuilt from the fine-tuned CPU state dict.
        """
        if not self._use_builders:
            raise NotImplementedError(
                "CoTrainingEnsemble_v2 fine-tuning requires setup_training_builder (builder path).")
        spec = FineTuneSpec(
            module_builder=self.module_builders[model_index],
            current_state_dict=current_state_dict,
            lr_factor=self.fine_tune_lr_factor,
            trainable_param_names=None,
            max_epochs=self.fine_tune_max_epochs,
            patience=self.fine_tune_patience,
            batch_size=self.batchs_size[model_index],
            shuffle=self.shuffle_dataloaders[model_index],
            train_x=x.detach().cpu(),
            train_y=y.detach().cpu(),
            val_x=val_cpu[0],
            val_y=val_cpu[1],
            return_state=True,
            accelerator=self._inline_accelerator,
            devices=self._inline_devices,
        )
        result = run_finetune_job(spec)
        return self._rebuild_module(model_index, result["state_dict"])

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
            calib_data: torch.Tensor,
            calib_label: torch.Tensor,
            suspension_lower_bounds: torch.Tensor | None,
            metrics_enabled: bool,
            test_data: torch.Tensor | None,
            test_label: torch.Tensor | None,
            score_callback: Callable[[torch.Tensor, torch.Tensor], float] | None,
            weight_callback: Callable[[torch.Tensor, torch.Tensor], float] | None,
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

        ``val_data``/``val_label`` are used only for early-stopping (``_make_fit_spec``);
        ``calib_data``/``calib_label`` (already resolved by the caller to fall back to
        ``val_data``/``val_label`` when no dedicated calibration set was given) are used only
        for the conformal calibration (``ConformalScoreSpec``) — the two are kept as separate
        CPU pairs so calibration is decoupled from the early-stopping model-selection data.
        """
        n = self.number_of_models
        val_cpu = self._cpu_pair(val_data, val_label)
        calib_cpu = self._cpu_pair(calib_data, calib_label)
        # Whether the censoring clip is active this run (projection on AND bounds available).
        clip_bounds = self.use_monotone_projection and suspension_lower_bounds is not None

        pool = CoTrainingGpuPool(self.gpu_ids)
        pool.start()
        try:
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]] = [
                self._bootstrap_sample(failure_data, failure_label) if self.bagging_failure_data
                else (failure_data, failure_label)
                for _ in range(n)
            ]

            # --- Initial training: one from-scratch job per model, round-robin. ---
            self._log(1, f"[CoTraining] Initial parallel training of {n} models...")
            job_ids = {
                j: pool.submit_job(
                    pool.round_robin_gpu(j),
                    self._make_fit_spec(j, *models_datasets[j], val_cpu))
                for j in range(n)
            }
            results = pool.gather(list(job_ids.values()))
            h = [self._rebuild_module(j, results[job_ids[j]]["state_dict"]) for j in range(n)]
            self._log(1, f"[CoTraining] Initial training done.")

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="initial", h=h, models_datasets=models_datasets,
                    test_data=test_data, test_label=test_label, val_data=val_data,
                    val_label=val_label, score_callback=score_callback,
                    weight_callback=weight_callback,
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
                # xu (and its lower bounds) are identical across models for a given unit, so
                # build the per-unit sequences once and reuse them for every model's job.
                unit_x = [suspension_data[suspension_ids == uid].detach().cpu() for uid in pool_ids]
                unit_lb = (
                    [suspension_lower_bounds[suspension_ids == uid].detach().cpu() for uid in pool_ids]
                    if clip_bounds else None
                )

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
                        val_x=calib_cpu[0],
                        val_y=calib_cpu[1],
                        unit_ids=unit_ids_int,
                        unit_x=unit_x,
                        confidence=self.confidence,
                        use_monotone_projection=self.use_monotone_projection,
                        unit_lower_bounds=unit_lb,
                        accelerator="gpu",
                        devices=1,
                    )
                    score_jobs[j] = pool.submit_conformal(pool.round_robin_gpu(j), spec)

                results = pool.gather(list(score_jobs.values()))

                # Rebuild all_preds[j] from each worker's per-unit
                # (unit_id, label, lower, upper, width, residual, raw_label).
                all_preds: dict[int, OrderedDict] = {}
                xu_by_unit = {uid: xu for uid, xu in zip(unit_ids_int, unit_x)}
                for j in range(n):
                    units = results[score_jobs[j]]["units"]
                    candidates = []
                    for unit_id_int, lu_p, lower, upper, width, residual, raw_lu_p in units:
                        uid_tensor = torch.tensor(unit_id_int)
                        candidates.append(
                            (uid_tensor, xu_by_unit[unit_id_int], lu_p, lower, upper, width, residual, raw_lu_p))
                    candidates.sort(key=lambda e: e[5])
                    all_preds[j] = OrderedDict(
                        (uid.item(), (uid, xu, lu_p, lower, upper, width, residual, raw_lu_p))
                        for uid, xu, lu_p, lower, upper, width, residual, raw_lu_p in candidates
                    )
                    if self.verbose >= 2:
                        ranking = [(uid.item(), round(w, 4)) for uid, _, _, _, _, w, _, _ in candidates]
                        self._log(2, f"[CoTraining]   Model {j} candidate ranking "
                                     f"(most confident first): {ranking}")

                norm_width = self._selection_scores(all_preds)

                # Log each model's full unit ranking (most confident first) so it can be checked
                # whether the models agree on which censored units are confident or not.
                self._log_confidence_ranking(all_preds, norm_width)

                # --- Phase 2: selection (main process). ---
                n_add = max(1, round(add_ratio * len(pool_ids)))
                start = i % n
                # Pure width term (full pool) for selection logging only; None when projection off.
                width_norm = self._normalized_widths(all_preds) if self.use_monotone_projection else None
                selected_per_model, remaining_suspension_ids, added = self._assign_units_round_robin(
                    all_preds=all_preds,
                    norm_width=norm_width,
                    n_add=n_add,
                    start=start,
                    remaining_suspension_ids=remaining_suspension_ids,
                    width_norm=width_norm,
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
                        val_label=val_label, score_callback=score_callback,
                        weight_callback=weight_callback,
                        weight_mode=weight_mode, metrics_file=metrics_file,
                    )

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="final", h=h, models_datasets=models_datasets,
                    test_data=test_data, test_label=test_label, val_data=val_data,
                    val_label=val_label, score_callback=score_callback,
                    weight_callback=weight_callback,
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
            score_callback: Callable[[torch.Tensor, torch.Tensor], float],
            weight_callback: Callable[[torch.Tensor, torch.Tensor], float],
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

        The reported test score (per model, averaged and weighted) comes from
        ``score_callback`` (e.g. the Scania score), while the ensemble weights are derived
        from ``weight_callback`` (e.g. RMSE) — the two are intentionally decoupled so the
        score columns are not just the RMSE used for weighting.

        Args:
            stage: label for the row ("initial", "iteration_<k>" or "final").
            h: the current best model per index.
            models_datasets: per-model ``(x, y)`` accumulated training split.
            test_data, test_label: test set used only for the metrics.
            val_data, val_label: validation set (used for val RMSE and the weights).
            score_callback: score used for the per-model / averaged / weighted test score.
            weight_callback: score used to compute the reported ensemble weights.
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
            test_scores.append(score_callback(pred_j, test_label_flat))

        n = len(h)
        avg_test_rmse = sum(test_rmses) / n
        avg_test_score = sum(test_scores) / n

        # Weights come from the validation set (no test leakage) and do NOT mutate
        # self.weights — they exist only to report the weighted-ensemble metrics.
        # Pass ``h`` explicitly: self.lightning_modules still holds the untrained template
        # modules at this point (they are only replaced by ``h`` when train() finishes),
        # so weighting against it would give weights unrelated to the trained models —
        # and would not match the caller's post-train ``calculate_weights``.
        weights = self._compute_weights(val_data, val_label, weight_callback, weight_mode, models=h)
        weighted_pred = torch.stack(
            [w * pred for w, pred in zip(weights, test_preds)], dim=0
        ).sum(dim=0).view(-1)
        weighted_test_rmse = (((test_label_flat - weighted_pred) ** 2).mean().item()) ** 0.5
        weighted_test_score = score_callback(weighted_pred, test_label_flat)

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

        # Bound the transient (n_query x n_train) distance buffer sklearn allocates inside the
        # kNN DifficultyEstimator (used by crepes for calibrate / predict_int). The default
        # working_memory (1024 MB) can spike host RAM well past a small budget; 128 MB only
        # changes the internal chunk size, not the numerical result.
        sklearn.set_config(working_memory=128)

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
        device = next(model.parameters()).device

        with torch.no_grad():
            if self._inference_batch_size is None:
                return model(x.to(device))

            # Chunk the forward pass so peak activation memory is O(batch) rather than
            # O(len(x)). Each chunk's output is moved back to the input's device before
            # concatenation so the result matches the single-shot path exactly.
            outputs = []
            for start in range(0, len(x), self._inference_batch_size):
                chunk = x[start:start + self._inference_batch_size].to(device)
                outputs.append(model(chunk).to(x.device))
            return torch.cat(outputs, dim=0)

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
                ``{unit_id_int: (unit_id, xu, lu_p, lower, upper, width, residual, raw_lu_p)}`` (index 5 = raw width).

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

    def _log_confidence_ranking(
            self,
            all_preds: dict[int, OrderedDict],
            selection_scores: dict[int, dict[int, float]],
    ) -> None:
        r"""Log, one line per model, every pooled censored unit sorted from most to least confident.

        This makes it easy to eyeball whether the models agree on which units are confident
        (they rank the same unit ids first) or disagree. The score shown is the per-model
        selection score (:meth:`_selection_scores`): the median-normalized conformal interval
        width — blended with the median-normalized projection residual when monotone projection
        is on — so scores are on a comparable, model-relative scale. Smaller = more confident, so
        units are listed in ascending score order (most confident first).

        Format per model::

            [CoTraining]   Model j confidence ranking (most confident first): [uid: score, uid: score, ...]

        Args:
            all_preds: mapping from model index j to its OrderedDict of scored units (only its
                keys are used, to know which units the model scored this iteration).
            selection_scores: mapping from model index j to ``{unit_id_int: selection_score}``
                (the output of :meth:`_selection_scores`).
        """
        self._log(1, "[CoTraining]   Confidence ranking per model (unit_id: selection_score, "
                     "smaller = more confident):")
        for j in range(self.number_of_models):
            scores_j = selection_scores.get(j, {})
            ranked = sorted(scores_j.items(), key=lambda kv: kv[1])
            formatted = ", ".join(f"{uid}: {score:.4f}" for uid, score in ranked)
            self._log(1, f"[CoTraining]   Model {j} confidence ranking (most confident first): "
                         f"[{formatted}]")

    def _selection_scores(
            self,
            all_preds: dict[int, OrderedDict],
    ) -> dict[int, dict[int, float]]:
        r"""Per-model, per-unit score used to rank censored units (smaller = better).

        The base score is the median-normalized conformal interval width
        (:meth:`_normalized_widths`). When monotone projection is enabled *and*
        ``monotone_residual_weight`` is non-zero, the median-normalized projection residual is
        blended in:

            ``score = width_norm + monotone_residual_weight * residual_norm``

        so a unit is preferred only when a model is both *narrow* (a confident interval) and
        *physically self-consistent* (a small residual — its raw predictions barely had to move
        to become monotone / above the survival bound). With projection off (or the weight 0)
        this returns exactly :meth:`_normalized_widths` — the legacy width-only behavior.

        Each model's residuals are normalized by that model's own **mean** (not the median used
        for widths). Residuals are a mostly-zero-with-spikes quantity: censored pools often have
        >= half their units at residual 0 (single-window or already-valid units), so a median
        would be 0 and — with the zero fallback below — the residual term would be silently
        dropped exactly when some units *do* violate. The mean is > 0 whenever *any* unit has a
        violation, so the residual signal is applied consistently. The fallback when the mean is
        ``<= 0`` (all residuals zero) is **0.0**, not the ``1.0`` used for widths: with no
        violations the residual term must contribute nothing rather than a phantom typical value.

        Args:
            all_preds: mapping from model index j to an OrderedDict of
                ``{unit_id_int: (uid, xu, lu_p, lower, upper, width, residual)}``.

        Returns:
            ``{model_index: {unit_id_int: score}}``.
        """
        width_norm = self._normalized_widths(all_preds)
        if not self.use_monotone_projection or self.monotone_residual_weight == 0:
            return width_norm

        scores: dict[int, dict[int, float]] = {}
        for j, preds in all_preds.items():
            scores[j] = {}
            if not preds:
                continue
            residual_values = [tup[6] for tup in preds.values()]
            mean = float(np.mean(residual_values))
            for uid, tup in preds.items():
                res_norm = tup[6] / mean if mean > 0 else 0.0
                scores[j][uid] = width_norm[j][uid] + self.monotone_residual_weight * res_norm
        return scores

    def _voting_censored_data_selection(
            self,
            all_preds: dict[int, OrderedDict],
            norm_width: dict[int, dict[int, float]],
            model_index_to_exclude: int,
            width_norm: dict[int, dict[int, float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        r"""Select the censored unit the *other* models are most confident about.

        For each candidate unit u, the average **normalized** interval width (see
        ``_normalized_widths``) is computed over all models j ≠ k. Normalizing first matters
        because each model calibrates its own conformal regressor independently, so raw widths
        aren't on a comparable scale — a systematically noisier model would otherwise dominate
        the average regardless of which unit is actually easiest for it. The unit with the
        *smallest* average normalized width (i.e. the tightest, most confident consensus among
        the other models) is selected.

        The pseudo-label assigned to model k comes from its peers j ≠ k (model k's own
        predictions are ignored so it genuinely learns from its peers — the co-training
        principle). By default it is the single peer with the smallest *normalized* interval
        width for that unit (the most confident peer). When ``self.peer_weighted_pseudo_label``
        is on, it is instead the ``1 / confidence_score**2``-weighted average of *all* peers'
        per-window predictions, where the confidence score is each peer's own per-unit
        ``norm_width`` (a tighter, more confident peer contributes more).

        Args:
            all_preds: mapping from model index j to an OrderedDict of
                ``{unit_id_int: (unit_id_tensor, xu, lu_p, lower, upper, width, residual, raw_lu_p)}``.
            norm_width: mapping from model index j to ``{unit_id_int: score}`` (the blended
                selection score from ``_selection_scores`` — width-only when projection is off).
                Also used, when ``self.peer_weighted_pseudo_label`` is on, as each peer's
                confidence score to weight the blended pseudo-label by
                ``1 / (score**2 + eps)``.
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

        # The peers j ≠ k that scored this unit contribute the pseudo-label.
        contributors = [
            j for j in all_preds
            if j != model_index_to_exclude and best_unit_id_int in all_preds[j]
        ]
        most_confident_j = min(contributors, key=lambda j: norm_width[j][best_unit_id_int])

        unit_id, xu, lu_p, _, _, _, _, _ = all_preds[most_confident_j][best_unit_id_int]

        if self.peer_weighted_pseudo_label:
            # Blend all peers' per-window predictions, weighting each by
            # 1 / (norm_width**2 + eps) so a tighter, more confident peer contributes more.
            # This is the same per-unit confidence score already used to pick most_confident_j
            # above. All peers scored the same unit windows (same xu), so their label tensors
            # share a shape; a convex combination of non-increasing sequences stays
            # non-increasing (monotone projection is preserved).
            eps = 1e-8
            weighted_sum = None
            weight_total = 0.0
            for j in contributors:
                score_j = norm_width[j][best_unit_id_int]
                w_j = 1.0 / (score_j ** 2 + eps)
                lu_j = all_preds[j][best_unit_id_int][2]
                weighted_sum = lu_j * w_j if weighted_sum is None else weighted_sum + lu_j * w_j
                weight_total += w_j
            lu_p = weighted_sum / weight_total

        self._log(1, f"[CoTraining]     Unit {best_unit_id_int} selected for model {model_index_to_exclude} | "
                     f"best avg normalized width = {best_avg_width:.4f}")

        num_sequences = lu_p.view(-1).shape[0]

        if self.use_monotone_projection:
            # Effectiveness tracking for the monotone-projection score. ``norm_width`` here is the
            # *blended* score (width_norm + lambda*residual_norm); ``width_norm`` is the pure width
            # term, computed once over the full pool and passed in (recomputing here would use the
            # already-popped ``all_preds`` and shift the medians, so "before"/"after" must not be
            # derived from the shrunk map).
            width_before = width_norm if width_norm is not None else self._normalized_widths(all_preds)
            self._log(1, f"[CoTraining]     Unit {best_unit_id_int} normalized width before -> after "
                         f"residual blend (residual) per peer model:")
            for j in contributors:
                self._log(1, f"[CoTraining]     \tmodel {j}: {width_before[j][best_unit_id_int]:.4f} -> "
                             f"{norm_width[j][best_unit_id_int]:.4f} "
                             f"(residual {all_preds[j][best_unit_id_int][6]:.4f})")

            self._log(1, f"[CoTraining]     Unit {best_unit_id_int} RUL before / after monotone projection "
                         f"per peer model ({num_sequences} sequences for this unit):")
            for j in contributors:
                raw_list = [round(v, 4) for v in all_preds[j][best_unit_id_int][7].view(-1).tolist()]
                proj_list = [round(v, 4) for v in all_preds[j][best_unit_id_int][2].view(-1).tolist()]
                self._log(1, f"[CoTraining]     \tmodel {j} before: {raw_list}")
                self._log(1, f"[CoTraining]     \tmodel {j} after : {proj_list}")
        else:
            self._log(1, f"[CoTraining]     Unit {best_unit_id_int} width per peer model (raw / normalized):")
            for j in contributors:
                self._log(1, f"[CoTraining]     \tmodel {j}: {all_preds[j][best_unit_id_int][5]:.4f} / "
                             f"{norm_width[j][best_unit_id_int]:.4f}")

            self._log(1, f"[CoTraining]     Unit {best_unit_id_int} RUL predicted by all peer models "
                         f"({num_sequences} sequences for this unit):")
            for j in contributors:
                self._log(1, f"[CoTraining]     \tmodel {j}: "
                             f"{[round(v, 4) for v in all_preds[j][best_unit_id_int][2].view(-1).tolist()]}")

        label_source = (
            f"1/confidence_score**2-weighted average over peers {contributors}"
            if self.peer_weighted_pseudo_label
            else f"most confident peer, model {most_confident_j}"
        )
        self._log(1, f"[CoTraining]     Unit {best_unit_id_int} chosen RUL ({num_sequences} sequences for this "
                     f"unit, from {label_source}): "
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
