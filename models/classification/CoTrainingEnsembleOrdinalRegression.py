"""Co-training ensemble for the Scania Component X **ordinal classification** problem.

This is the classification counterpart of :class:`models.CoTrainingEnsemble_v2`. The skeleton is
the same — a committee of architectures, each trained on the labelled (uncensored) vehicles,
iteratively self-labelling censored vehicles for one another — but everything that depends on
the target type is different:

* Each model is an ordinal logistic classifier (see
  :class:`~scania.lightning_module.OrdinalLightningModule`), so a forward pass yields a
  ``(N, num_classes)`` **probability matrix** instead of a scalar RUL.
* Confidence comes from **conformal classification** (``crepes``) rather than conformal
  regression: per censored window we compute a p-value per class and a prediction set at
  ``confidence``. v2's "narrow interval = confident" becomes "singleton set with a high
  p-value = confident".
* A censored vehicle is only usable if its predicted class sequence is *physically possible*.
  Class index is a non-increasing step function of the remaining time, so along a vehicle's
  time-ordered windows the label must be **non-decreasing**, and it can never exceed the
  ``class_upper_bound`` implied by how long the vehicle was actually observed.
* Selection is by **agreement** rather than v2's round-robin: the models that agree on a
  vehicle's whole label sequence *own* it and hand their labels to every model that disagreed
  or had no usable prediction.

The algorithm is specified in ``algorithm_architecture/CoTrainingEnsembleClassification.md``.
"""

import csv
import gc
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from crepes import ConformalClassifier
from crepes.extras import hinge
from lightning import LightningModule

from models.classification import ordinal_metrics
from models.coprog_gpu_pool import run_training_job, TrainingSpec
from models.cotraining_gpu_pool import FineTuneSpec, run_finetune_job


class ComputingWeightMode(Enum):
    """Metric used to weight the models in the ensemble prediction."""

    VAL_RMSE = "val_rmse"
    VAL_ACCURACY = "val_accuracy"
    CONFIDENCE = "confidence"
    F1_SCORE_MACRO = "f1_score_macro"
    SCANIA_COST = "scania_cost"


class ImprovingMetricMode(Enum):
    """Metric used to decide whether a newly trained model replaces the previous one."""

    MSE_ON_ORIGINAL_DATASET = "mse_on_original_dataset"
    VAL_RMSE = "val_rmse"
    VAL_ACCURACY = "val_accuracy"
    CONFIDENCE = "confidence"
    F1_SCORE_MACRO = "f1_score_macro"
    SCANIA_COST = "scania_cost"


# Enum member -> (metric name in ``ordinal_metrics``, direction). ``CONFIDENCE`` is not in
# ``ordinal_metrics`` because it scores a probability matrix rather than class indices, and
# ``MSE_ON_ORIGINAL_DATASET`` is scored on a different dataset; both are special-cased.
_MODE_TO_METRIC: dict[str, str] = {
    "val_rmse": "rmse",
    "val_accuracy": "accuracy",
    "f1_score_macro": "f1_macro",
    "scania_cost": "scania_cost",
}

# Mean top predicted probability — the model's own (unconformalized) certainty. Bigger is better.
_CONFIDENCE_METRIC = "confidence"

# Mean squared error on the class index, evaluated on a model's *original* (pre-pseudo-label)
# training split. Bigger is worse.
_ORIGINAL_MSE_METRIC = "mse_on_original_dataset"

_SPECIAL_METRIC_DIRECTION: dict[str, str] = {
    _CONFIDENCE_METRIC: "max",
    _ORIGINAL_MSE_METRIC: "min",
}

# Metrics written to the per-stage CSV, for every split (``val``/``test``), for every model as
# well as for the arithmetic-mean and weighted-ensemble summaries. Keys of
# ``ordinal_metrics.ORDINAL_METRICS``, except ``scania_cost`` which is reported as ``cost``.
_REPORTED_METRICS: tuple[str, ...] = ("accuracy", "f1_macro", "mae", "rmse", "cost")

# CSV metric name -> the ``ordinal_metrics`` callable computing it.
_REPORTED_METRIC_FUNCTIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "accuracy": ordinal_metrics.accuracy,
    "f1_macro": ordinal_metrics.f1_macro,
    "mae": ordinal_metrics.mae,
    "rmse": ordinal_metrics.rmse,
    "cost": ordinal_metrics.scania_cost,
}


# Rejection reasons counted by ``_filter_unit``, in the order the rules are applied, and also
# the order they are reported in the per-model filter summary.
_FILTER_AMBIGUOUS = "ambiguous"
_FILTER_UNCERTAIN = "uncertain"
_FILTER_DECREASING = "decreasing"
_FILTER_IMPOSSIBLE = "impossible"

_FILTER_REASONS: tuple[str, ...] = (
    _FILTER_AMBIGUOUS,
    _FILTER_UNCERTAIN,
    _FILTER_DECREASING,
    _FILTER_IMPOSSIBLE,
)


@dataclass
class _UnitPrediction:
    """One model's usable prediction for one censored vehicle.

    Only produced for vehicles that survived every filter in
    :meth:`CoTrainingEnsembleOrdinalRegression._filter_unit`.

    Attributes:
        unit_id: The vehicle id, as the scalar tensor taken from ``suspension_ids``.
        x: The vehicle's censored windows, shape ``(n_windows, seq_len, feature_num)``,
            in time order.
        labels: Predicted class index per window, shape ``(n_windows,)``, non-decreasing.
        confidence: Mean, over the vehicle's windows, of the p-value of the predicted class.
    """

    unit_id: torch.Tensor
    x: torch.Tensor
    labels: np.ndarray
    confidence: float


@dataclass
class _UnitAssignment:
    """One censored vehicle that reached a strict majority and may be consumed this iteration.

    Attributes:
        score: Mean confidence of the owning models, the ranking key.
        unit_id: The vehicle id.
        owners: Indices of the models forming the strict majority; they already predict
            ``prediction.labels`` and so learn nothing from it.
        receivers: Indices of every other model — the ones handed ``prediction``.
        prediction: The owners' shared prediction, used as the pseudo-label.
    """

    score: float
    unit_id: int
    owners: list[int]
    receivers: list[int]
    prediction: _UnitPrediction


class CoTrainingEnsembleOrdinalRegression:
    """Committee of ordinal classifiers that self-label censored vehicles for one another."""

    def __init__(
            self,
            models: list[nn.Module],
            weights: list[float] | None = None,
            verbose: int = 0,
            num_classes: int = ordinal_metrics.NUM_CLASSES,
            bagging_failure_data: bool = False,
            confidence: float = 0.99,
            p_value_treshold: float = 0.70,
            inference_batch_size: int | None = None,
            use_fine_tuning: bool = False,
            fine_tune_lr_factor: float = 0.1,
            fine_tune_max_epochs: int = 20,
            fine_tune_patience: int = 5,
            keep_best_model_mode: ImprovingMetricMode | None = None,
            computing_weight_mode: ComputingWeightMode = ComputingWeightMode.F1_SCORE_MACRO,
    ):
        """
        Args:
            models: One ``nn.Module`` per committee member. Only their count and identity are
                used here; the modules actually trained are rebuilt from ``module_builders``.
            weights: Optional pre-set ensemble weights (one per model). Normally left ``None``
                and computed by :meth:`calculate_weights` after training.
            verbose: 0 = silent, 1 = per-iteration progress plus the per-model filter and
                assignment summaries, 2 = also per-model scoring traces.
            num_classes: Number of ordered failure-urgency classes.
            bagging_failure_data: When ``True``, each model's initial training set is a
                bootstrap resample of the labelled data, adding diversity to the committee.
            confidence: Confidence level in ``(0, 1)`` for the conformal prediction sets. A
                vehicle is only usable by a model when **every** window's set is a singleton.
            p_value_treshold: Minimum p-value the predicted class must reach on **every**
                window of a vehicle for that model to keep it.
            inference_batch_size: Chunk size for every forward pass, so peak activation memory
                stays ``O(batch)``. ``None`` runs single-shot inference.
            use_fine_tuning: When ``True``, an updated model is obtained by fine-tuning the
                post-initial-training snapshot on its accumulated pseudo-labels instead of
                retraining from scratch on the full augmented split (see :meth:`train`).
            fine_tune_lr_factor: Multiplier applied to the learning rate when fine-tuning.
            fine_tune_max_epochs: Epoch budget for a fine-tune.
            fine_tune_patience: ``EarlyStopping`` patience for a fine-tune.
            keep_best_model_mode: When set, a newly trained model only replaces the previous
                one if it improves this metric; otherwise the previous weights are kept.
                ``None`` always accepts the new model.
            computing_weight_mode: Metric used by :meth:`calculate_weights`.

        Raises:
            ValueError: If ``weights`` does not have one entry per model, or ``confidence`` /
                ``p_value_treshold`` / ``fine_tune_lr_factor`` are out of range.
            TypeError: If ``keep_best_model_mode`` or ``computing_weight_mode`` is not the
                expected enum.
        """
        if weights is not None and len(models) != len(weights):
            raise ValueError("The number of weights must be the same as the number of models.")

        if not (0 < confidence < 1):
            raise ValueError("confidence must be in the range (0, 1).")

        if fine_tune_lr_factor <= 0:
            raise ValueError("fine_tune_lr_factor must be positive.")

        if not (0 < p_value_treshold < 1):
            raise ValueError("p_value_treshold must be in the range (0, 1).")

        if keep_best_model_mode is not None and not isinstance(keep_best_model_mode, ImprovingMetricMode):
            raise TypeError(
                "keep_best_model_mode must be an ImprovingMetricMode or None"
            )

        if not isinstance(computing_weight_mode, ComputingWeightMode):
            raise TypeError(
                "computing_weight_mode must be an ComputingWeightMode"
            )

        self.models = models
        self.number_of_models = len(self.models)
        self.lightning_modules = None
        self.batchs_size = None
        self.shuffle_dataloaders = None
        self.weights = weights
        self.verbose = verbose
        self.num_classes = num_classes
        self.confidence = confidence
        self._inference_batch_size = inference_batch_size
        self.use_fine_tuning = use_fine_tuning
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.fine_tune_max_epochs = fine_tune_max_epochs
        self.fine_tune_patience = fine_tune_patience
        self.keep_best_model_mode = keep_best_model_mode
        self.bagging_failure_data = bagging_failure_data
        self.computing_weight_mode = computing_weight_mode
        self.p_value_treshold = p_value_treshold

        # Class labels in the order crepes expects them (column order of the probability matrix).
        self._classes = np.arange(self.num_classes)

        self.module_builders: list[Callable[[], LightningModule]] | None = None
        self.max_epochs: list[int] | None = None
        self.patiences: list[int] | None = None
        self.gpu_ids: list[int] | None = None
        self._initial_state_dicts: list[dict[str, torch.Tensor]] | None = None
        self._inline_accelerator: str = "auto"
        self._inline_devices = None
        self._configured: bool = False

        # Optional path to a .txt log file (set by ``train``). When not None, every
        # ``_log`` message is appended to it regardless of ``verbose``.
        self._log_file_path: str | None = None

    def setup_training_builder(
            self,
            module_builders: list[Callable[[], LightningModule]],
            max_epochs: list[int],
            patiences: list[int],
            batchs_size: list[int],
            shuffle_dataloaders: list[bool],
            gpu_ids: list[int] | None = None,
    ) -> None:
        r"""Configure builder-style training.

        A fresh module is built from a *picklable* builder, its initial weights are pinned to a
        one-time snapshot, and the ``Trainer`` is built internally by
        :func:`~models.coprog_gpu_pool.run_training_job`. Each list has one entry per model
        (same order as ``models``).

        Unlike v2 this ensemble is **single-GPU only**: conformal scoring here is a handful of
        numpy operations on probability matrices (v2 had to fit a kNN difficulty estimator per
        model per iteration), so there is nothing left worth fanning out across GPUs.

        Args:
            module_builders: Picklable callables (module-level functions or ``functools.partial``
                — no lambdas/closures) each returning a *fresh* ``LightningModule``.
            max_epochs: Max training epochs per model.
            patiences: ``EarlyStopping`` patience per model.
            batchs_size: Batch size used to train each model.
            shuffle_dataloaders: Whether to shuffle each training ``DataLoader``.
            gpu_ids: ``None`` → auto device selection; ``[g]`` → pin training to GPU ``g``.

        Raises:
            ValueError: If a list does not have one entry per model, or more than one GPU id
                is given.
        """
        model_number = len(self.models)
        if (len(module_builders) != model_number or len(max_epochs) != model_number
                or len(patiences) != model_number or len(batchs_size) != model_number
                or len(shuffle_dataloaders) != model_number):
            raise ValueError(
                f"module_builders, max_epochs, patiences, batchs_size and shuffle_dataloaders "
                f"must all have length {model_number}.")

        if gpu_ids is not None and len(gpu_ids) > 1:
            raise ValueError(
                f"CoTrainingEnsembleOrdinalRegression is single-GPU only, got gpu_ids={gpu_ids}. "
                f"Pass a single id (e.g. [0]) or None for automatic device selection.")

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
            self._inline_accelerator = "auto"
            self._inline_devices = None
        else:
            self._inline_accelerator = "gpu"
            self._inline_devices = [self.gpu_ids[0]]

        self._configured = True

    def train(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            suspension_class_upper_bounds: torch.Tensor,
            iterations: int,
            suspension_pool_size: float,
            add_ratio: float,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            calib_data: torch.Tensor,
            calib_label: torch.Tensor,
            test_data: torch.Tensor | None = None,
            test_label: torch.Tensor | None = None,
            metrics_file: str | None = None,
            log_file: str | None = None,
            pretrained_models: list[LightningModule] | None = None,
            pretrained_models_datasets: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
            pool_seed: int | None = None,
    ) -> None:
        """Run the co-training loop and store the trained models in ``self.lightning_modules``.

        Each iteration: draw a pool of censored vehicles, score them with every model's
        conformal classifier, filter out the impossible/uncertain ones, decide per vehicle which
        models *own* it, hand the owners' labels to every other model, and retrain (or
        fine-tune) the models that received something.

        Fine-tuning, when enabled, always warm-starts from the **post-initial-training**
        snapshot and trains on **all** the pseudo-labels a model has accumulated so far — a
        model is never fine-tuned on top of an already fine-tuned model. The previous
        iteration's model is still what scores and selects the censored vehicles.

        Args:
            failure_data: Labelled (uncensored) training windows, ``(N, seq_len, feature_num)``.
            failure_label: Class indices for ``failure_data``, ``(N, 1)``.
            suspension_data: Censored windows, ``(M, seq_len, feature_num)``.
            suspension_ids: Vehicle id per censored window, ``(M,)``.
            suspension_class_upper_bounds: Per-window upper bound on the unknown true class
                (``class_upper_bound``), row-aligned with ``suspension_data``. A predicted class
                above this value is physically impossible and disqualifies the vehicle.
            iterations: Maximum number of co-training iterations.
            suspension_pool_size: Fraction in ``(0, 1]`` of the remaining censored vehicles
                sampled as the candidate pool each iteration.
            add_ratio: Fraction in ``(0, 1]`` of the pool consumed each iteration.
            val_data: Validation windows (early stopping, best-checkpoint selection, weights).
            val_label: Class indices for ``val_data``.
            calib_data: Conformal calibration windows. Should be disjoint from ``val_data``,
                which is also used for model selection.
            calib_label: Class indices for ``calib_data``.
            test_data: Test windows; required to log per-stage metrics.
            test_label: Class indices for ``test_data``.
            metrics_file: Destination CSV for the per-stage metrics. ``None`` disables them.
            log_file: Destination ``.txt`` receiving every log message regardless of ``verbose``.
            pretrained_models: Already-trained models to reuse instead of running the initial
                training, so every config of a sweep starts from an identical baseline.
            pretrained_models_datasets: The per-model ``(x, y)`` splits matching
                ``pretrained_models``. Required when it is given.
            pool_seed: Seed for the dedicated pool RNG, kept separate from the global RNG so
                the pool sequence does not shift with how much randomness training consumes.

        Raises:
            ValueError: If the ensemble was not configured, or a required split is missing.
        """
        self._log_file_path = log_file

        metrics_enabled = metrics_file is not None

        self._check_if_training_is_possible(
            val_data,
            val_label,
            calib_data,
            calib_label,
            metrics_enabled,
            test_data,
            test_label,
        )

        self._log(1, f"[CoTraining] Training parameters | "
                     f"num_classes: {self.num_classes} |"
                     f"confidence: {self.confidence} |"
                     f"p_value_treshold: {self.p_value_treshold} |"
                     f"use_fine_tuning: {self.use_fine_tuning} |"
                     f"fine_tune_lr_factor: {self.fine_tune_lr_factor} |"
                     f"fine_tune_max_epochs: {self.fine_tune_max_epochs} |"
                     f"fine_tune_patience: {self.fine_tune_patience} |"
                     f"keep_best_model_mode: {self.keep_best_model_mode} |"
                     f"bagging_failure_data: {self.bagging_failure_data} |"
                     f"computing_weight_mode: {self.computing_weight_mode} |"
                     f"pool_seed: {pool_seed}")

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
                     f"max iterations: {iterations} | "
                     f"pool fraction: {suspension_pool_size} (size: {pool_size}) | "
                     f"add ratio: {add_ratio} | "
                     f"device: {self.gpu_ids if self.gpu_ids else 'auto'}")

        if pretrained_models is not None:
            # Reusing already-trained models (see the ``pretrained_models`` docstring): every
            # config of a hyperparameter sweep starts from the identical Initial-training result.
            h = list(pretrained_models)
            models_datasets = list(pretrained_models_datasets)
            self._log(1, f"[CoTraining] Initial training skipped: reusing {len(h)} pretrained "
                         f"model(s) (dataset sizes: {[len(x) for x, _ in models_datasets]}).")
        else:
            h, models_datasets = self.pretrain_initial_models(
                failure_data=failure_data,
                failure_label=failure_label,
                val_data=val_data,
                val_label=val_label,
            )

        # Snapshot of every model right after Initial training. A plain list copy is enough:
        # these model objects are never mutated in place afterward (h[j] is only ever reassigned
        # to a new candidate object, never trained further itself). It is the permanent
        # warm-start base for fine-tuning, so no model is ever fine-tuned twice over.
        h_initial: list[LightningModule] = list(h)

        # The pre-pseudo-label split per model, needed by ImprovingMetricMode.MSE_ON_ORIGINAL_DATASET.
        original_datasets = list(models_datasets)

        # Pseudo-labelled rows accumulated per model across all iterations. This is what a
        # fine-tune trains on: since the warm start always resets to h_initial[j], using only
        # the current iteration's units would discard every earlier iteration's pseudo-labels.
        pseudo_datasets: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.number_of_models

        if metrics_enabled:
            self._log_stage_metrics(
                stage="initial",
                h=h,
                models_datasets=models_datasets,
                test_data=test_data,
                test_label=test_label,
                val_data=val_data,
                val_label=val_label,
                metrics_file=metrics_file,
            )

        remaining_suspension_ids = torch.unique(suspension_ids)

        # Dedicated pool RNG (see the ``pool_seed`` docstring): kept separate from the global RNG
        # so the pool sequence does not shift with how much randomness model training consumes.
        pool_generator = torch.Generator().manual_seed(pool_seed) if pool_seed is not None else None

        for i in range(iterations):
            if len(remaining_suspension_ids) == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i}: no remaining censored units.")
                break

            # Draw a random candidate pool from the remaining units. Re-sampling every iteration
            # means the same units are not always scored
            pool_size_iter = min(pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[
                torch.randperm(len(remaining_suspension_ids), generator=pool_generator)]
            pool_ids = shuffled_ids[:pool_size_iter]

            self._log(1, f"[CoTraining] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            # Phase 1 — for each model, calibrate a conformal classifier on the calibration
            # split and score every pooled censored unit. Units that fail any feasibility or
            # confidence filter simply never enter that model's map (and are logged with the
            # reason and the proof).
            all_preds: dict[int, OrderedDict[int, _UnitPrediction]] = {}

            for j in range(self.number_of_models):
                hj = h[j]

                self._log(2, f"[CoTraining]   Model {j}: calibrating conformal classifier and "
                             f"scoring {len(pool_ids)} pooled censored units...")

                conformal = self._build_calibrated_conformal_classifier(hj, calib_data, calib_label)

                candidates: list[_UnitPrediction] = []
                rejections: Counter[str] = Counter()
                for unit_id in pool_ids:
                    mask = (suspension_ids == unit_id)
                    xu = suspension_data[mask]
                    bounds_u = suspension_class_upper_bounds[mask].reshape(-1)

                    probs = self._predict_proba(hj, xu)
                    alphas = hinge(probs)
                    # smoothing=False keeps p-values (and therefore every selection decision)
                    # deterministic; smoothed p-values draw randomness per call.
                    p_values = conformal.predict_p(alphas, smoothing=False)
                    prediction_set = conformal.predict_set(
                        alphas, confidence=self.confidence, smoothing=False)

                    filtered = self._filter_unit(
                        j, int(unit_id), p_values, prediction_set, bounds_u, rejections)
                    if filtered is None:
                        continue
                    labels, confidence = filtered
                    candidates.append(_UnitPrediction(unit_id, xu, labels, confidence))

                self._log_filter_summary(j, len(pool_ids), rejections)

                # Bigger mean p-value = more confident, so sort descending (best candidate first).
                candidates.sort(key=lambda c: c.confidence, reverse=True)
                all_preds[j] = OrderedDict((int(c.unit_id), c) for c in candidates)

                # Release this model's calibrated classifier before the next model builds its
                # own, so at most one is alive at a time.
                del conformal, candidates
                gc.collect()

            self._log_confidence_ranking(all_preds)

            # Phase 2 — decide, per unit, which models own it and which receive it.
            n_add = max(1, round(add_ratio * len(pool_ids)))
            selected_per_model, remaining_suspension_ids, added = self._assign_units_by_agreement(
                all_preds=all_preds,
                n_add=n_add,
                remaining_suspension_ids=remaining_suspension_ids,
            )

            if added == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: "
                             f"no censored unit could be assigned to any model.")
                break

            # Now we need to train from scratch/fine-tune the models with the new data
            for j in range(self.number_of_models):
                if not selected_per_model[j]:
                    self._log(1, f"[CoTraining] no censored unit could be assigned to the model {j}.")
                    continue

                xj, yj = models_datasets[j]
                new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)

                models_datasets[j] = (torch.cat([xj, new_xu], dim=0), torch.cat([yj, new_lu], dim=0))
                pseudo_datasets[j] = self._append_pseudo_rows(pseudo_datasets[j], new_xu, new_lu)

                candidate = self._train_candidate(
                    model_index=j,
                    h_initial_j=h_initial[j],
                    augmented=models_datasets[j],
                    pseudo=pseudo_datasets[j],
                    selected_count=len(selected_per_model[j]),
                    val_data=val_data,
                    val_label=val_label,
                )

                h[j] = self._select_better_model(
                    model_index=j,
                    incumbent=h[j],
                    candidate=candidate,
                    original_dataset=original_datasets[j],
                    val_data=val_data,
                    val_label=val_label,
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
                    metrics_file=metrics_file,
                )

            # Drop this iteration's scoring structures (each holds full per-unit tensors)
            # before the next iteration allocates its own, so peak RAM does not accumulate.
            del all_preds, selected_per_model
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
                metrics_file=metrics_file,
            )

        self._log(1, f"[CoTraining] Training complete.")
        self.lightning_modules = h

# ──────────────────────────────────────────────────────────────────────────────
# Conformal scoring + per-unit filtering
# ──────────────────────────────────────────────────────────────────────────────

    def _build_calibrated_conformal_classifier(
            self,
            model: LightningModule,
            calib_data: torch.Tensor,
            calib_label: torch.Tensor,
    ) -> ConformalClassifier:
        """Calibrate a ``crepes`` conformal classifier on the held-out calibration split.

        Unlike v2's regression path there is no sklearn adapter and no flattening: crepes'
        :class:`~crepes.ConformalClassifier` works directly on non-conformity scores, so the
        model's probability matrix can be fed to :func:`~crepes.extras.hinge` as-is.

        Args:
            model: The trained model to calibrate.
            calib_data: Calibration windows.
            calib_label: True class indices for ``calib_data``.

        Returns:
            A fitted ``ConformalClassifier`` ready for ``predict_p`` / ``predict_set``.
        """
        probs = self._predict_proba(model, calib_data)
        y = calib_label.detach().cpu().reshape(-1).round().numpy().astype(np.int64)
        alphas = hinge(probs, self._classes, y)

        conformal = ConformalClassifier()
        conformal.fit(alphas=alphas)
        return conformal

    def _filter_unit(
            self,
            model_index: int,
            unit_id: int,
            p_values: np.ndarray,
            prediction_set: np.ndarray,
            class_upper_bounds: torch.Tensor,
            rejection_counts: Counter[str],
    ) -> tuple[np.ndarray, float] | None:
        """Decide whether one model may use one censored vehicle, and with which labels.

        Four rules, all of which must hold for **every** window of the vehicle. The first
        violation found short-circuits, and its reason is counted in ``rejection_counts`` —
        the per-unit proofs are kept commented out because they flood the console; the caller
        reports the aggregated counts through :meth:`_log_filter_summary`.

        1. The conformal prediction set at ``confidence`` must be a singleton — otherwise the
           model is formally undecided between several classes.
        2. The predicted class's p-value must reach ``p_value_treshold`` — the set being a
           singleton bounds how wrong the label can be, this bounds how *uncertain* it is.
        3. The label sequence must be non-decreasing in time. Class index grows as the
           remaining time shrinks, so a vehicle can never become *less* urgent.
        4. No label may exceed the window's ``class_upper_bound``. That bound is derived from
           how long the vehicle was actually observed, so a higher class is impossible however
           confident the model is.

        Args:
            model_index: Index of the model being filtered (for the log messages).
            unit_id: The censored vehicle id.
            p_values: ``(n_windows, num_classes)`` conformal p-values, time-ordered.
            prediction_set: ``(n_windows, num_classes)`` binary conformal prediction sets.
            class_upper_bounds: ``(n_windows,)`` upper bound on the true class per window.
            rejection_counts: Mutated in place — the reason of a rejection (one of
                :data:`_FILTER_REASONS`) is incremented by one.

        Returns:
            ``(labels, confidence)`` where ``labels`` is the ``(n_windows,)`` predicted class
            sequence and ``confidence`` its mean p-value, or ``None`` if the vehicle was
            rejected.
        """
        # Only needed by the per-unit proofs below, which stay commented out; see the summary
        # logged by ``_log_filter_summary`` instead.
        # prefix = f"[CoTraining]     Model {model_index} dropped unit {unit_id}"

        set_sizes = prediction_set.sum(axis=1)
        ambiguous = np.flatnonzero(set_sizes != 1)
        if ambiguous.size:
            w = int(ambiguous[0])
            members = np.flatnonzero(prediction_set[w]).tolist()
            # self._log(2, f"{prefix} | reason=ambiguous_prediction_set | proof=window {w} of "
            #              f"{len(set_sizes)} has prediction set {members} at confidence "
            #              f"{self.confidence} (p-values {np.round(p_values[w], 4).tolist()})")
            rejection_counts[_FILTER_AMBIGUOUS] += 1
            return None

        labels = prediction_set.argmax(axis=1)
        # Get all p_values that are selected by prediction_set
        chosen_p = p_values[np.arange(len(labels)), labels]

        uncertain = np.flatnonzero(chosen_p < self.p_value_treshold)
        if uncertain.size:
            w = int(uncertain[0])
            # self._log(2, f"{prefix} | reason=p_value_below_threshold | proof=window {w} of "
            #              f"{len(labels)} predicts class {int(labels[w])} with p-value "
            #              f"{chosen_p[w]:.4f} < {self.p_value_treshold}")
            rejection_counts[_FILTER_UNCERTAIN] += 1
            return None

        decreasing = np.flatnonzero(np.diff(labels) < 0)
        if decreasing.size:
            w = int(decreasing[0])
            # self._log(2, f"{prefix} | reason=non_monotone_labels | proof=class drops from "
            #              f"{int(labels[w])} at window {w} to {int(labels[w + 1])} at window "
            #              f"{w + 1}; urgency can only increase | sequence={labels.tolist()}")
            rejection_counts[_FILTER_DECREASING] += 1
            return None

        bounds = class_upper_bounds.detach().cpu().numpy()
        impossible = np.flatnonzero(labels > bounds)
        if impossible.size:
            w = int(impossible[0])
            # self._log(2, f"{prefix} | reason=class_above_upper_bound | proof=window {w} of "
            #              f"{len(labels)} predicts class {int(labels[w])} but the observed "
            #              f"survival time only allows up to class {int(bounds[w])} | "
            #              f"sequence={labels.tolist()} | bounds={bounds.astype(int).tolist()}")
            rejection_counts[_FILTER_IMPOSSIBLE] += 1
            return None

        return labels, float(chosen_p.mean())

    def _log_filter_summary(
            self,
            model_index: int,
            pool_size: int,
            rejection_counts: Counter[str],
    ) -> None:
        """Log how many pooled vehicles one model lost to the filters, and to which rule.

        Replaces the per-unit rejection logs of :meth:`_filter_unit`, which flood the console
        as soon as the pool holds more than a handful of vehicles.

        Args:
            model_index: Index of the model that was scoring the pool.
            pool_size: Number of censored vehicles the model scored this iteration.
            rejection_counts: Rejection counts per reason, as filled by :meth:`_filter_unit`.
        """
        filtered = sum(rejection_counts.values())
        detail = " | ".join(f"{reason}: {rejection_counts[reason]}" for reason in _FILTER_REASONS)
        self._log(1, f"[CoTraining]   Model {model_index} filter summary | "
                     f"filtered {filtered}/{pool_size} unit(s) | {detail}")

    def _log_confidence_ranking(
            self,
            all_preds: dict[int, OrderedDict[int, _UnitPrediction]],
    ) -> None:
        """Log every model's surviving units, most confident first.

        Makes it visible whether the committee agrees on which censored vehicles are easy, and
        how many each model lost to the filters.

        Args:
            all_preds: Per-model map of usable unit predictions.
        """
        for j, preds in all_preds.items():
            ranking = [(uid, round(pred.confidence, 4)) for uid, pred in preds.items()]
            self._log(1, f"[CoTraining]   Model {j} kept {len(ranking)} unit(s) "
                         f"(most confident first): {ranking}")

    def _assign_units_by_agreement(
            self,
            all_preds: dict[int, OrderedDict[int, _UnitPrediction]],
            n_add: int,
            remaining_suspension_ids: torch.Tensor,
    ) -> tuple[list[list[_UnitPrediction]], torch.Tensor, int]:
        """Assign censored vehicles to the models that should learn from them.

        For each pooled vehicle the surviving models are grouped by the **whole** predicted
        label sequence. The strict-majority group *owns* the vehicle and every other model —
        those that predicted a different sequence and those the filters rejected — *receives*
        the owners' labels. Two cases are dropped instead:

        * the entire committee agrees, so nobody has anything to learn;
        * the largest group is tied with another, so there is no majority to trust.

        Vehicles are then taken in decreasing order of their owners' mean confidence until the
        iteration's budget ``n_add`` is spent.

        Args:
            all_preds: Per-model map of usable unit predictions.
            n_add: Maximum number of vehicles to consume this iteration.
            remaining_suspension_ids: Censored vehicle ids not yet consumed.

        Returns:
            ``(selected_per_model, remaining_suspension_ids, added)`` where
            ``selected_per_model[j]`` holds the predictions handed to model ``j``.
        """
        unit_ids = sorted({uid for preds in all_preds.values() for uid in preds})

        eligible: list[_UnitAssignment] = []
        dropped_unanimous = 0
        dropped_tied_majority = 0

        for unit_id in unit_ids:
            survivors = [j for j in range(self.number_of_models) if unit_id in all_preds[j]]

            # A group is formed with the labels of all windows (all_preds[j][unit_id].labels) and all models that predict this tuple.
            # For example if we have a sequence of len 3 we can have tuple[0, 0, 1] with label (0, 0, 1) respectively for each sequence
            # If we have 3 models with 2 models (0 and 2) that predict tuple[0, 0, 1] and one (model 1) with tuple[0, 2, 3]
            # The group will be :
            # groups = {
            #     tuple(0, 0, 1): [0, 2],
            #     tuple(0, 2, 3): [1],
            # }
            groups: dict[tuple[int, ...], list[int]] = {}
            for j in survivors:
                groups.setdefault(tuple(all_preds[j][unit_id].labels.tolist()), []).append(j)

            if len(groups) == 1 and len(survivors) == self.number_of_models:
                # self._log(2, f"[CoTraining]     Unit {unit_id} dropped | reason=unanimous | "
                #              f"proof=all {self.number_of_models} models predict "
                #              f"{list(groups)[0]}; no model would learn anything")
                dropped_unanimous += 1
                continue

            sizes = sorted((len(members) for members in groups.values()), reverse=True)
            if len(sizes) >= 2 and sizes[0] == sizes[1]:
                # detail = {seq: members for seq, members in groups.items()}
                # self._log(2, f"[CoTraining]     Unit {unit_id} dropped | reason=tied_majority | "
                #              f"proof=groups {detail} have no strict majority")
                dropped_tied_majority += 1
                continue

            # To select who are the owners and who are the receivers,
            # we prefer to select the result where the numbers of owner is the biggest rather than using the confidence score.
            # The reason is that a bad model can be overconfident.
            # A good confidence score doesn't mean that the model is good.
            # It means that the model is more confident with a given result than another.

            # We get the owners group which have the most models
            _, owners = max(groups.items(), key=lambda kv: len(kv[1]))
            receivers = [j for j in range(self.number_of_models) if j not in owners]
            score = float(np.mean([all_preds[j][unit_id].confidence for j in owners]))
            eligible.append(_UnitAssignment(
                score=score,
                unit_id=unit_id,
                owners=owners,
                receivers=receivers,
                prediction=all_preds[owners[0]][unit_id],
            ))

        self._log(1, f"[CoTraining]   Agreement summary | {len(unit_ids)} unit(s) with at least "
                     f"one usable prediction | dropped unanimous: {dropped_unanimous} | "
                     f"dropped tied_majority: {dropped_tied_majority} | "
                     f"eligible: {len(eligible)}")

        eligible.sort(key=lambda e: e.score, reverse=True)

        selected_per_model: list[list[_UnitPrediction]] = [[] for _ in range(self.number_of_models)]
        chosen = eligible[:n_add]
        added = 0
        for assignment in chosen:
            prediction = assignment.prediction
            # self._log(2, f"[CoTraining]     Unit {assignment.unit_id} selected | "
            #              f"owners={assignment.owners} | receivers={assignment.receivers} | "
            #              f"mean owner confidence={assignment.score:.4f} | "
            #              f"labels={prediction.labels.tolist()} "
            #              f"({len(prediction.labels)} sequences)")
            for j in assignment.receivers:
                selected_per_model[j].append(prediction)
            remaining_suspension_ids = remaining_suspension_ids[remaining_suspension_ids != prediction.unit_id]
            added += 1

        self._log(1, f"[CoTraining]   Assigned {added}/{n_add} unit(s) | per-model received: "
                     f"{[len(s) for s in selected_per_model]}")
        self._log_assignment_summary(chosen, selected_per_model)

        return selected_per_model, remaining_suspension_ids, added

    def _log_assignment_summary(
            self,
            chosen: list[_UnitAssignment],
            selected_per_model: list[list[_UnitPrediction]],
    ) -> None:
        """Log, per model, what this iteration's assignment produced.

        Replaces the per-unit selection log, which flooded the console. Three views are
        reported: the class distribution of the pseudo-labelled sequences each model receives,
        how many vehicles each model owns (i.e. it was in the majority group and therefore
        learns nothing from them), and how many vehicles every combination of models owns
        together — a direct read on how much the committee agrees.

        Args:
            chosen: The assignments consumed this iteration.
            selected_per_model: ``selected_per_model[j]`` holds the predictions handed to
                model ``j``.
        """
        owners_per_model = [
            sum(1 for assignment in chosen if j in assignment.owners)
            for j in range(self.number_of_models)
        ]

        for j in range(self.number_of_models):
            class_counts: Counter[int] = Counter()
            for prediction in selected_per_model[j]:
                class_counts.update(int(label) for label in prediction.labels)

            distribution = " | ".join(
                f"class {c}: {class_counts.get(c, 0)} sequences" for c in range(self.num_classes))
            self._log(1, f"[CoTraining]   Model {j} assignment summary | "
                         f"owned {owners_per_model[j]} unit(s) | "
                         f"received {len(selected_per_model[j])} unit(s) | {distribution}")

        # Every combination of at least two models, so a pair that never co-owns a vehicle is
        # visible as a zero rather than silently missing.
        co_ownership: list[str] = []
        for size in range(2, self.number_of_models + 1):
            for combination in combinations(range(self.number_of_models), size):
                shared = sum(
                    1 for assignment in chosen
                    if all(j in assignment.owners for j in combination))
                co_ownership.append(f"models {combination}: {shared}")

        if co_ownership:
            self._log(1, f"[CoTraining]   Units co-owned | {' | '.join(co_ownership)}")

    @staticmethod
    def _concat_selected_units(
            selected: list[_UnitPrediction],
            reference_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack the windows and pseudo-labels of every unit assigned to one model.

        Args:
            selected: The unit predictions handed to that model this iteration.
            reference_y: The model's existing label tensor, used to match dtype and layout.

        Returns:
            ``(features, labels)`` ready to be concatenated onto the model's split.
        """
        features = torch.cat([pred.x for pred in selected], dim=0)
        labels = torch.from_numpy(np.concatenate([pred.labels for pred in selected])).to(
            dtype=reference_y.dtype)
        if reference_y.dim() > 1:
            labels = labels.view(-1, reference_y.shape[1])
        else:
            labels = labels.view(-1)
        return features, labels

    @staticmethod
    def _append_pseudo_rows(
            current: tuple[torch.Tensor, torch.Tensor] | None,
            new_x: torch.Tensor,
            new_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append newly pseudo-labelled rows to a model's accumulated pseudo-label split.

        Args:
            current: The rows accumulated so far, or ``None`` on the first iteration.
            new_x: This iteration's pseudo-labelled windows.
            new_y: This iteration's pseudo-labels.

        Returns:
            The updated ``(features, labels)`` pair.
        """
        if current is None:
            return new_x, new_y
        return torch.cat([current[0], new_x], dim=0), torch.cat([current[1], new_y], dim=0)

# ──────────────────────────────────────────────────────────────────────────────
# Training dispatcher + builder-style spec helpers
# ──────────────────────────────────────────────────────────────────────────────

    def _train_candidate(
            self,
            model_index: int,
            h_initial_j: LightningModule,
            augmented: tuple[torch.Tensor, torch.Tensor],
            pseudo: tuple[torch.Tensor, torch.Tensor],
            selected_count: int,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
    ) -> LightningModule:
        """Produce the updated model for one committee member.

        Args:
            model_index: Which model is being updated.
            h_initial_j: That model's post-initial-training snapshot (the fine-tune warm start).
            augmented: The model's full ``(features, labels)`` split, labelled + pseudo-labelled.
            pseudo: The model's accumulated pseudo-labelled rows only.
            selected_count: How many units it received this iteration (for the log line).
            val_data: Validation features.
            val_label: Validation labels.

        Returns:
            The newly trained (or fine-tuned) model.
        """
        if self.use_fine_tuning:
            self._log(1, f"[CoTraining]   Fine-tuning model {model_index} from its initial "
                         f"snapshot | added {selected_count} unit(s) | "
                         f"pseudo-label rows: {len(pseudo[0])}")
            return self._fine_tune(model_index, h_initial_j, pseudo[0], pseudo[1], val_data, val_label)

        self._log(1, f"[CoTraining]   Retraining model {model_index} from scratch | "
                     f"added {selected_count} unit(s) | dataset size: {len(augmented[0])} samples")
        return self._fit_from_scratch(model_index, augmented[0], augmented[1], val_data, val_label)

    def _fit_from_scratch(
            self,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None,
            val_y: torch.Tensor | None,
    ) -> LightningModule:
        """Train one model from scratch and rebuild it in this process for inference.

        Args:
            model_index: Which model to train.
            x: Training features.
            y: Training labels.
            val_x: Validation features (early stopping / best checkpoint).
            val_y: Validation labels.

        Returns:
            The trained model, holding the best-``val_loss`` checkpoint's weights.
        """
        spec = self._make_fit_spec(model_index, x, y, self._cpu_pair(val_x, val_y))
        result = run_training_job(spec)
        return self._rebuild_module(model_index, result["state_dict"])

    def _fine_tune(
            self,
            model_index: int,
            base_model: LightningModule,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None,
            val_y: torch.Tensor | None,
    ) -> LightningModule:
        """Warm-start from ``base_model`` and fine-tune on ``(x, y)``.

        ``base_model`` is always the post-initial-training snapshot, so repeated iterations
        never stack fine-tune on fine-tune; the growing pseudo-label set is what carries the
        accumulated knowledge forward.

        Args:
            model_index: Which model to fine-tune.
            base_model: The warm-start model whose weights are loaded before training.
            x: Fine-tune features (the accumulated pseudo-labelled rows).
            y: Fine-tune labels.
            val_x: Validation features (early stopping / best checkpoint).
            val_y: Validation labels.

        Returns:
            The fine-tuned model.
        """
        spec = self._make_finetune_spec(model_index, base_model, x, y, self._cpu_pair(val_x, val_y))
        result = run_finetune_job(spec)
        return self._rebuild_module(model_index, result["state_dict"])

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
            accelerator=self._inline_accelerator,
            devices=self._inline_devices,
        )

    def _make_finetune_spec(
            self,
            model_index: int,
            base_model: LightningModule,
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> FineTuneSpec:
        """Build a picklable :class:`FineTuneSpec` warm-started from ``base_model``.

        ``trainable_param_names`` is ``None``: the whole network, cutpoints included, is
        fine-tuned — freezing the predictor would leave only ``num_classes - 1`` scalars free.
        """
        return FineTuneSpec(
            module_builder=self.module_builders[model_index],
            current_state_dict={
                k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()},
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

    def pretrain_initial_models(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
    ) -> tuple[list[LightningModule], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Train every committee member on the labelled data only.

        Args:
            failure_data: Labelled (uncensored) training windows.
            failure_label: Their class indices.
            val_data: Validation features.
            val_label: Validation labels.

        Returns:
            ``(models, datasets)`` — the trained models and the split each was trained on
            (identical for every model unless ``bagging_failure_data`` is set).
        """
        models_datasets = []
        h: list[LightningModule] = []

        for j in range(self.number_of_models):
            if self.bagging_failure_data:
                x_i, y_i = self._bootstrap_sample(failure_data, failure_label)
            else:
                x_i, y_i = failure_data, failure_label

            models_datasets.append((x_i, y_i))

            self._log(1, f"[CoTraining] Initial training of model {j} on {len(x_i)} failure samples...")
            h_j = self._fit_from_scratch(j, x_i, y_i, val_data, val_label)

            h.append(h_j)

        self._log(1, f"[CoTraining] Initial training done.")

        return h, models_datasets

# ──────────────────────────────────────────────────────────────────────────────
# Model scoring / selection
# ──────────────────────────────────────────────────────────────────────────────

    def _metric_for_mode(self, mode: ComputingWeightMode | ImprovingMetricMode) -> tuple[str, str]:
        """Resolve an enum member into ``(metric name, direction)``.

        Args:
            mode: A ``ComputingWeightMode`` or ``ImprovingMetricMode`` member.

        Returns:
            The metric name understood by :meth:`_score_model` and ``"min"``/``"max"``.
        """
        if mode.value in _MODE_TO_METRIC:
            name = _MODE_TO_METRIC[mode.value]
            return name, ordinal_metrics.METRIC_DIRECTION[name]
        return mode.value, _SPECIAL_METRIC_DIRECTION[mode.value]

    def _score_model(
            self,
            model: LightningModule,
            x: torch.Tensor,
            y: torch.Tensor,
            metric: str,
    ) -> float:
        """Score one model on ``(x, y)`` with the named metric.

        Args:
            model: The model to score.
            x: Features.
            y: True class indices.
            metric: A key of ``ordinal_metrics.ORDINAL_METRICS``, ``"confidence"`` (mean top
                predicted probability, ignores ``y``) or ``"mse_on_original_dataset"``
                (squared error on the class index).

        Returns:
            The metric value.
        """
        if metric == _CONFIDENCE_METRIC:
            return float(self._predict_proba(model, x).max(axis=1).mean())

        preds = self._predict_class(model, x)
        if metric == _ORIGINAL_MSE_METRIC:
            return ordinal_metrics.rmse(preds, y) ** 2
        return ordinal_metrics.ORDINAL_METRICS[metric](preds, y)

    def _select_better_model(
            self,
            model_index: int,
            incumbent: LightningModule,
            candidate: LightningModule,
            original_dataset: tuple[torch.Tensor, torch.Tensor],
            val_data: torch.Tensor,
            val_label: torch.Tensor,
    ) -> LightningModule:
        """Keep whichever of ``incumbent`` / ``candidate`` scores better, per ``keep_best_model_mode``.

        Only the *weights* can revert — the pseudo-labels stay in the model's split either way,
        so a rejected iteration still changes what the next from-scratch training sees.

        Args:
            model_index: Which model is being updated (for the log line).
            incumbent: The model from the previous iteration.
            candidate: The freshly trained/fine-tuned model.
            original_dataset: The model's pre-pseudo-label split, used by
                ``MSE_ON_ORIGINAL_DATASET``.
            val_data: Validation features, used by every other mode.
            val_label: Validation labels.

        Returns:
            ``candidate`` when it improves the metric (or when no mode is configured),
            otherwise ``incumbent``.
        """
        if self.keep_best_model_mode is None:
            return candidate

        metric, direction = self._metric_for_mode(self.keep_best_model_mode)
        if metric == _ORIGINAL_MSE_METRIC:
            x, y = original_dataset
        else:
            x, y = val_data, val_label

        incumbent_score = self._score_model(incumbent, x, y, metric)
        candidate_score = self._score_model(candidate, x, y, metric)

        improved = (candidate_score < incumbent_score if direction == "min"
                    else candidate_score > incumbent_score)

        self._log(1, f"[CoTraining]   Model {model_index} {self.keep_best_model_mode.value} "
                     f"(lower is better: {direction == 'min'}) | previous: {incumbent_score:.4f} | "
                     f"new: {candidate_score:.4f} | keeping the "
                     f"{'new' if improved else 'previous'} model")

        return candidate if improved else incumbent

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

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

    def _log(self, level: int, message: str) -> None:
        """Print a message when ``verbose`` allows it, and always append it to the log file."""
        if self.verbose >= level:
            print(message)
        # When a log file is configured, capture every message regardless of level;
        # append-per-call keeps it crash-safe and needs no file-handle lifecycle.
        if self._log_file_path is not None:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def _check_if_training_is_possible(
            self,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            calib_data: torch.Tensor,
            calib_label: torch.Tensor,
            metrics_enabled: bool,
            test_data: torch.Tensor | None,
            test_label: torch.Tensor | None,
    ) -> None:
        """Validate that :meth:`train` has everything it needs.

        Raises:
            ValueError: If the ensemble is unconfigured or a required split is missing.
        """
        if not self._configured:
            raise ValueError("You need to call setup_training_builder before calling train.")

        if val_data is None or val_label is None:
            raise ValueError(
                "val_data and val_label are required (used for early stopping / "
                "best-checkpoint selection).")

        if calib_data is None or calib_label is None:
            raise ValueError("calib_data and calib_label are required to calibrate the "
                             "conformal classifiers.")

        if metrics_enabled and (test_data is None or test_label is None):
            raise ValueError("test_data and test_label are required to log per-stage metrics.")

    def _predict(
            self,
            model: LightningModule,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model's forward pass, chunked when ``inference_batch_size`` is set.

        Args:
            model: The model to run.
            x: Input windows.

        Returns:
            The ``(N, num_classes)`` probability matrix, on ``x``'s device.
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

    def _predict_proba(self, model: LightningModule, x: torch.Tensor) -> np.ndarray:
        """Predict class probabilities as the float64 numpy array ``crepes`` expects."""
        return self._predict(model, x).detach().cpu().numpy().astype(np.float64)

    def _predict_class(self, model: LightningModule, x: torch.Tensor) -> torch.Tensor:
        """Predict hard class indices, shape ``(N,)``."""
        return self._predict(model, x).argmax(dim=1).detach().cpu()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the weighted-ensemble class probabilities.

        Args:
            x: Input windows.

        Returns:
            ``(N, num_classes)`` probabilities, the weighted mean of the members' matrices.

        Raises:
            ValueError: If the weights have not been computed yet.
        """
        if self.weights is None:
            raise ValueError("Weights are not computed, call calculate_weights first.")

        stacked = torch.stack(
            [w * self._predict(model, x).cpu() for w, model in zip(self.weights, self.lightning_modules)],
            dim=0,
        )
        return stacked.sum(dim=0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the weighted-ensemble class index.

        Args:
            x: Input windows.

        Returns:
            ``(N, 1)`` ``long`` tensor of class indices.
        """
        return self.predict_proba(x).argmax(dim=1, keepdim=True)

    def predict_per_model(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Predict the class index of each committee member separately.

        Args:
            x: Input windows.

        Returns:
            One ``(N, 1)`` ``long`` tensor per model.
        """
        return [self._predict_class(model, x).view(-1, 1) for model in self.lightning_modules]

    def calculate_weights(
            self,
            x_test: torch.Tensor,
            target: torch.Tensor,
    ) -> None:
        """Compute and store the ensemble weights from ``computing_weight_mode``.

        Args:
            x_test: Features to score the models on (pass the validation set, not the test set,
                to avoid leaking test information into the weighting).
            target: True class indices for ``x_test``.
        """
        self.weights = self._compute_weights(x_test, target)
        self._log(1, f"[CoTraining] Weights: {[round(w, 4) for w in self.weights]}")

    def _compute_weights(
            self,
            x_test: torch.Tensor,
            target: torch.Tensor,
            models: list[LightningModule] | None = None,
    ) -> list[float]:
        """Compute normalized ensemble weights **without** mutating ``self.weights``.

        Holds the scoring→weight math shared by :meth:`calculate_weights` (which stores the
        result) and the per-stage metrics logging in :meth:`train` (which needs weights for the
        "weighted" columns without changing the ensemble's state).

        For a "min" metric the weights are inversely proportional to the score. A score of
        exactly 0 (a perfect model) makes the inverse undefined, so in that case the weight is
        split evenly among the perfect models — crashing a long sweep over a perfect score
        would be worse than the degenerate-but-correct answer.

        Args:
            x_test: Features to score on.
            target: True class indices.
            models: The models to weight. Defaults to ``self.lightning_modules``. During
                ``train`` the trained models live in a local list, so the per-stage metrics
                must pass that list explicitly.

        Returns:
            One normalized weight per model, summing to 1.

        Raises:
            ValueError: If a "max" metric scores 0 for every model, leaving nothing to weight by.
        """
        if models is None:
            models = self.lightning_modules

        metric, mode = self._metric_for_mode(self.computing_weight_mode)
        target_flat = target.detach().cpu().reshape(-1)

        scores = [self._score_model(model, x_test, target_flat, metric) for model in models]

        self._log(1, f"[CoTraining] Calculating weights | metric: {metric} ({mode}) | "
                     f"scores per model: {[round(s, 4) for s in scores]}")

        if mode == "min":
            if any(s == 0 for s in scores):
                perfect = [1.0 if s == 0 else 0.0 for s in scores]
                total = sum(perfect)
                return [p / total for p in perfect]
            inv_scores = [1.0 / s for s in scores]
            total = sum(inv_scores)
            return [inv_s / total for inv_s in inv_scores]

        total = sum(scores)
        if total == 0:
            raise ValueError(f"The sum of scores from all models is zero, cannot calculate weights : {scores}")
        return [s / total for s in scores]

    @staticmethod
    def _metric_set(
            preds: torch.Tensor,
            target: torch.Tensor,
            prefix: str,
    ) -> dict[str, float]:
        """Score one prediction vector with every metric in :data:`_REPORTED_METRICS`.

        Args:
            preds: Predicted class indices.
            target: True class indices, same number of elements as ``preds``.
            prefix: Split name prepended to each key (``"val"`` or ``"test"``).

        Returns:
            ``{f"{prefix}_{metric}": value}`` for every reported metric.
        """
        return {
            f"{prefix}_{name}": _REPORTED_METRIC_FUNCTIONS[name](preds, target)
            for name in _REPORTED_METRICS
        }

    @staticmethod
    def _weighted_prediction(
            weights: list[float],
            probs_per_model: list[torch.Tensor],
    ) -> torch.Tensor:
        """Combine per-model probability matrices into the weighted-ensemble class index.

        Args:
            weights: One weight per model, in the same order as ``probs_per_model``.
            probs_per_model: Each model's ``(N, num_classes)`` probability matrix on one split.

        Returns:
            The ``(N,)`` predicted class indices of the weighted ensemble.
        """
        return torch.stack(
            [w * probs for w, probs in zip(weights, probs_per_model)], dim=0
        ).sum(dim=0).argmax(dim=1)

    def _log_stage_metrics(
            self,
            stage: str,
            h: list[LightningModule],
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]],
            test_data: torch.Tensor,
            test_label: torch.Tensor,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            metrics_file: str,
    ) -> None:
        """Append one row of per-stage metrics to ``metrics_file``.

        Records, per model, the train accuracy (on that model's own accumulated split) and the
        full picture on **both** the validation and the test split (accuracy, macro-F1, MAE and
        RMSE on the class index, and the Scania cost); then the arithmetic mean of those
        per-model validation and test metrics; then the same validation and test metrics for
        the weighted-ensemble prediction, whose weights are computed on the **validation** set
        (so ``self.weights`` is left untouched and no test information leaks into the weighting).

        Note that the validation split is what the weights are fitted on, so the
        ``weighted_val_*`` columns are optimistic by construction — they are there to explain
        the weights, not to compare against ``weighted_test_*``.

        Args:
            stage: Row label ("initial", "iteration_<k>" or "final").
            h: The current model per index.
            models_datasets: Per-model ``(x, y)`` accumulated training split.
            test_data: Test features.
            test_label: Test labels.
            val_data: Validation features.
            val_label: Validation labels.
            metrics_file: Destination CSV; the header is written only when it does not yet exist.
        """
        test_target = test_label.detach().cpu().reshape(-1)
        val_target = val_label.detach().cpu().reshape(-1)

        per_model: list[dict[str, float]] = []
        val_probs: list[torch.Tensor] = []
        test_probs: list[torch.Tensor] = []

        for j, model in enumerate(h):
            xj, yj = models_datasets[j]
            train_pred = self._predict_class(model, xj)

            val_probs_j = self._predict(model, val_data).cpu()
            val_probs.append(val_probs_j)
            val_pred = val_probs_j.argmax(dim=1)

            test_probs_j = self._predict(model, test_data).cpu()
            test_probs.append(test_probs_j)
            test_pred = test_probs_j.argmax(dim=1)

            per_model.append({
                "train_accuracy": ordinal_metrics.accuracy(train_pred, yj),
                **self._metric_set(val_pred, val_target, prefix="val"),
                **self._metric_set(test_pred, test_target, prefix="test"),
            })

        n = len(h)
        val_keys = [f"val_{name}" for name in _REPORTED_METRICS]
        test_keys = [f"test_{name}" for name in _REPORTED_METRICS]
        summary_keys = val_keys + test_keys
        averages = {key: sum(m[key] for m in per_model) / n for key in summary_keys}

        weights = self._compute_weights(val_data, val_label, models=h)
        weighted_val_pred = self._weighted_prediction(weights, val_probs)
        weighted_test_pred = self._weighted_prediction(weights, test_probs)
        weighted = {
            **self._metric_set(weighted_val_pred, val_target, prefix="val"),
            **self._metric_set(weighted_test_pred, test_target, prefix="test"),
        }

        model_keys = ["train_accuracy"] + summary_keys

        header = ["stage"]
        row: list[str | float] = [stage]
        for j in range(n):
            header += [f"{key}_{j}" for key in model_keys]
            row += [per_model[j][key] for key in model_keys]
        header += [f"avg_{key}" for key in summary_keys]
        row += [averages[key] for key in summary_keys]
        header += [f"weighted_{key}" for key in summary_keys]
        row += [weighted[key] for key in summary_keys]
        header += [f"weight_{j}" for j in range(n)]
        row += list(weights)

        # Append per call (crash-safe, no file-handle lifecycle), writing the header only
        # the first time the file is created — mirrors the append style of ``_log``.
        write_header = not os.path.exists(metrics_file)
        with open(metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        self._log(1, f"[CoTraining] Metrics [{stage}] | "
                     f"avg val accuracy: {averages['val_accuracy']:.4f} | "
                     f"avg val macro-F1: {averages['val_f1_macro']:.4f} | "
                     f"avg val MAE: {averages['val_mae']:.4f} | "
                     f"avg val RMSE: {averages['val_rmse']:.4f} | "
                     f"avg val cost: {averages['val_cost']:.1f} | "
                     f"weighted val accuracy: {weighted['val_accuracy']:.4f} | "
                     f"weighted val macro-F1: {weighted['val_f1_macro']:.4f} | "
                     f"weighted val MAE: {weighted['val_mae']:.4f} | "
                     f"weighted val RMSE: {weighted['val_rmse']:.4f} | "
                     f"weighted val cost: {weighted['val_cost']:.1f} | "
                     f"avg test accuracy: {averages['test_accuracy']:.4f} | "
                     f"avg test macro-F1: {averages['test_f1_macro']:.4f} | "
                     f"avg test MAE: {averages['test_mae']:.4f} | "
                     f"avg test RMSE: {averages['test_rmse']:.4f} | "
                     f"avg test cost: {averages['test_cost']:.1f} | "
                     f"weighted test accuracy: {weighted['test_accuracy']:.4f} | "
                     f"weighted test macro-F1: {weighted['test_f1_macro']:.4f} | "
                     f"weighted test MAE: {weighted['test_mae']:.4f} | "
                     f"weighted test RMSE: {weighted['test_rmse']:.4f} | "
                     f"weighted test cost: {weighted['test_cost']:.1f}")
