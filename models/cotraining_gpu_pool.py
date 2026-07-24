"""Process-based multi-GPU training pool for the ``CoTrainingEnsemble`` algorithms.

Both co-training ensembles (``models/CoTrainingEnsemble.py`` v1 and
``models/CoTrainingEnsemble_v2.py`` v2) perform many *independent* per-model operations
per phase — the initial from-scratch fit of every model, the per-candidate train /
fine-tune of the censored-data search (v1), the per-model conformal scoring (v2), and the
end-of-iteration retrain / fine-tune. This module lets those independent operations run
concurrently, each pinned to a specific physical GPU, using one persistent worker process
per GPU.

It mirrors ``models/coprog_gpu_pool.py`` (spawn-based workers, ``CUDA_VISIBLE_DEVICES``
pinning, a per-worker input queue plus one shared output queue, ``gather`` with dead-worker
detection, CPU-thread capping) and **reuses** its from-scratch training primitives
(:class:`~models.coprog_gpu_pool.TrainingSpec`, :func:`~models.coprog_gpu_pool.run_training_job`,
:class:`~models.coprog_gpu_pool.CandidateContext` and the label/sse helpers). On top of that
it adds the job kinds the co-training ensembles need but COPROG does not:

* **fine-tune** jobs — warm-start from a *provided* (current) state dict rather than the
  from-scratch snapshot, scale the learning rate, optionally freeze parameters by name, and
  train with a (possibly smaller) fine-tune epoch budget (v1 only).
* **conformal-score** jobs — build a ``crepes`` normalized conformal regressor per model and
  return each pooled censored unit's interval width + per-window pseudo-labels (v2 only).

Unlike COPROG (exactly two models, GPU list split in half), the co-training ensembles have
``N`` models, so callers distribute jobs **round-robin** across all GPUs via
:meth:`CoTrainingGpuPool.round_robin_gpu`.
"""

import os
import queue as _queue
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.multiprocessing as mp
from lightning import LightningModule, Trainer
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader, TensorDataset

# Reuse COPROG's proven from-scratch primitives so there is a single training code path for
# every "train a fresh model from the initial snapshot" job (initial fit, from-scratch
# candidate, from-scratch retrain).
from models.coprog_gpu_pool import (  # noqa: F401  (re-exported for callers)
    CandidateContext,
    TrainingSpec,
    _augment_labels,
    _summed_squared_error,
    run_training_job,
)

# Sentinel used to tell a worker to exit its loop.
_SHUTDOWN = "__shutdown__"


class _TorchRegressorAdapter:
    """Minimal sklearn-style learner exposing only ``predict`` for ``crepes.WrapRegressor``.

    ``WrapRegressor``/``DifficultyEstimator`` only ever call ``learner.predict(X)`` (never
    ``.fit``) and work with 2-D numpy arrays, whereas the wrapped torch model expects a
    ``(N, seq_len, n_features)`` tensor. This adapter bridges the two: it reshapes the
    flattened features back to the model layout, runs the model and returns a 1-D numpy array
    aligned with the numpy labels crepes uses for residuals.

    It lives here (not in the v2 module) so it can be used both by v2's in-process
    (sequential) conformal scoring and by :func:`run_conformal_score_job` inside a worker,
    without either importing the other.
    """

    def __init__(
            self,
            predict_fn: Callable[[LightningModule, torch.Tensor], torch.Tensor],
            model: LightningModule,
            seq_len: int,
            n_features: int,
    ) -> None:
        self._predict_fn = predict_fn
        self._model = model
        self._seq_len = seq_len
        self._n_features = n_features

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        x = x.view(-1, self._seq_len, self._n_features)
        preds = self._predict_fn(self._model, x)
        return preds.detach().cpu().numpy().reshape(-1)


def _predict(model: LightningModule, x: torch.Tensor) -> torch.Tensor:
    """Run ``model`` on ``x`` (moved to the model's device) without tracking gradients."""
    model.eval()
    with torch.no_grad():
        x = x.to(next(model.parameters()).device)
        return model(x)


def _flatten(x: torch.Tensor) -> np.ndarray:
    """Flatten ``(N, seq_len, n_features)`` features to a 2-D ``(N, seq_len*n_features)``
    float32 numpy array, as required by ``crepes`` and its kNN ``DifficultyEstimator``."""
    return x.reshape(x.shape[0], -1).detach().cpu().numpy().astype(np.float32)


def _monotone_project(
        preds: torch.Tensor,
        lower_bounds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Project a censored unit's per-window RUL predictions onto the closest valid sequence.

    A censored unit's windows are ordered oldest -> newest, so its true RUL must be
    **non-increasing** along that axis. The raw per-window predictions (independent forward
    passes) are noisy and need not respect that, so they are projected onto the closest
    non-increasing sequence via isotonic regression (``increasing=False``). When
    ``lower_bounds`` is given (the per-window survival lower bound of a censored unit, itself
    non-increasing), the projection is additionally clipped up to it — the ``max`` of two
    non-increasing sequences stays non-increasing, so monotonicity is preserved and the clip
    distance is folded into the residual.

    The **residual** (mean absolute deviation between the raw and projected sequence) measures
    how far the model's predictions had to move to become physically valid: a small residual
    means the model was already self-consistent on that unit (a strong trust signal), a large
    one means it was incoherent. It is defined here once and reused by both v2 scoring paths
    (sequential in ``CoTrainingEnsemble_v2`` and this module's :func:`run_conformal_score_job`)
    so the two can never drift.

    :param preds: Raw per-window predictions, shape ``(m,)`` or ``(m, 1)``.
    :param lower_bounds: Optional per-window lower bounds, broadcastable to ``(m,)``; projected
        values below their bound are raised to it. ``None`` skips the censoring clip.
    :return: ``(projected, residual)`` where ``projected`` has the same shape/dtype as
        ``preds`` and ``residual`` is ``mean(|raw - projected|)`` (``0.0`` when nothing moved).
    """
    from sklearn.isotonic import IsotonicRegression

    r = preds.detach().cpu().reshape(-1).numpy().astype(np.float64)
    m = r.shape[0]

    # A single window is trivially "monotone" (nothing to order), but a lower-bound clip can
    # still apply, so only the isotonic step is skipped for m == 1.
    if m == 1:
        proj = r.copy()
    else:
        proj = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
            np.arange(m), r)

    if lower_bounds is not None:
        lb = lower_bounds.detach().cpu().reshape(-1).numpy().astype(np.float64)
        proj = np.maximum(proj, lb)

    residual = float(np.mean(np.abs(r - proj)))
    projected = torch.from_numpy(proj).to(dtype=preds.dtype).reshape(preds.shape)
    return projected, residual


def _reshape_pseudo_labels(reference_y: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """Reshape ``pseudo_labels`` to match the target layout of ``reference_y``.

    Mirrors the ``lu_p.view(-1, y.shape[1]) if y.dim() > 1 else lu_p.view(-1)`` reshaping used
    at the call sites of v1's ``_fine_tune_fun`` so the fine-tune target set is built
    identically whether the fine-tune runs inline or in a worker. Unlike
    :func:`~models.coprog_gpu_pool._augment_labels`, it does *not* concatenate (a fine-tune
    candidate trains on the candidate's data only).
    """
    if reference_y.dim() > 1:
        return pseudo_labels.view(-1, reference_y.shape[1])
    return pseudo_labels.view(-1)


@dataclass
class FineTuneSpec:
    """Fully self-contained description of one fine-tuning job (v1).

    Unlike :class:`~models.coprog_gpu_pool.TrainingSpec` (which loads the shared *initial*
    snapshot and trains from scratch), a fine-tune warm-starts from ``current_state_dict``
    (the model's current trained weights), scales its learning rate by ``lr_factor`` and
    optionally freezes every parameter whose name is *not* in ``trainable_param_names``.

    All fields are picklable (CPU tensors / plain data / a module-level callable or
    ``functools.partial``) so a spec can cross a process boundary.

    :param module_builder: Picklable callable returning a *fresh* ``LightningModule``.
    :param current_state_dict: CPU ``state_dict`` of the current model, loaded before
        fine-tuning (the warm start).
    :param lr_factor: Multiplier applied to ``model.lr`` for the fine-tune (e.g. ``0.1``).
        Ignored if the module has no ``lr`` attribute.
    :param trainable_param_names: If not ``None``, the set of ``state_dict``/``named_parameters``
        names to keep trainable; every other parameter is frozen (``requires_grad=False``)
        for this fine-tune. ``None`` fine-tunes all parameters.
    :param max_epochs: Fine-tune epoch budget.
    :param patience: ``EarlyStopping`` patience (monitors ``val_loss``).
    :param batch_size: Training/validation batch size.
    :param shuffle: Whether to shuffle the training ``DataLoader``.
    :param train_x: Fine-tune features (CPU tensor) — only the newly added data.
    :param train_y: Fine-tune targets (CPU tensor).
    :param val_x: Optional validation features for early stopping / checkpointing.
    :param val_y: Optional validation targets.
    :param eval_x: Optional evaluation features; if given, the trained model's summed
        squared error on ``(eval_x, eval_y)`` is returned as ``"sse"`` (used for the v1
        candidate delta).
    :param eval_y: Optional evaluation targets (required iff ``eval_x`` is given).
    :param return_state: If ``True``, the trained CPU ``state_dict`` is returned as
        ``"state_dict"``.
    :param accelerator: ``Trainer`` accelerator (``"gpu"`` in workers).
    :param devices: ``Trainer`` devices argument (``1`` in workers, ``[gpu_id]`` / ``None``
        for the inline path).
    """

    module_builder: Callable[[], LightningModule]
    current_state_dict: dict[str, torch.Tensor]
    lr_factor: float
    trainable_param_names: set[str] | None
    max_epochs: int
    patience: int
    batch_size: int
    shuffle: bool
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None
    eval_x: torch.Tensor | None = None
    eval_y: torch.Tensor | None = None
    return_state: bool = False
    accelerator: str = "auto"
    devices: Any = None


@dataclass
class FineTuneCandidateContext:
    """Static per-(model, iteration) warm-start data cached on a worker for v1 candidate scoring.

    During the fine-tune candidate search a model fine-tunes one temporary model per candidate
    censored unit; every one of those fine-tunes shares the same current weights, labelled
    evaluation set ``L``, validation set, builder, freeze spec and trainer config. Caching that
    (potentially large) data once per iteration — instead of once per candidate — avoids
    re-serialising it for every candidate. Each candidate job then only carries the small
    ``xu`` + pseudo-label ``lu_p`` (see :meth:`CoTrainingGpuPool.submit_finetune_candidate`).
    """

    module_builder: Callable[[], LightningModule]
    current_state_dict: dict[str, torch.Tensor]
    lr_factor: float
    trainable_param_names: set[str] | None
    max_epochs: int
    patience: int
    batch_size: int
    shuffle: bool
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None


@dataclass
class ConformalScoreSpec:
    """Description of one per-model conformal-scoring job (v2).

    The worker builds the trained model, wraps it in a ``crepes`` normalized conformal
    regressor (``DifficultyEstimator`` fitted on ``train_x``, calibrated on the validation
    set), predicts each pooled unit's per-window pseudo-labels and computes the prediction
    interval of each unit's last window. Only small artefacts cross back (per-unit widths +
    pseudo-labels).

    :param module_builder: Picklable callable returning a *fresh* ``LightningModule``.
    :param state_dict: CPU ``state_dict`` of the trained model to score with.
    :param train_x: The model's (flattened-internally) training features, for the
        ``DifficultyEstimator``.
    :param val_x: Validation features used to calibrate the conformal regressor.
    :param val_y: Validation targets used to calibrate the conformal regressor.
    :param unit_ids: The pooled censored unit ids (ints), aligned with ``unit_x``.
    :param unit_x: Per-unit censored sequences (list of CPU tensors), one per ``unit_ids``.
    :param confidence: Confidence level in ``(0, 1)`` passed to ``predict_int``.
    :param use_monotone_projection: When ``True``, each unit's per-window pseudo-labels are
        projected onto the closest non-increasing (and, if ``unit_lower_bounds`` is given,
        lower-bound-clipped) sequence via :func:`_monotone_project`; the projected labels are
        returned instead of the raw predictions and a per-unit residual is returned too.
    :param unit_lower_bounds: Optional per-unit survival lower bounds (list of CPU tensors,
        one per ``unit_ids``, aligned row-for-row with each ``unit_x`` entry). Used for the
        censoring clip when ``use_monotone_projection`` is set; ``None`` skips the clip.
    :param accelerator: ``Trainer``-style accelerator for placing the model (``"gpu"`` in workers).
    :param devices: Device selector (``1`` / ``[gpu_id]`` / ``None``); only ``"cuda"`` vs
        ``"cpu"`` placement is used here.
    """

    module_builder: Callable[[], LightningModule]
    state_dict: dict[str, torch.Tensor]
    train_x: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    unit_ids: list[int]
    unit_x: list[torch.Tensor]
    confidence: float
    use_monotone_projection: bool = False
    unit_lower_bounds: list[torch.Tensor] | None = None
    accelerator: str = "auto"
    devices: Any = None


@dataclass
class CpsScoreSpec:
    """Description of one per-model **Conformal Predictive System** scoring job (v3).

    Like :class:`ConformalScoreSpec`, but the model is wrapped in a ``crepes`` *conformal
    predictive system* (``calibrate(cps=True)``) instead of a plain conformal regressor, so
    each pooled unit's last window yields the requested predictive **percentiles**
    ``a = p_low``, ``c = p50``, ``b = p_high`` (asymmetric, so ``c`` need not be the midpoint of
    ``[a, b]``) rather than a single symmetric interval. ``width = b - a`` drives selection while
    ``a, b, c`` feed v3's label estimator. The ``DifficultyEstimator`` (fitted on ``train_x``)
    is used exactly as in the interval path, so the percentile spread still varies per unit.

    :param percentiles: The three percentiles to read, in order ``[low, 50, high]`` (e.g.
        ``[5.0, 50.0, 95.0]`` for a 90% band). Passed to ``predict_percentiles``.

    All other fields mirror :class:`ConformalScoreSpec`.
    """

    module_builder: Callable[[], LightningModule]
    state_dict: dict[str, torch.Tensor]
    train_x: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    unit_ids: list[int]
    unit_x: list[torch.Tensor]
    percentiles: list[float]
    use_monotone_projection: bool = False
    unit_lower_bounds: list[torch.Tensor] | None = None
    accelerator: str = "auto"
    devices: Any = None


def _fit_and_get_best_state(
        model: LightningModule,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        max_epochs: int,
        patience: int,
        accelerator: str,
        devices: Any,
) -> LightningModule:
    """Fit ``model`` with ``EarlyStopping`` + ``ModelCheckpoint`` (on ``val_loss``) in a
    throwaway temp dir, reload the best checkpoint, and return the model.

    Shared by :func:`run_finetune_job`; mirrors the checkpointing done by
    :func:`~models.coprog_gpu_pool.run_training_job` so fine-tune and from-scratch trainings
    behave identically w.r.t. best-model selection.
    """
    ckpt_dir = tempfile.mkdtemp(prefix="cotraining_ckpt_")
    try:
        early_stop_callback = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=patience,
            verbose=False,
            mode="min",
        )
        checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        )

        trainer_kwargs: dict[str, Any] = dict(
            default_root_dir=ckpt_dir,
            accelerator=accelerator,
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        if devices is not None:
            trainer_kwargs["devices"] = devices

        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = getattr(checkpoint_callback, "best_model_path", "")
        if best_model_path:
            device = next(model.parameters()).device
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        return model
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def run_finetune_job(spec: FineTuneSpec) -> dict[str, Any]:
    """Fine-tune one warm-started model and return the requested artefacts.

    Builds a fresh module, loads ``spec.current_state_dict`` (the warm start), scales
    ``model.lr`` by ``spec.lr_factor``, freezes every parameter not named in
    ``spec.trainable_param_names`` (when given), fits with the fine-tune epoch budget,
    reloads the best checkpoint, and optionally evaluates / exports weights.

    Module-level and picklable, so it runs unchanged in a worker process and inline.

    :param spec: The fine-tune job description.
    :return: A dict possibly containing ``"sse"`` (if ``spec.eval_x`` was given) and
        ``"state_dict"`` (CPU tensors, if ``spec.return_state`` was set).
    """
    model = spec.module_builder()
    model.load_state_dict(spec.current_state_dict)

    # Reduced learning rate so the update is a nudge rather than a full override.
    original_lr = getattr(model, "lr", None)
    if original_lr is not None:
        model.lr = original_lr * spec.lr_factor

    # Freeze everything but the requested parameters (by name).
    if spec.trainable_param_names is not None:
        for name, p in model.named_parameters():
            p.requires_grad = name in spec.trainable_param_names

    train_loader = DataLoader(
        TensorDataset(spec.train_x, spec.train_y),
        batch_size=spec.batch_size,
        shuffle=spec.shuffle,
    )
    val_loader = None
    if spec.val_x is not None and spec.val_y is not None:
        val_loader = DataLoader(TensorDataset(spec.val_x, spec.val_y), batch_size=spec.batch_size)

    model = _fit_and_get_best_state(
        model, train_loader, val_loader,
        max_epochs=spec.max_epochs,
        patience=spec.patience,
        accelerator=spec.accelerator,
        devices=spec.devices,
    )

    result: dict[str, Any] = {}
    if spec.eval_x is not None and spec.eval_y is not None:
        result["sse"] = _summed_squared_error(model, spec.eval_x, spec.eval_y)
    if spec.return_state:
        result["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return result


def run_conformal_score_job(spec: ConformalScoreSpec) -> dict[str, Any]:
    """Score a model's pooled censored units with ``crepes`` conformal intervals (v2).

    Builds the trained model, wraps it in a normalized conformal regressor
    (``DifficultyEstimator`` fitted on ``spec.train_x``, calibrated on the validation set),
    predicts each unit's per-window pseudo-labels, and computes the prediction interval of
    each unit's last window in one batched ``predict_int`` call. When
    ``spec.use_monotone_projection`` is set, each unit's pseudo-labels are additionally passed
    through :func:`_monotone_project` (returning the projected labels and a residual).

    ``crepes`` is imported lazily so this module (and v1's parallel path, which never scores
    conformally) does not hard-depend on it.

    :param spec: The conformal-scoring job description.
    :return: ``{"units": [(unit_id, label, lower, upper, width, residual, raw_label), ...]}``
        with one entry per pooled unit (order matches ``spec.unit_ids``). ``label`` is the raw
        per-window pseudo-label, or its monotone projection when
        ``spec.use_monotone_projection`` is set; ``raw_label`` is always the pre-projection
        prediction (equal to ``label`` when projection is off), kept for effectiveness logging;
        ``residual`` is ``0.0`` when projection is off.
    """
    from crepes import WrapRegressor
    from crepes.extras import DifficultyEstimator

    model = spec.module_builder()
    model.load_state_dict(spec.state_dict)
    if spec.accelerator == "gpu" and torch.cuda.is_available():
        model = model.to("cuda")

    train_x = spec.train_x
    seq_len, n_features = train_x.shape[1], train_x.shape[2]

    # Normalized conformal regressor: DifficultyEstimator makes widths vary per instance.
    de = DifficultyEstimator()
    de.fit(X=_flatten(train_x))
    wrapper = WrapRegressor(_TorchRegressorAdapter(_predict, model, seq_len, n_features))
    wrapper.calibrate(
        X=_flatten(spec.val_x),
        y=spec.val_y.view(-1).detach().cpu().numpy().astype(np.float32),
        de=de,
    )

    # Per-unit pseudo-labels + last windows.
    lu_ps: list[torch.Tensor] = []
    last_windows: list[torch.Tensor] = []
    for xu in spec.unit_x:
        lu_ps.append(_predict(model, xu).detach().cpu())
        last_windows.append(xu[-1])

    last_windows_tensor = torch.stack(last_windows, dim=0)
    intervals = wrapper.predict_int(_flatten(last_windows_tensor), confidence=spec.confidence)

    units: list[tuple[int, torch.Tensor, float, float, float, float, torch.Tensor]] = []
    for idx, unit_id in enumerate(spec.unit_ids):
        lower = float(intervals[idx, 0])
        upper = float(intervals[idx, 1])
        lu_p = lu_ps[idx]
        if spec.use_monotone_projection:
            lb_u = spec.unit_lower_bounds[idx] if spec.unit_lower_bounds is not None else None
            label, residual = _monotone_project(lu_p, lb_u)
        else:
            label, residual = lu_p, 0.0
        # raw_label (lu_p) is kept alongside the (possibly projected) label so the selection
        # step can log predictions before vs after projection.
        units.append((unit_id, label, lower, upper, upper - lower, residual, lu_p))
    return {"units": units}


def run_cps_score_job(spec: CpsScoreSpec) -> dict[str, Any]:
    """Score a model's pooled censored units with a ``crepes`` conformal predictive system (v3).

    Builds the trained model, wraps it in a **normalized conformal predictive system**
    (``DifficultyEstimator`` fitted on ``spec.train_x``, calibrated on the validation set with
    ``cps=True``), predicts each unit's raw per-window RUL and reads the requested predictive
    percentiles ``a = p_low``, ``c = p50``, ``b = p_high`` of each unit's last window in one
    batched ``predict_percentiles`` call. The per-window monotone residual (self-consistency
    signal) is computed via :func:`_monotone_project` exactly as in the interval path.

    ``crepes`` is imported lazily so this module does not hard-depend on it.

    :param spec: The CPS-scoring job description.
    :return: ``{"units": [(unit_id, raw_preds, a, b, width, residual, c), ...]}`` with one entry
        per pooled unit (order matches ``spec.unit_ids``). ``raw_preds`` are the per-window RUL
        predictions; ``width = b - a``; ``residual`` is ``0.0`` when projection is off.
    """
    from crepes import WrapRegressor
    from crepes.extras import DifficultyEstimator

    model = spec.module_builder()
    model.load_state_dict(spec.state_dict)
    if spec.accelerator == "gpu" and torch.cuda.is_available():
        model = model.to("cuda")

    train_x = spec.train_x
    seq_len, n_features = train_x.shape[1], train_x.shape[2]

    de = DifficultyEstimator()
    de.fit(X=_flatten(train_x))
    wrapper = WrapRegressor(_TorchRegressorAdapter(_predict, model, seq_len, n_features))
    wrapper.calibrate(
        X=_flatten(spec.val_x),
        y=spec.val_y.view(-1).detach().cpu().numpy().astype(np.float32),
        de=de,
        cps=True,
    )

    lu_ps: list[torch.Tensor] = []
    last_windows: list[torch.Tensor] = []
    for xu in spec.unit_x:
        lu_ps.append(_predict(model, xu).detach().cpu())
        last_windows.append(xu[-1])

    last_windows_tensor = torch.stack(last_windows, dim=0)
    # (U, 3) columns aligned with spec.percentiles == [low, 50, high] -> a, c, b.
    percentiles = wrapper.predict_percentiles(
        _flatten(last_windows_tensor), lower_percentiles=spec.percentiles)

    units: list[tuple[int, torch.Tensor, float, float, float, float, float]] = []
    for idx, unit_id in enumerate(spec.unit_ids):
        a = float(percentiles[idx, 0])
        c = float(percentiles[idx, 1])
        b = float(percentiles[idx, 2])
        lu_p = lu_ps[idx]
        if spec.use_monotone_projection:
            lb_u = spec.unit_lower_bounds[idx] if spec.unit_lower_bounds is not None else None
            _, residual = _monotone_project(lu_p, lb_u)
        else:
            residual = 0.0
        units.append((unit_id, lu_p, a, b, b - a, residual, c))
    return {"units": units}


def _worker_loop(
        gpu_id: int,
        num_threads: int,
        in_queue: "mp.Queue",
        out_queue: "mp.Queue",
) -> None:
    """Entry point of a per-GPU worker process.

    Pins the process to ``gpu_id`` via ``CUDA_VISIBLE_DEVICES`` (as the very first action,
    before any CUDA context is created), then serves messages until the shutdown sentinel.
    Any exception while handling a job is captured and reported on ``out_queue`` so the main
    process can re-raise it instead of hanging.

    Message protocol (tuples read from ``in_queue``):
      * ``("job", job_id, spec)`` — run :func:`run_training_job` (from-scratch train).
      * ``("finetune", job_id, spec)`` — run :func:`run_finetune_job` (warm-start fine-tune).
      * ``("conformal", job_id, spec)`` — run :func:`run_conformal_score_job` (v2).
      * ``("ctx", ctx_id, context)`` — cache a from-scratch :class:`CandidateContext`.
      * ``("ft_ctx", ctx_id, context)`` — cache a :class:`FineTuneCandidateContext`.
      * ``("candidate", job_id, ctx_id, xu, lu_p)`` — from-scratch candidate (returns ``sse``).
      * ``("ft_candidate", job_id, ctx_id, xu, lu_p)`` — fine-tune candidate (returns ``sse``).
      * ``("clear_ctx", ctx_id)`` — drop a cached context (either kind).
      * ``_SHUTDOWN`` — exit.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass

    contexts: dict[str, CandidateContext] = {}
    ft_contexts: dict[str, FineTuneCandidateContext] = {}

    while True:
        message = in_queue.get()
        if message == _SHUTDOWN:
            break

        kind = message[0]
        if kind == "ctx":
            _, ctx_id, context = message
            contexts[ctx_id] = context
            continue
        if kind == "ft_ctx":
            _, ctx_id, context = message
            ft_contexts[ctx_id] = context
            continue
        if kind == "clear_ctx":
            _, ctx_id = message
            contexts.pop(ctx_id, None)
            ft_contexts.pop(ctx_id, None)
            continue

        # Training messages carry a job_id we must always answer (result or error).
        job_id = message[1]
        try:
            if kind == "job":
                _, _, spec = message
                result = run_training_job(spec)
            elif kind == "finetune":
                _, _, spec = message
                result = run_finetune_job(spec)
            elif kind == "conformal":
                _, _, spec = message
                result = run_conformal_score_job(spec)
            elif kind == "cps":
                _, _, spec = message
                result = run_cps_score_job(spec)
            elif kind == "candidate":
                _, _, ctx_id, xu, lu_p = message
                ctx = contexts[ctx_id]
                spec = TrainingSpec(
                    module_builder=ctx.module_builder,
                    initial_state_dict=ctx.initial_state_dict,
                    max_epochs=ctx.max_epochs,
                    patience=ctx.patience,
                    batch_size=ctx.batch_size,
                    shuffle=ctx.shuffle,
                    train_x=torch.cat([ctx.labelled_x, xu], dim=0),
                    train_y=_augment_labels(ctx.labelled_y, lu_p),
                    val_x=ctx.val_x,
                    val_y=ctx.val_y,
                    eval_x=ctx.labelled_x,
                    eval_y=ctx.labelled_y,
                    return_state=False,
                    accelerator="gpu",
                    devices=1,
                )
                result = run_training_job(spec)
            elif kind == "ft_candidate":
                _, _, ctx_id, xu, lu_p = message
                ctx = ft_contexts[ctx_id]
                spec = FineTuneSpec(
                    module_builder=ctx.module_builder,
                    current_state_dict=ctx.current_state_dict,
                    lr_factor=ctx.lr_factor,
                    trainable_param_names=ctx.trainable_param_names,
                    max_epochs=ctx.max_epochs,
                    patience=ctx.patience,
                    batch_size=ctx.batch_size,
                    shuffle=ctx.shuffle,
                    # Fine-tune warm-starts from the current model, so only the candidate's
                    # data is used for training (matching v1's sequential _fine_tune_fun),
                    # with the pseudo-labels reshaped to the model's target layout.
                    train_x=xu,
                    train_y=_reshape_pseudo_labels(ctx.eval_y, lu_p),
                    val_x=ctx.val_x,
                    val_y=ctx.val_y,
                    eval_x=ctx.eval_x,
                    eval_y=ctx.eval_y,
                    return_state=False,
                    accelerator="gpu",
                    devices=1,
                )
                result = run_finetune_job(spec)
            else:
                raise ValueError(f"Unknown worker message kind: {kind!r}")
            out_queue.put(("result", job_id, result))
        except Exception:  # noqa: BLE001 - report any failure back to the parent
            out_queue.put(("error", job_id, traceback.format_exc()))


class CoTrainingGpuPool:
    """A persistent pool of one worker process per GPU for co-training ensemble jobs.

    Each GPU id in ``gpu_ids`` gets its own worker (pinned via ``CUDA_VISIBLE_DEVICES``) and
    its own input queue, so callers control exactly which GPU runs which job. Results are
    collected from a single shared queue keyed by job id.

    With ``N`` models, callers distribute jobs round-robin across all GPUs using
    :meth:`round_robin_gpu` (there is no per-model GPU split as in COPROG).

    :param gpu_ids: Physical GPU ids to spawn workers for.
    """

    def __init__(self, gpu_ids: list[int]) -> None:
        if not gpu_ids:
            raise ValueError("CoTrainingGpuPool requires at least one GPU id.")
        self.gpu_ids = list(gpu_ids)
        # CUDA requires the 'spawn' start method; it also works on Windows and Linux.
        self._ctx = mp.get_context("spawn")
        self._in_queues: dict[int, "mp.Queue"] = {}
        self._out_queue: "mp.Queue" = self._ctx.Queue()
        self._processes: dict[int, "mp.Process"] = {}
        self._job_counter = 0
        self._started = False

        cpu_count = os.cpu_count() or 4
        self._num_threads = max(1, cpu_count // len(self.gpu_ids))

    def round_robin_gpu(self, index: int) -> int:
        """Return the GPU id for the ``index``-th job in a round-robin distribution."""
        return self.gpu_ids[index % len(self.gpu_ids)]

    def start(self) -> None:
        """Spawn one worker process per GPU. Idempotent."""
        if self._started:
            return
        for gpu_id in self.gpu_ids:
            in_queue: "mp.Queue" = self._ctx.Queue()
            process = self._ctx.Process(
                target=_worker_loop,
                args=(gpu_id, self._num_threads, in_queue, self._out_queue),
                daemon=True,
            )
            process.start()
            self._in_queues[gpu_id] = in_queue
            self._processes[gpu_id] = process
        self._started = True

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def submit_job(self, gpu_id: int, spec: TrainingSpec) -> int:
        """Queue a from-scratch :class:`TrainingSpec` on ``gpu_id``'s worker."""
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("job", job_id, spec))
        return job_id

    def submit_finetune(self, gpu_id: int, spec: FineTuneSpec) -> int:
        """Queue a warm-start :class:`FineTuneSpec` on ``gpu_id``'s worker."""
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("finetune", job_id, spec))
        return job_id

    def submit_conformal(self, gpu_id: int, spec: ConformalScoreSpec) -> int:
        """Queue a :class:`ConformalScoreSpec` on ``gpu_id``'s worker (v2)."""
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("conformal", job_id, spec))
        return job_id

    def submit_cps(self, gpu_id: int, spec: CpsScoreSpec) -> int:
        """Queue a :class:`CpsScoreSpec` on ``gpu_id``'s worker (v3)."""
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("cps", job_id, spec))
        return job_id

    def set_context(self, gpu_ids: list[int], ctx_id: str, context: CandidateContext) -> None:
        """Cache a from-scratch :class:`CandidateContext` on every worker in ``gpu_ids``."""
        for gpu_id in gpu_ids:
            self._in_queues[gpu_id].put(("ctx", ctx_id, context))

    def set_finetune_context(
            self, gpu_ids: list[int], ctx_id: str, context: FineTuneCandidateContext,
    ) -> None:
        """Cache a :class:`FineTuneCandidateContext` on every worker in ``gpu_ids``."""
        for gpu_id in gpu_ids:
            self._in_queues[gpu_id].put(("ft_ctx", ctx_id, context))

    def clear_context(self, gpu_ids: list[int], ctx_id: str) -> None:
        """Drop a cached context (either kind) from the given workers to free memory."""
        for gpu_id in gpu_ids:
            self._in_queues[gpu_id].put(("clear_ctx", ctx_id))

    def submit_candidate(
            self, gpu_id: int, ctx_id: str, xu: torch.Tensor, lu_p: torch.Tensor,
    ) -> int:
        """Queue a from-scratch candidate-scoring job referencing a cached context."""
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("candidate", job_id, ctx_id, xu, lu_p))
        return job_id

    def submit_finetune_candidate(
            self, gpu_id: int, ctx_id: str, xu: torch.Tensor, lu_p: torch.Tensor,
    ) -> int:
        """Queue a fine-tune candidate-scoring job referencing a cached fine-tune context."""
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("ft_candidate", job_id, ctx_id, xu, lu_p))
        return job_id

    def gather(self, job_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Block until all ``job_ids`` have results, returning them keyed by job id.

        :raises RuntimeError: If any worker reported an exception (its traceback is included)
            or a worker died before answering.
        """
        pending = set(job_ids)
        results: dict[int, dict[str, Any]] = {}
        while pending:
            try:
                message = self._out_queue.get(timeout=1.0)
            except _queue.Empty:
                if any(not p.is_alive() for p in self._processes.values()):
                    raise RuntimeError(
                        "A GPU worker process died before returning all results "
                        f"(still waiting on job ids {sorted(pending)})."
                    )
                continue

            status, job_id, payload = message
            if job_id not in pending:
                continue
            if status == "error":
                raise RuntimeError(f"GPU worker job {job_id} failed:\n{payload}")
            results[job_id] = payload
            pending.discard(job_id)
        return results

    def shutdown(self) -> None:
        """Stop all workers and join them. Safe to call more than once."""
        if not self._started:
            return
        for gpu_id in self.gpu_ids:
            try:
                self._in_queues[gpu_id].put(_SHUTDOWN)
            except Exception:
                pass
        for process in self._processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._processes.clear()
        self._in_queues.clear()
        self._started = False
