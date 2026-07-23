from collections import OrderedDict
from typing import Callable

import numpy as np
import sklearn
import torch
from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator
from lightning import LightningModule

from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2
from models.cotraining_gpu_pool import FineTuneSpec, _TorchRegressorAdapter, run_finetune_job


class _LatentDifficultyEstimator:
    """``crepes`` difficulty-estimator adapter that measures difficulty in a model's latent space.

    ``crepes`` normalises interval widths by calling ``de.apply(X)`` on the same (raw, flattened)
    features it hands the learner. This adapter intercepts that call: it reshapes ``X`` back to
    sequences, embeds them with ``embed_fn`` (the owning model's pre-head latent), and delegates to
    a latent-fitted :class:`crepes.extras.DifficultyEstimator`. So the learner still predicts on raw
    features while the difficulty (and hence each unit's width) comes from the model's own
    representation.
    """

    def __init__(
            self,
            embed_fn: Callable[[torch.Tensor], np.ndarray],
            inner: DifficultyEstimator,
            seq_len: int,
            n_features: int,
    ):
        self._embed_fn = embed_fn
        self._inner = inner
        self._seq_len = seq_len
        self._n_features = n_features

    def apply(self, X: np.ndarray = None) -> np.ndarray:
        """Embed the flattened features ``X`` and return the latent-space difficulty (``sigmas``)."""
        x = torch.as_tensor(np.asarray(X, dtype=np.float32)).reshape(
            -1, self._seq_len, self._n_features)
        return self._inner.apply(self._embed_fn(x))


class CoTrainingEnsemble_v3(CoTrainingEnsemble_v2):
    """Single-GPU co-training ensemble with confidence-based owner selection (v3).

    v3 is a deliberately simple, sequential (single-GPU) rewrite. It keeps v2's co-training
    idea — trusted censored units are pseudo-labelled by the models that are most confident
    about them and injected into the other models — but changes how confidence, selection and
    labelling work, and adds a per-model best-model retention safeguard:

    * **Confidence from a ``predict_int`` interval.** Each model is wrapped in a ``crepes``
      normalized conformal regressor (calibrated on a dedicated calibration set when available,
      else the validation set) and each censored unit's last window yields a prediction interval
      ``[lower, upper]``. A tighter interval is more confident. The interval's per-unit width is
      normalised by a ``crepes`` ``DifficultyEstimator`` whose neighbourhood is measured either in
      the raw feature space (shared across models) or in each model's own **latent space**
      (``difficulty_space``) — the latter lets the models genuinely disagree about which units
      they understand best. The confidence also rewards physically consistent predictions: a
      unit's per-window RUL should decrease over time and never drop below its survival lower
      bound (see :meth:`_confidence_score`).
    * **Owner-based selection by confidence.** Each model selects its own top ``add_ratio``
      fraction of the pool by confidence (optionally gated by ``confidence_threshold``). A unit's
      *owners* are the models that selected it; the *receivers* (the rest) are taught its label.
      A unit selected by nobody, or by everybody, is skipped (no information asymmetry).
    * **Pseudo-label = the owner's own prediction.** The last window's label is the owner model's
      own last-window prediction; with several owners it is their prediction averaged with weights
      ``1 / val_rmse`` (more accurate models count more). Earlier windows are filled by backward
      extrapolation ``RUL_i = label_last + (t_last - t_i)`` using the per-window ``time_step``
      (non-increasing by construction).
    * **Best-model retention.** Each model keeps its best validation-RMSE weights and dataset.
      A receiver is fine-tuned (warm-started) on its accumulated data plus the new units; if that
      makes its validation RMSE worse, the model is reverted and the units added this iteration
      are dropped (logged with the count removed).

    Everything else (setup, prediction, ensemble weighting, per-stage metrics, the calibrated
    regressor, the from-scratch fit) is inherited unchanged from :class:`CoTrainingEnsemble_v2`.
    """

    def __init__(
            self,
            models: list,
            weights: list[float] | None = None,
            verbose: int = 0,
            confidence: float = 0.9,
            inference_batch_size: int | None = None,
            difficulty_space: str = "raw",
            confidence_threshold: float | None = None,
            w_mono: float = 1.0,
            w_lb: float = 1.0,
            keep_best_model: bool = True,
            acceptance_metric: str = "val_rmse",
            fine_tune_lr_factor: float = 0.1,
            fine_tune_max_epochs: int = 20,
            fine_tune_patience: int = 5,
    ):
        """Build a v3 co-training ensemble.

        Args:
            models: The models used in the co-training ensemble.
            weights: Optional pre-defined per-model ensemble weights.
            verbose: Verbosity level (0 silent, 1 key decisions, 2 full per-candidate detail).
            confidence: Confidence level in ``(0, 1)`` passed to ``crepes`` ``predict_int`` when
                building the prediction interval whose width scores each censored unit. Defaults
                to ``0.9``.
            inference_batch_size: If set, forward passes run in chunks of this many samples to cap
                peak memory. ``None`` keeps single-shot inference.
            difficulty_space: Where the conformal ``DifficultyEstimator`` measures a unit's
                neighbourhood (which normalises its interval width). ``"raw"`` (default) uses the
                flattened input features — the same for every model, so the models rank units
                almost identically. ``"latent"`` uses each model's own pre-head embedding, so the
                models genuinely disagree about which units they find easy/hard (recommended when
                selection stalls because the per-model rankings are near-identical).
            confidence_threshold: Optional lower bound in ``(0, 1]`` on a unit's confidence score.
                When set, a model may only select a unit whose confidence is at least this value
                (useful to reject clearly bad units). ``None`` (default) disables the gate, so
                selection is pure top-``add_ratio`` ranking — the mode used when exploring which
                units are worth trusting.
            w_mono: Non-negative weight of the monotonicity-violation term in the confidence score.
            w_lb: Non-negative weight of the lower-bound-violation term in the confidence score.
            keep_best_model: When ``True`` (default), a fine-tuned model is kept only if its
                acceptance metric improved; otherwise the model is reverted and that iteration's
                added units are dropped. When ``False``, every fine-tuned model is kept regardless
                (no rollback, no dropping) — the baseline to compare best-model retention against.
            acceptance_metric: What best-model retention minimises on the validation set.
                ``"val_rmse"`` (default) uses the validation RMSE. ``"score"`` uses the
                ``score_callback`` passed to :meth:`train` (e.g. the asymmetric Scania score, which
                penalises over-prediction) — useful when val RMSE improves but the test score keeps
                worsening. The label-weighting ``1 / val_rmse`` always uses RMSE regardless.
            fine_tune_lr_factor: Multiplier applied to each model's learning rate during
                fine-tuning (warm start), e.g. ``0.1``.
            fine_tune_max_epochs: Max epochs per fine-tuning call.
            fine_tune_patience: ``EarlyStopping`` patience per fine-tuning call.
        """
        super().__init__(
            models=models,
            weights=weights,
            verbose=verbose,
            confidence=confidence,
            inference_batch_size=inference_batch_size,
        )

        if difficulty_space not in ("raw", "latent"):
            raise ValueError("difficulty_space must be 'raw' or 'latent'.")
        if acceptance_metric not in ("val_rmse", "score"):
            raise ValueError("acceptance_metric must be 'val_rmse' or 'score'.")
        if confidence_threshold is not None and not (0 < confidence_threshold <= 1):
            raise ValueError("confidence_threshold must be in (0, 1] or None.")
        if w_mono < 0 or w_lb < 0:
            raise ValueError("w_mono and w_lb must be non-negative.")
        if fine_tune_lr_factor <= 0:
            raise ValueError("fine_tune_lr_factor must be positive.")

        self.difficulty_space = difficulty_space
        self.keep_best_model = keep_best_model
        self.acceptance_metric = acceptance_metric
        self.confidence_threshold = confidence_threshold
        self.w_mono = w_mono
        self.w_lb = w_lb
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.fine_tune_max_epochs = fine_tune_max_epochs
        self.fine_tune_patience = fine_tune_patience

    # ------------------------------------------------------------------ #
    # Latent-space difficulty (optional, difficulty_space="latent")
    # ------------------------------------------------------------------ #

    @staticmethod
    def _head_module(module: LightningModule) -> torch.nn.Module:
        """Return the final regression head; the tensor fed into it is the model's latent vector.

        All Scania architectures wrap their network as ``BasicLightningModule.net`` and end in a
        head named ``regressor`` (CNN1D) or ``linear`` (LSTM / transformer variants).
        """
        net = getattr(module, "net", module)
        head = getattr(net, "regressor", None)
        if head is None:
            head = getattr(net, "linear", None)
        if head is None:
            raise ValueError(
                "Cannot locate the regression head (expected 'regressor' or 'linear') for latent "
                "embedding extraction.")
        return head

    def _embed(self, module: LightningModule, x: torch.Tensor) -> np.ndarray:
        """Embed ``x`` into the model's latent space = the input to its regression head.

        A forward-pre-hook on the head captures its input during an ``eval`` / ``no_grad`` forward
        pass (chunked by ``inference_batch_size`` to cap memory). The hook is always removed.

        Args:
            module: The trained model to embed with.
            x: Input features ``(N, seq_len, n_features)``.

        Returns:
            Latent embeddings ``(N, D)`` as a CPU float32 numpy array (``D`` is model-specific).
        """
        module.eval()
        device = next(module.parameters()).device
        captured: dict[str, torch.Tensor] = {}

        def _hook(_mod: torch.nn.Module, inputs: tuple) -> None:
            captured["z"] = inputs[0].detach().cpu()

        handle = self._head_module(module).register_forward_pre_hook(_hook)
        try:
            with torch.no_grad():
                if self._inference_batch_size is None:
                    module(x.to(device))
                    z = captured["z"]
                else:
                    chunks: list[torch.Tensor] = []
                    for start in range(0, len(x), self._inference_batch_size):
                        module(x[start:start + self._inference_batch_size].to(device))
                        chunks.append(captured["z"])
                    z = torch.cat(chunks, dim=0)
        finally:
            handle.remove()
        return z.reshape(z.shape[0], -1).numpy().astype(np.float32)

    def _build_latent_calibrated_regressor(
            self,
            model: LightningModule,
            train_x: torch.Tensor,
            calib_x: torch.Tensor,
            calib_y: torch.Tensor,
    ) -> WrapRegressor:
        """Wrap ``model`` in a conformal regressor whose width is normalised by *latent* difficulty.

        A :class:`crepes.extras.DifficultyEstimator` is fitted on the model's latent embeddings of
        ``train_x`` and wrapped in a :class:`_LatentDifficultyEstimator` so ``crepes`` measures every
        unit's difficulty in this model's own representation (while the learner keeps predicting on
        the raw flattened features). The wrapper stores that ``de``, so a later ``predict_int`` call
        automatically re-derives the latent difficulty for the query points — no manual ``sigmas``.

        Args:
            model: The trained model to wrap.
            train_x: The model's accumulated training features (the difficulty reference set).
            calib_x: Calibration features.
            calib_y: Calibration labels.

        Returns:
            The calibrated regressor (its ``de`` measures difficulty in the model's latent space).
        """
        seq_len, n_features = train_x.shape[1], train_x.shape[2]
        sklearn.set_config(working_memory=128)

        inner = DifficultyEstimator()
        inner.fit(X=self._embed(model, train_x))
        latent_de = _LatentDifficultyEstimator(
            embed_fn=lambda x: self._embed(model, x), inner=inner,
            seq_len=seq_len, n_features=n_features)

        wrapper = WrapRegressor(_TorchRegressorAdapter(self._predict, model, seq_len, n_features))
        wrapper.calibrate(
            X=self._flatten(calib_x),
            y=calib_y.view(-1).detach().cpu().numpy().astype(np.float32),
            de=latent_de,
        )
        return wrapper

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #

    def _confidence_score(
            self,
            raw_preds: torch.Tensor,
            width: float,
            lower_bounds: torch.Tensor | None,
    ) -> float:
        """Confidence of one censored unit for one model (higher is better, in ``(0, 1]``).

        The score rewards a tight ``predict_int`` interval and penalises two physical
        inconsistencies of the model's own per-window predictions:

        * **Monotonicity** — a unit's windows are chronological, so its true RUL is non-increasing;
          any window predicting a *higher* RUL than the previous one is a violation.
        * **Lower bound** — a censored unit was observed to survive until its per-window survival
          lower bound, so a predicted RUL *below* that bound is a violation.

        Both violations are averaged (in RUL units, like the width) and added to an "effective
        width", so the score is::

            confidence = 1 / (1 + width + w_mono * mono_violation + w_lb * lb_violation)

        Args:
            raw_preds: The model's per-window predictions ``(T,)`` (chronological, oldest first).
            width: The ``predict_int`` interval width at the unit's last window (``>= 0``).
            lower_bounds: Optional per-window survival lower bounds ``(T,)``; when ``None`` the
                lower-bound term is ``0``.

        Returns:
            The confidence score in ``(0, 1]``.
        """
        preds = raw_preds.view(-1).float()

        if preds.numel() >= 2:
            increases = preds[1:] - preds[:-1]  # RUL should be non-increasing, so > 0 is a violation
            mono_violation = torch.clamp(increases, min=0.0).mean().item()
        else:
            mono_violation = 0.0

        if lower_bounds is not None:
            lb = lower_bounds.view(-1).float()
            lb_violation = torch.clamp(lb - preds, min=0.0).mean().item()
        else:
            lb_violation = 0.0

        effective_width = width + self.w_mono * mono_violation + self.w_lb * lb_violation
        return 1.0 / (1.0 + effective_width)

    def _score_pool(
            self,
            h: list[LightningModule],
            datasets: list[tuple[torch.Tensor, torch.Tensor]],
            unit_ids: list[int],
            unit_x: list[torch.Tensor],
            unit_lb: list[torch.Tensor] | None,
            calib_x: torch.Tensor,
            calib_y: torch.Tensor,
    ) -> dict[int, OrderedDict]:
        """Score every pooled censored unit with every model's conformal interval + confidence.

        For each model a calibrated conformal regressor is built once (``DifficultyEstimator``
        fitted on the model's own accumulated data, calibrated on ``(calib_x, calib_y)``), then
        every pooled unit's last window gets a ``predict_int`` interval and a confidence score. The
        difficulty that normalises each interval's width is measured in the raw feature space or in
        the model's latent space depending on ``self.difficulty_space``.

        Args:
            h: The current models.
            datasets: Per-model accumulated ``(x, y)`` (the regressor's ``DifficultyEstimator`` set).
            unit_ids: Pooled unit ids (ints).
            unit_x: Per-unit window sequences, aligned with ``unit_ids``.
            unit_lb: Optional per-unit survival lower bounds, aligned with ``unit_ids``.
            calib_x: Calibration features for ``crepes``.
            calib_y: Calibration labels for ``crepes``.

        Returns:
            ``{model_index: OrderedDict{unit_id: entry}}`` sorted by confidence (best first), where
            each ``entry`` is ``{"raw", "lower", "upper", "width", "confidence"}``.
        """
        all_preds: dict[int, OrderedDict] = {}
        for j, model in enumerate(h):
            if self.difficulty_space == "latent":
                wrapper = self._build_latent_calibrated_regressor(
                    model, datasets[j][0], calib_x, calib_y)
            else:
                wrapper = self._build_calibrated_regressor(model, datasets[j][0], calib_x, calib_y)

            raw_by_unit = {uid: self._predict(model, xu).detach().cpu().view(-1)
                           for uid, xu in zip(unit_ids, unit_x)}
            last_windows = torch.stack([xu[-1] for xu in unit_x], dim=0)
            # The wrapper's difficulty estimator (raw or latent) re-derives each unit's width.
            intervals = wrapper.predict_int(self._flatten(last_windows), confidence=self.confidence)

            entries: list[tuple[int, dict]] = []
            for idx, uid in enumerate(unit_ids):
                lower, upper = float(intervals[idx, 0]), float(intervals[idx, 1])
                width = upper - lower
                lb = unit_lb[idx] if unit_lb is not None else None
                confidence = self._confidence_score(raw_by_unit[uid], width, lb)
                entries.append((uid, {
                    "raw": raw_by_unit[uid],
                    "lower": lower,
                    "upper": upper,
                    "width": width,
                    "confidence": confidence,
                }))

            entries.sort(key=lambda e: e[1]["confidence"], reverse=True)
            all_preds[j] = OrderedDict(entries)
        return all_preds

    def _print_confidence_ranking(self, all_preds: dict[int, OrderedDict]) -> None:
        """Log one line per model: its censored units ranked by confidence (best first)."""
        for j, preds in all_preds.items():
            ranked = sorted(preds.items(), key=lambda kv: kv[1]["confidence"], reverse=True)
            listing = " ".join(f"uid={uid}:{entry['confidence']:.4f}" for uid, entry in ranked)
            self._log(1, f"[CoTraining]   Model {j} censored-unit confidence (best->worst): {listing}")

    # ------------------------------------------------------------------ #
    # Owner-based selection + pseudo-labelling
    # ------------------------------------------------------------------ #

    def _select_owner_units(
            self,
            all_preds: dict[int, OrderedDict],
            pool_uids: list[int],
            add_ratio: float,
    ) -> list[tuple[int, list[int], list[int]]]:
        """Per-model top-confidence selection with set-membership ownership.

        Each model selects its own top ``k = max(1, round(add_ratio * pool))`` units by confidence
        (only among units at or above ``confidence_threshold`` when it is set). A unit's *owners*
        are the models that selected it; a unit selected by nobody, or by everybody, is skipped.
        The *receivers* (models that did not select it) are taught its label.

        Args:
            all_preds: ``{model_index: OrderedDict{unit_id: entry}}`` from :meth:`_score_pool`.
            pool_uids: The pooled unit ids available this iteration.
            add_ratio: Fraction in ``(0, 1]`` each model selects from the pool.

        Returns:
            ``[(unit_id, owners, receivers), ...]``.
        """
        n_models = self.number_of_models
        k = max(1, round(add_ratio * len(pool_uids)))

        selected_sets: list[set[int]] = []
        for j in range(n_models):
            eligible = [
                uid for uid in pool_uids
                if self.confidence_threshold is None
                or all_preds[j][uid]["confidence"] >= self.confidence_threshold
            ]
            eligible.sort(key=lambda uid: all_preds[j][uid]["confidence"], reverse=True)
            selected_sets.append(set(eligible[:k]))

        selected: list[tuple[int, list[int], list[int]]] = []
        for uid in pool_uids:
            owners = [j for j in range(n_models) if uid in selected_sets[j]]
            if not owners or len(owners) >= n_models:
                # Selected by nobody (ignore) or by everybody (no asymmetry to teach) -> skip.
                continue
            receivers = [j for j in range(n_models) if j not in owners]
            selected.append((uid, owners, receivers))
        return selected

    @staticmethod
    def _estimate_label_last(
            uid: int,
            owners: list[int],
            all_preds: dict[int, OrderedDict],
            best_val_rmse: list[float],
    ) -> float:
        """Estimate a unit's last-window pseudo-RUL as the owners' own prediction.

        Each owner contributes its own last-window prediction of the unit. With several owners the
        predictions are averaged with weights ``1 / val_rmse`` so a more accurate model counts
        more.

        Args:
            uid: The censored unit id.
            owners: The owner model indices.
            all_preds: The scoring dict from :meth:`_score_pool`.
            best_val_rmse: Per-model current best validation RMSE (the weighting basis).

        Returns:
            The weighted last-window pseudo-RUL.
        """
        eps = 1e-8
        weights = [1.0 / (best_val_rmse[j] + eps) for j in owners]
        last_preds = [float(all_preds[j][uid]["raw"].view(-1)[-1]) for j in owners]
        return sum(w * p for w, p in zip(weights, last_preds)) / sum(weights)

    @staticmethod
    def _backward_extrapolate(
            label_last: float,
            unit_time_steps: torch.Tensor | None,
            num_windows: int,
    ) -> torch.Tensor:
        """Fill a unit's per-window RUL from its last-window value.

        ``RUL_i = label_last + (t_last - t_i)``. With ``unit_time_steps`` given (per-window,
        chronological) this uses the real elapsed time; otherwise unit-spaced window indices are
        used (``t_i = i``). Non-increasing by construction; the last window equals ``label_last``.

        Args:
            label_last: The estimated RUL of the last window.
            unit_time_steps: Optional per-window time steps ``(T,)`` (chronological).
            num_windows: Number of windows ``T``.

        Returns:
            Per-window RUL ``(T,)`` (CPU float tensor).
        """
        if unit_time_steps is None:
            t = torch.arange(num_windows, dtype=torch.float32)
        else:
            t = unit_time_steps.detach().cpu().reshape(-1).float()
        return label_last + (t[-1] - t)

    @staticmethod
    def _build_unit_pool_tensors(
            pool_ids: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            suspension_lower_bounds: torch.Tensor | None,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor] | None]:
        """Build the per-unit window sequences (and optional lower bounds) for one iteration's pool.

        The sequences and bounds are identical across models, so they are built once and reused.

        Args:
            pool_ids: The sampled unit ids for this iteration.
            suspension_data: All censored windows.
            suspension_ids: Per-window unit id, row-aligned with ``suspension_data``.
            suspension_lower_bounds: Optional per-window survival lower bounds, row-aligned.

        Returns:
            ``(unit_ids_int, unit_x, unit_lb)`` — ints, per-unit windows, per-unit bounds (or None).
        """
        unit_ids_int = [int(uid.item()) for uid in pool_ids]
        unit_x = [suspension_data[suspension_ids == uid].detach().cpu() for uid in pool_ids]
        unit_lb = (
            [suspension_lower_bounds[suspension_ids == uid].detach().cpu() for uid in pool_ids]
            if suspension_lower_bounds is not None else None
        )
        return unit_ids_int, unit_x, unit_lb

    # ------------------------------------------------------------------ #
    # Fine-tuning (single GPU, inline)
    # ------------------------------------------------------------------ #

    def _fine_tune(
            self,
            model_index: int,
            current_state_dict: dict[str, torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> LightningModule:
        """Warm-start fine-tune one model on ``(x, y)`` (all params, reduced LR) and rebuild it.

        Runs in this process on the single resolved device. Requires the builder path
        (:meth:`setup_training_builder`).

        Args:
            model_index: Index of the model to fine-tune.
            current_state_dict: CPU ``state_dict`` warm start (the model's current weights).
            x: Full accumulated training features.
            y: Full accumulated training targets.
            val_cpu: CPU ``(val_x, val_y)`` for early stopping / best-checkpoint selection.

        Returns:
            The fine-tuned model rebuilt from the returned state dict.
        """
        if not self._use_builders:
            raise NotImplementedError(
                "CoTrainingEnsemble_v3 fine-tuning requires setup_training_builder (builder path).")
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

    @staticmethod
    def _cpu_state_dict(module: LightningModule) -> dict[str, torch.Tensor]:
        """Detach-and-clone a module's ``state_dict`` to CPU (a picklable warm-start snapshot)."""
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _evaluate_on_val(
            self,
            model: LightningModule,
            val_x: torch.Tensor,
            val_y_flat: torch.Tensor,
            score_callback: Callable[[torch.Tensor, torch.Tensor], float] | None,
    ) -> tuple[float, float]:
        """Return ``(acceptance_value, val_rmse)`` of ``model`` on the validation set.

        Both are computed from a single forward pass. ``val_rmse`` is always returned (it drives
        the label-weighting ``1 / val_rmse``); the acceptance value is the RMSE when
        ``acceptance_metric == "val_rmse"`` and ``score_callback(pred, target)`` (e.g. the Scania
        score) when ``acceptance_metric == "score"``. Both are minimised (lower = better).

        Args:
            model: The model to evaluate.
            val_x: Validation features.
            val_y_flat: Validation targets, flattened to ``(N,)`` and float.
            score_callback: Score used when ``acceptance_metric == "score"`` (required in that mode).

        Returns:
            ``(acceptance_value, val_rmse)``.
        """
        pred = self._predict(model, val_x).view(-1).to(val_y_flat.device)
        val_rmse = (((val_y_flat - pred) ** 2).mean().item()) ** 0.5
        if self.acceptance_metric == "score":
            return float(score_callback(pred, val_y_flat)), val_rmse
        return val_rmse, val_rmse

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

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
            suspension_time_steps: torch.Tensor | None = None,
            suspension_lower_bounds: torch.Tensor | None = None,
            test_data: torch.Tensor | None = None,
            test_label: torch.Tensor | None = None,
            score_callback: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
            weight_callback: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
            weight_mode: str = "min",
            metrics_file: str | None = None,
            log_file: str | None = None,
    ) -> None:
        """Train the v3 co-training ensemble (sequential, single GPU).

        Each iteration samples a pool of censored units, scores every pooled unit with each
        model's conformal interval + confidence, selects units by owner-based confidence, labels
        each selected unit with its owners' prediction (backward-extrapolated), fine-tunes each
        receiver on its accumulated data, and keeps the fine-tuned model only if its validation
        RMSE improved (otherwise reverts and drops that iteration's units).

        Args:
            train_with_censored_data: Kept for signature compatibility (v3 always co-trains).
            failure_data: Features of the uncensored (failure) data.
            failure_label: Targets of the uncensored (failure) data.
            suspension_data: The censored windows.
            suspension_ids: Per-window unit id, row-aligned with ``suspension_data``.
            iterations: Number of co-training iterations.
            suspension_pool_size: Fraction in ``(0, 1]`` of censored units sampled as the pool each
                iteration (re-sampled every iteration; ``>= 1`` uses the whole remaining set).
            add_ratio: Per-model top fraction in ``(0, 1]`` each model selects from the pool.
            val_data: Validation features. Required (early stopping, ensemble weights, and the
                fallback calibration set when ``calib_data`` is not given).
            val_label: Validation labels. Required.
            calib_data: Optional calibration features kept separate from ``val_data``; when given
                the conformal regressors are calibrated on it instead of the validation set.
            calib_label: Calibration labels, required together with ``calib_data``.
            suspension_time_steps: Optional per-window ``time_step`` for the censored data,
                row-aligned with ``suspension_data`` (shape ``(N,)`` or ``(N, 1)``). Used to
                backward-extrapolate each unit's last-window RUL. ``None`` uses window indices.
            suspension_lower_bounds: Optional per-window survival lower bounds, row-aligned with
                ``suspension_data``; used by the confidence score's lower-bound term.
            test_data: Optional test features used only for the per-stage metrics.
            test_label: Optional test labels associated with ``test_data``.
            score_callback: Score reported in the metrics file (e.g. the Scania score). Required
                when metrics logging is enabled.
            weight_callback: Score used to compute the reported ensemble weights (e.g. RMSE).
                Required when metrics logging is enabled.
            weight_mode: "min" or "max" passed to the ensemble weighting (defaults to "min").
            metrics_file: Optional ``.csv`` destination for per-stage metrics.
            log_file: Optional ``.txt`` destination that captures every log message.
        """
        self._log_file_path = log_file
        self._check_if_training_is_possible()

        if val_data is None or val_label is None:
            raise ValueError(
                "val_data and val_label are required in v3 (early stopping, ensemble weights, and "
                "the fallback conformal calibration set).")
        if (calib_data is None) != (calib_label is None):
            raise ValueError("calib_data and calib_label must be provided together.")
        if not (0 < suspension_pool_size <= 1):
            raise ValueError("suspension_pool_size must be a fraction in (0, 1].")
        if not (0 < add_ratio <= 1):
            raise ValueError("add_ratio must be a fraction in (0, 1].")

        metrics_enabled = test_data is not None
        if metrics_enabled:
            if test_label is None:
                raise ValueError("test_label must be provided together with test_data.")
            if score_callback is None or weight_callback is None or metrics_file is None:
                raise ValueError(
                    "score_callback, weight_callback and metrics_file are required to log metrics.")
        if self.acceptance_metric == "score" and score_callback is None:
            raise ValueError(
                "acceptance_metric='score' requires score_callback (the metric to minimise).")

        if self._parallel:
            self._log(1, "[CoTraining] Multiple GPUs were configured, but v3 is single-GPU; "
                         f"using the first device only ({self._inline_devices}).")

        # Calibration set for crepes: the dedicated set when given, else the validation set.
        calib_x = calib_data if calib_data is not None else val_data
        calib_y = calib_label if calib_label is not None else val_label
        val_cpu = self._cpu_pair(val_data, val_label)

        total_units = len(torch.unique(suspension_ids))
        pool_size = total_units if suspension_pool_size >= 1.0 else max(
            1, round(suspension_pool_size * total_units))

        self._log(1, f"[CoTraining] Starting v3 training | models: {self.number_of_models} | "
                     f"failure samples: {len(failure_data)} | censored units: {total_units} | "
                     f"max iterations: {iterations} | confidence: {self.confidence} | "
                     f"pool fraction: {suspension_pool_size} (size: {pool_size}) | "
                     f"per-model select ratio: {add_ratio} | "
                     f"confidence threshold: {self.confidence_threshold} | "
                     f"difficulty space: {self.difficulty_space} | "
                     f"keep best model: {self.keep_best_model} | "
                     f"acceptance metric: {self.acceptance_metric} | "
                     f"calibration set: {'dedicated' if calib_data is not None else 'validation (fallback)'}")

        # Per-model state: current best weights, its acceptance value + validation RMSE (RMSE is
        # kept separately because the label weighting always uses 1 / val_rmse), and its dataset.
        h: list[LightningModule] = []
        best_accept: list[float] = []
        best_val_rmse: list[float] = []
        best_dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
        for j in range(self.number_of_models):
            self._log(1, f"[CoTraining] Initial training of model {j} on {len(failure_data)} "
                         f"failure samples...")
            model = self._fit_from_scratch(j, failure_data, failure_label, val_data, val_label)
            h.append(model)
            accept_value, val_rmse = self._evaluate_on_val(model, val_data, val_label, score_callback)
            best_accept.append(accept_value)
            best_val_rmse.append(val_rmse)
            best_dataset.append((failure_data, failure_label))
            self._log(1, f"[CoTraining]   Model {j} initial val RMSE: {val_rmse:.4f} | "
                         f"acceptance ({self.acceptance_metric}): {accept_value:.4f}")
        self._log(1, "[CoTraining] Initial training done.")

        if metrics_enabled:
            self._log_stage_metrics(
                stage="initial", h=h, models_datasets=best_dataset, test_data=test_data,
                test_label=test_label, val_data=val_data, val_label=val_label,
                score_callback=score_callback, weight_callback=weight_callback,
                weight_mode=weight_mode, metrics_file=metrics_file)

        remaining_ids = torch.unique(suspension_ids)

        for i in range(iterations):
            if len(remaining_ids) == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i}: no remaining censored units.")
                break

            pool_size_iter = min(pool_size, len(remaining_ids))
            shuffled = remaining_ids[torch.randperm(len(remaining_ids))]
            pool_ids = shuffled[:pool_size_iter]
            self._log(1, f"[CoTraining] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            unit_ids, unit_x, unit_lb = self._build_unit_pool_tensors(
                pool_ids, suspension_data, suspension_ids, suspension_lower_bounds)
            xu_by_unit = dict(zip(unit_ids, unit_x))

            # Score + rank every pooled unit for every model, then log the per-model ranking.
            all_preds = self._score_pool(h, best_dataset, unit_ids, unit_x, unit_lb, calib_x, calib_y)
            self._print_confidence_ranking(all_preds)

            selected = self._select_owner_units(all_preds, unit_ids, add_ratio)
            if not selected:
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: no unit selected.")
                break

            # Build each receiver's new pseudo-labelled units, and permanently retire the selected
            # units from the pool (a later rejection drops them for good).
            selected_per_model: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
                [] for _ in range(self.number_of_models)]
            for uid, owners, receivers in selected:
                label_last = self._estimate_label_last(uid, owners, all_preds, best_val_rmse)
                xu = xu_by_unit[uid]
                unit_ts = (suspension_time_steps[suspension_ids == uid]
                           if suspension_time_steps is not None else None)
                lu = self._backward_extrapolate(label_last, unit_ts, xu.shape[0])
                for r in receivers:
                    selected_per_model[r].append((torch.tensor(uid), xu, lu))
                remaining_ids = remaining_ids[remaining_ids != uid]
                self._log(1, f"[CoTraining]   Unit {uid}: owners={owners} receivers={receivers} | "
                             f"last-window RUL={label_last:.4f}")

            # Fine-tune each receiver; keep it only if its validation RMSE improved (unless
            # keep_best_model is off, in which case every fine-tuned model is kept).
            for j in range(self.number_of_models):
                if not selected_per_model[j]:
                    continue
                xj, yj = best_dataset[j]
                new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)
                candidate_x = torch.cat([xj, new_xu], dim=0)
                candidate_y = torch.cat([yj, new_lu], dim=0)
                n_added = len(new_xu)
                self._log(1, f"[CoTraining]   Fine-tuning model {j} | added {len(selected_per_model[j])} "
                             f"unit(s) ({n_added} sample(s)) | dataset: {len(candidate_x)} samples")

                candidate = self._fine_tune(j, self._cpu_state_dict(h[j]), candidate_x, candidate_y, val_cpu)
                candidate_accept, candidate_rmse = self._evaluate_on_val(
                    candidate, val_data, val_label, score_callback)
                metric = self.acceptance_metric

                if not self.keep_best_model:
                    # Retention off: always accept the fine-tuned model and its added data (the
                    # baseline this run is meant to compare against).
                    self._log(1, f"[CoTraining]   Model {j}: kept (retention off | {metric} "
                                 f"{candidate_accept:.4f}, previous {best_accept[j]:.4f}).")
                    accept = True
                elif candidate_accept < best_accept[j]:
                    self._log(1, f"[CoTraining]   Model {j}: kept ({metric} {candidate_accept:.4f} "
                                 f"< best {best_accept[j]:.4f}).")
                    accept = True
                else:
                    self._log(1, f"[CoTraining]   Model {j}: iteration {i + 1} rejected ({metric} "
                                 f"{candidate_accept:.4f} >= best {best_accept[j]:.4f}); reverted "
                                 f"and dropped {n_added} censored sample(s).")
                    accept = False

                if accept:
                    h[j] = candidate
                    best_accept[j] = candidate_accept
                    best_val_rmse[j] = candidate_rmse
                    best_dataset[j] = (candidate_x, candidate_y)

            if metrics_enabled:
                self._log_stage_metrics(
                    stage=f"iteration_{i + 1}", h=h, models_datasets=best_dataset,
                    test_data=test_data, test_label=test_label, val_data=val_data,
                    val_label=val_label, score_callback=score_callback,
                    weight_callback=weight_callback, weight_mode=weight_mode, metrics_file=metrics_file)

        if metrics_enabled:
            self._log_stage_metrics(
                stage="final", h=h, models_datasets=best_dataset, test_data=test_data,
                test_label=test_label, val_data=val_data, val_label=val_label,
                score_callback=score_callback, weight_callback=weight_callback,
                weight_mode=weight_mode, metrics_file=metrics_file)

        self._log(1, "[CoTraining] Training complete.")
        self.lightning_modules = h
