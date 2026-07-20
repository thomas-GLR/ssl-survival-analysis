import gc
from collections import OrderedDict
from typing import Callable

import torch
from lightning import LightningModule

from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2
from models.cotraining_gpu_pool import (
    CoTrainingGpuPool,
    CpsScoreSpec,
    FineTuneSpec,
    run_cps_score_job,
    run_finetune_job,
)


class CoTrainingEnsemble_v3(CoTrainingEnsemble_v2):
    """Owner-based co-training ensemble with a latent-kNN pseudo-label estimator (v3).

    v3 keeps v2's confidence scoring (per-model conformal spread of every censored unit,
    normalized per model) but changes **who owns a unit, how its pseudo-label is computed, and
    how the models are updated**:

    * **Conformal predictive system** instead of a symmetric interval: each model is calibrated
      with ``crepes`` ``cps=True`` so a unit's last window yields asymmetric percentiles
      ``a = p_low``, ``c = p50`` (median), ``b = p_high``. ``width = b - a`` drives selection;
      ``a, b, c`` bound / seed the label estimator.
    * **Owner-based selection**: for each unit the most-confident model(s) (within
      ``confidence_tol`` of the best normalized score) *own* it and decide its label; the label
      is injected into the *other* models (co-training). If every model is equally confident the
      unit carries no information asymmetry and is skipped. Two knobs bound how many units are
      taken: ``add_ratio`` (target = top fraction of the pool) and ``best_ratio`` (hard
      eligibility cap — only the top-confidence fraction is ever eligible).
    * **Closed-form label estimator**: the owner's pseudo-label for the unit's last window is the
      distance-weighted k-NN of the labelled set's RULs in the model's latent space, clipped to
      the conformal band ``[a, b]`` (optionally blended with the model's median ``c`` via
      ``model_pred_blend``). Earlier windows are filled by backward extrapolation
      ``RUL_i = RUL_last + (t_last - t_i)`` using the per-window ``time_step`` (monotone by
      construction).
    * **Fine-tuning** (warm-start from current weights, reduced learning rate) on each model's
      full accumulated ``failure + pseudo`` set, instead of retraining from scratch.

    Everything not listed here (setup, per-stage metrics, ensemble weighting, prediction, the
    normalized-width / selection-score helpers) is inherited unchanged from
    :class:`CoTrainingEnsemble_v2`.
    """

    def __init__(
            self,
            models: list,
            weights: list[float] | None = None,
            verbose: int = 0,
            confidence: float = 0.9,
            inference_batch_size: int | None = None,
            use_monotone_projection: bool = False,
            monotone_residual_weight: float = 1.0,
            confidence_tol: float = 0.01,
            best_ratio: float = 0.2,
            n_neighbors: int = 10,
            model_pred_blend: float = 0.0,
            fine_tune_lr_factor: float = 0.1,
            fine_tune_max_epochs: int = 20,
            fine_tune_patience: int = 5,
    ):
        """
        :param models: The models used in the co-training ensemble.
        :param weights: Optional pre-defined per-model ensemble weights.
        :param verbose: Verbosity level (0 silent, 1 key decisions, 2 full per-candidate detail).
        :param confidence: Confidence level in ``(0, 1)`` defining the conformal percentile band:
            ``a`` is the ``100*(1-confidence)/2`` percentile, ``b`` the ``100*(1+confidence)/2``
            percentile, ``c`` the median. Defaults to ``0.9`` (a 5/50/95 band).
        :param inference_batch_size: If set, forward passes (prediction and latent embedding) run
            in chunks of this many samples to cap peak memory. ``None`` keeps single-shot inference.
        :param use_monotone_projection: When ``True``, the monotone-projection residual of each
            unit's raw per-window predictions is blended into the selection score (a
            self-consistency signal). It does **not** change the injected label (which comes from
            the k-NN estimator + backward extrapolation). ``False`` keeps width-only selection.
        :param monotone_residual_weight: Weight of the residual term in the selection score (only
            used when ``use_monotone_projection`` is ``True``).
        :param confidence_tol: Tolerance on the normalized selection score for co-ownership: model
            ``j`` co-owns a unit if ``score_j <= best_score + confidence_tol``.
        :param best_ratio: Eligibility cap in ``(0, 1]`` — only the top ``best_ratio`` fraction of
            the pool (by best-model confidence) is ever eligible to be pseudo-labelled.
        :param n_neighbors: Number of nearest labelled neighbours (in latent space) used by the
            k-NN label estimator.
        :param model_pred_blend: ``alpha`` in ``[0, 1]`` blending the model's conformal median
            ``c`` with the k-NN estimate: ``label = alpha*c + (1-alpha)*RUL_kNN`` (then clipped to
            ``[a, b]``). ``0`` (default) = pure k-NN clamped to the conformal band.
        :param fine_tune_lr_factor: Multiplier applied to each model's learning rate during
            fine-tuning (warm start), e.g. ``0.1``.
        :param fine_tune_max_epochs: Max epochs per fine-tuning call.
        :param fine_tune_patience: ``EarlyStopping`` patience per fine-tuning call.
        """
        super().__init__(
            models=models,
            weights=weights,
            verbose=verbose,
            confidence=confidence,
            inference_batch_size=inference_batch_size,
            use_monotone_projection=use_monotone_projection,
            monotone_residual_weight=monotone_residual_weight,
        )

        if not (0 < best_ratio <= 1):
            raise ValueError("best_ratio must be a fraction in (0, 1].")
        if confidence_tol < 0:
            raise ValueError("confidence_tol must be non-negative.")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if not (0 <= model_pred_blend <= 1):
            raise ValueError("model_pred_blend must be in [0, 1].")
        if fine_tune_lr_factor <= 0:
            raise ValueError("fine_tune_lr_factor must be positive.")

        self.confidence_tol = confidence_tol
        self.best_ratio = best_ratio
        self.n_neighbors = n_neighbors
        self.model_pred_blend = model_pred_blend
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.fine_tune_max_epochs = fine_tune_max_epochs
        self.fine_tune_patience = fine_tune_patience

    # ------------------------------------------------------------------ #
    # Small config helpers
    # ------------------------------------------------------------------ #

    def _percentile_band(self) -> list[float]:
        """Return the ``[low, 50, high]`` percentiles for the current ``confidence`` band."""
        low = 100.0 * (1.0 - self.confidence) / 2.0
        high = 100.0 * (1.0 + self.confidence) / 2.0
        return [low, 50.0, high]

    # ------------------------------------------------------------------ #
    # Latent embedding extraction (for the k-NN label estimator)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _head_module(module: LightningModule) -> torch.nn.Module:
        """Return the regression-head submodule whose input is the model's latent vector.

        All Scania architectures feed their pre-head latent straight into a final head named
        ``regressor`` (CNN1D) or ``linear`` (LSTM / transformer variants), wrapped by a
        ``BasicLightningModule`` as ``.net``. A forward-pre-hook on that head captures the latent.
        """
        net = getattr(module, "net", module)
        head = getattr(net, "regressor", None)
        if head is None:
            head = getattr(net, "linear", None)
        if head is None:
            raise ValueError(
                "Cannot locate the regression head (expected 'regressor' or 'linear') on the "
                "model for latent embedding extraction.")
        return head

    def _embed(self, module: LightningModule, x: torch.Tensor) -> torch.Tensor:
        """Return the latent embedding of ``x`` (the input to the model's regression head).

        Runs a forward pass under ``eval`` / ``no_grad`` with a temporary forward-pre-hook on the
        head submodule that captures its input. The hook is always removed afterwards. Forward
        passes are chunked by ``inference_batch_size`` to cap peak memory.

        :param module: The trained model to embed with.
        :param x: Input features ``(N, seq_len, n_features)``.
        :return: Latent embeddings ``(N, D)`` on the model's device.
        """
        module.eval()
        device = next(module.parameters()).device
        captured: dict[str, torch.Tensor] = {}

        def _hook(_mod: torch.nn.Module, inputs: tuple) -> None:
            captured["z"] = inputs[0].detach()

        handle = self._head_module(module).register_forward_pre_hook(_hook)
        try:
            with torch.no_grad():
                if self._inference_batch_size is None:
                    module(x.to(device))
                    return captured["z"]
                chunks: list[torch.Tensor] = []
                for start in range(0, len(x), self._inference_batch_size):
                    module(x[start:start + self._inference_batch_size].to(device))
                    chunks.append(captured["z"])
                return torch.cat(chunks, dim=0)
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    # Phase 1 helpers — CPS scoring pool assembly
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_unit_pool_tensors(
            pool_ids: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            suspension_lower_bounds: torch.Tensor | None,
            clip_bounds: bool,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor] | None]:
        """Build the per-unit sequences (and optional lower bounds) for one iteration's pool.

        ``xu`` (and its lower bounds) are identical across models for a given unit, so they are
        built once and reused for every model's CPS-scoring job.
        """
        unit_ids_int = [int(uid.item()) for uid in pool_ids]
        unit_x = [suspension_data[suspension_ids == uid].detach().cpu() for uid in pool_ids]
        unit_lb = (
            [suspension_lower_bounds[suspension_ids == uid].detach().cpu() for uid in pool_ids]
            if clip_bounds else None
        )
        return unit_ids_int, unit_x, unit_lb

    def _make_cps_spec(
            self,
            model_index: int,
            state_dict: dict[str, torch.Tensor],
            train_x: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
            unit_ids_int: list[int],
            unit_x: list[torch.Tensor],
            unit_lb: list[torch.Tensor] | None,
    ) -> CpsScoreSpec:
        """Build a :class:`CpsScoreSpec` for one model's pooled-unit conformal-predictive scoring."""
        return CpsScoreSpec(
            module_builder=self.module_builders[model_index],
            state_dict=state_dict,
            train_x=train_x.detach().cpu(),
            val_x=val_cpu[0],
            val_y=val_cpu[1],
            unit_ids=unit_ids_int,
            unit_x=unit_x,
            percentiles=self._percentile_band(),
            use_monotone_projection=self.use_monotone_projection,
            unit_lower_bounds=unit_lb,
            accelerator="gpu",
            devices=1,
        )

    @staticmethod
    def _assemble_all_preds(
            units: list[tuple],
            xu_by_unit: dict[int, torch.Tensor],
    ) -> OrderedDict:
        """Turn a CPS job's per-unit results into a width-sorted OrderedDict for one model.

        ``units`` entries are ``(unit_id, raw_preds, a, b, width, residual, c)`` (as returned by
        :func:`run_cps_score_job`). The stored tuple is
        ``(unit_id_tensor, xu, raw_preds, a, b, width, residual, c)`` — ``width`` at index 5 and
        ``residual`` at index 6 so the inherited ``_normalized_widths`` / ``_selection_scores``
        apply unchanged; ``c`` (median) at index 7.
        """
        candidates = [
            (torch.tensor(uid), xu_by_unit[uid], raw, a, b, width, residual, c)
            for uid, raw, a, b, width, residual, c in units
        ]
        candidates.sort(key=lambda e: e[5])
        return OrderedDict((e[0].item(), e) for e in candidates)

    # ------------------------------------------------------------------ #
    # Phase 2 helpers — owner-based selection + label estimation
    # ------------------------------------------------------------------ #

    def _select_owner_units(
            self,
            scores: dict[int, dict[int, float]],
            pool_uids: list[int],
            n_target: int,
    ) -> list[tuple[int, list[int], list[int]]]:
        """Owner-based selection over one iteration's pool.

        Units are ranked by their best (min-over-models) selection score; only the top
        ``best_ratio`` fraction is eligible. Walking the eligible units in confidence order, a
        unit's owners are the models within ``confidence_tol`` of its best score; if *every* model
        is an owner the unit is skipped (no information asymmetry). Selection stops once
        ``n_target`` units are assigned or the eligible set is exhausted.

        :param scores: ``{model_index: {unit_id_int: score}}`` (smaller = more confident).
        :param pool_uids: The pooled unit ids (ints) available this iteration.
        :param n_target: Target number of units to assign (``round(add_ratio * pool)``).
        :return: ``[(unit_id_int, owners, receivers), ...]``.
        """
        n_models = self.number_of_models

        def best_score(uid: int) -> float:
            vals = [scores[j][uid] for j in range(n_models) if uid in scores[j]]
            return min(vals) if vals else float("inf")

        ranked = sorted(pool_uids, key=best_score)
        n_eligible = max(1, round(self.best_ratio * len(pool_uids)))
        eligible = ranked[:n_eligible]

        selected: list[tuple[int, list[int], list[int]]] = []
        for uid in eligible:
            if len(selected) >= n_target:
                break
            per_model = {j: scores[j][uid] for j in range(n_models) if uid in scores[j]}
            if not per_model:
                continue
            best = min(per_model.values())
            owners = [j for j, s in per_model.items() if s <= best + self.confidence_tol]
            if len(owners) >= n_models:
                # Every model is (equally) confident -> nothing to teach, skip.
                continue
            receivers = [j for j in range(n_models) if j not in owners]
            selected.append((uid, owners, receivers))
        return selected

    def _compute_label_caches(
            self,
            h: list[LightningModule],
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]],
            owner_set: set[int],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Embed each owner model's accumulated labelled set once for this iteration's k-NN.

        :return: ``{model_index: (Z_label (N, D), Y_label (N,))}`` on that model's device.
        """
        cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for j in owner_set:
            x_j, y_j = models_datasets[j]
            z_label = self._embed(h[j], x_j)
            y_label = y_j.view(-1).to(z_label.device)
            cache[j] = (z_label, y_label)
        return cache

    def _estimate_label_last(
            self,
            uid: int,
            owners: list[int],
            all_preds: dict[int, OrderedDict],
            scores: dict[int, dict[int, float]],
            z_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
            h: list[LightningModule],
    ) -> float:
        """Estimate the pseudo-RUL of a unit's **last window** (owner-consensus, closed form).

        For each owner: embed the unit's last window, take the distance-weighted mean RUL of its
        ``n_neighbors`` nearest labelled neighbours in that model's latent space, optionally blend
        with the model's conformal median ``c`` (``model_pred_blend``), and clip to the conformal
        band ``[a, b]``. Multiple owners are combined by a confidence-weighted average
        (weight ``1/score``).
        """
        labels: list[float] = []
        weights: list[float] = []
        for j in owners:
            _, xu, _, a, b, _, _, c = all_preds[j][uid]
            z_u = self._embed(h[j], xu[-1:])  # (1, D)
            z_label, y_label = z_cache[j]
            k = min(self.n_neighbors, z_label.shape[0])
            dist = torch.cdist(z_u, z_label).view(-1)
            knn_dist, knn_idx = torch.topk(dist, k, largest=False)
            w = 1.0 / (knn_dist + 1e-8)
            rul_knn = float((w * y_label[knn_idx]).sum() / w.sum())
            label_j = self.model_pred_blend * c + (1.0 - self.model_pred_blend) * rul_knn
            label_j = float(min(max(label_j, a), b))
            labels.append(label_j)
            weights.append(1.0 / max(scores[j][uid], 1e-8))
        total = sum(weights)
        return sum(wt * lab for wt, lab in zip(weights, labels)) / total

    @staticmethod
    def _backward_extrapolate(
            label_last: float,
            unit_time_steps: torch.Tensor | None,
            num_windows: int,
    ) -> torch.Tensor:
        """Fill a unit's per-window RUL from its last-window value.

        ``RUL_i = label_last + (t_last - t_i)``. With ``unit_time_steps`` given (per-window,
        chronological), this uses the real elapsed time; otherwise unit-spaced window indices are
        used (``t_i = i``). Non-increasing by construction; the last window equals ``label_last``.

        :return: Per-window RUL ``(num_windows,)`` (CPU float tensor).
        """
        if unit_time_steps is None:
            t = torch.arange(num_windows, dtype=torch.float32)
        else:
            t = unit_time_steps.detach().cpu().reshape(-1).float()
        return label_last + (t[-1] - t)

    def _select_and_label(
            self,
            all_preds: dict[int, OrderedDict],
            h: list[LightningModule],
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]],
            pool_ids: torch.Tensor,
            remaining_suspension_ids: torch.Tensor,
            suspension_ids: torch.Tensor,
            suspension_time_steps: torch.Tensor | None,
            add_ratio: float,
    ) -> tuple[list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], torch.Tensor, int]:
        """Phase 2 (path-independent): owner-based selection + closed-form pseudo-labelling.

        Returns ``(selected_per_model, remaining_suspension_ids, added)`` where
        ``selected_per_model[r]`` is the list of ``(unit_id, xu, per_window_label)`` assigned to
        receiver model ``r`` (ready for the inherited ``_concat_selected_units``).
        """
        n_models = self.number_of_models
        selected_per_model: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(n_models)
        ]

        scores = self._selection_scores(all_preds)
        pool_uids = [int(u) for u in pool_ids]
        n_target = max(1, round(add_ratio * len(pool_uids)))

        selected = self._select_owner_units(scores, pool_uids, n_target)
        if not selected:
            return selected_per_model, remaining_suspension_ids, 0

        owner_set: set[int] = set()
        for _, owners, _ in selected:
            owner_set.update(owners)
        z_cache = self._compute_label_caches(h, models_datasets, owner_set)

        self._log(1, f"[CoTraining]   Selecting up to {n_target} unit(s) "
                     f"(eligible = top {self.best_ratio} of {len(pool_uids)} pooled).")

        added = 0
        for uid, owners, receivers in selected:
            label_last = self._estimate_label_last(uid, owners, all_preds, scores, z_cache, h)

            uid_tensor, xu = all_preds[owners[0]][uid][0], all_preds[owners[0]][uid][1]
            unit_ts = (
                suspension_time_steps[suspension_ids == uid_tensor]
                if suspension_time_steps is not None else None
            )
            lu = self._backward_extrapolate(label_last, unit_ts, xu.shape[0])

            for r in receivers:
                selected_per_model[r].append((uid_tensor, xu, lu))
            remaining_suspension_ids = remaining_suspension_ids[remaining_suspension_ids != uid_tensor]
            added += 1

            self._log(1, f"[CoTraining]   Unit {uid}: owners={owners} receivers={receivers} | "
                         f"last-window RUL={label_last:.4f}")

        # Free the per-model latent caches (each holds an (N, D) embedding of the labelled set).
        del z_cache
        gc.collect()

        return selected_per_model, remaining_suspension_ids, added

    # ------------------------------------------------------------------ #
    # Fine-tuning (replaces v2's from-scratch retrain during iterations)
    # ------------------------------------------------------------------ #

    def _make_finetune_spec(
            self,
            model_index: int,
            current_state_dict: dict[str, torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> FineTuneSpec:
        """Build a :class:`FineTuneSpec` warm-starting from ``current_state_dict``.

        Fine-tunes all parameters (``trainable_param_names=None``) on the model's full accumulated
        ``(x, y)`` with a reduced learning rate, so the model learns the new pseudo-labels without
        forgetting its prior knowledge.
        """
        return FineTuneSpec(
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
            accelerator="gpu",
            devices=1,
        )

    def _fine_tune_inline(
            self,
            model_index: int,
            current_state_dict: dict[str, torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> LightningModule:
        """Fine-tune one model inline (this process) and rebuild it from the returned state."""
        if not self._use_builders:
            raise NotImplementedError(
                "CoTrainingEnsemble_v3 fine-tuning requires setup_training_builder (builder path).")
        spec = self._make_finetune_spec(model_index, current_state_dict, x, y, val_cpu)
        spec.accelerator = self._inline_accelerator
        spec.devices = self._inline_devices
        result = run_finetune_job(spec)
        return self._rebuild_module(model_index, result["state_dict"])

    @staticmethod
    def _cpu_state_dict(module: LightningModule) -> dict[str, torch.Tensor]:
        """Detach-and-clone a module's ``state_dict`` to CPU (picklable warm-start snapshot)."""
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

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
        """Train the v3 co-training ensemble.

        Each iteration: sample a random pool of censored units, score every pooled unit with each
        model's conformal predictive system, select units by owner-based confidence, estimate each
        selected unit's pseudo-label with the latent k-NN estimator + backward extrapolation,
        inject it into the receiver models, and fine-tune those receivers on their accumulated
        data.

        Args mirror :meth:`CoTrainingEnsemble_v2.train`, with two additions:

        :param suspension_time_steps: Optional per-window ``time_step`` for the censored data,
            row-aligned with ``suspension_data`` / ``suspension_ids`` (shape ``(N,)`` or
            ``(N, 1)``). Used to backward-extrapolate each unit's last-window RUL to its earlier
            windows. If ``None``, unit-spaced window indices are used instead.
        :param add_ratio: Target fraction in ``(0, 1]`` of the pool to assign per iteration.
        """
        self._log_file_path = log_file
        self._check_if_training_is_possible()

        if val_data is None or val_label is None:
            raise ValueError(
                "val_data and val_label are required in v3 (used to calibrate the conformal "
                "predictive systems).")

        metrics_enabled = test_data is not None
        if metrics_enabled:
            if test_label is None:
                raise ValueError("test_label must be provided together with test_data.")
            if score_callback is None:
                raise ValueError("score_callback is required to log per-stage metrics.")
            if weight_callback is None:
                raise ValueError("weight_callback is required to log per-stage metrics.")
            if metrics_file is None:
                raise ValueError("metrics_file is required to log per-stage metrics.")

        if not (0 < suspension_pool_size <= 1):
            raise ValueError("suspension_pool_size must be a fraction in (0, 1].")
        if not (0 < add_ratio <= 1):
            raise ValueError("add_ratio must be a fraction in (0, 1].")

        if self.use_monotone_projection and suspension_lower_bounds is None:
            self._log(1, "[CoTraining] use_monotone_projection is on but suspension_lower_bounds "
                         "was not provided; the residual uses monotonicity only (no censoring clip).")
        clip_bounds = self.use_monotone_projection and suspension_lower_bounds is not None

        total_suspension_units = len(torch.unique(suspension_ids))
        if suspension_pool_size >= 1.0:
            pool_size = total_suspension_units
        else:
            pool_size = max(1, round(suspension_pool_size * total_suspension_units))

        self._log(1, f"[CoTraining] Starting v3 training | models: {self.number_of_models} | "
                     f"failure samples: {len(failure_data)} | censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | confidence band: {self._percentile_band()} | "
                     f"pool fraction: {suspension_pool_size} (size: {pool_size}) | "
                     f"add ratio: {add_ratio} | best ratio: {self.best_ratio} | "
                     f"mode: {'parallel(' + str(self.gpu_ids) + ')' if self._parallel else 'sequential'}")

        if self._parallel:
            self._train_parallel(
                failure_data=failure_data,
                failure_label=failure_label,
                suspension_data=suspension_data,
                suspension_ids=suspension_ids,
                suspension_time_steps=suspension_time_steps,
                suspension_lower_bounds=suspension_lower_bounds if clip_bounds else None,
                iterations=iterations,
                pool_size=pool_size,
                add_ratio=add_ratio,
                val_data=val_data,
                val_label=val_label,
                metrics_enabled=metrics_enabled,
                test_data=test_data,
                test_label=test_label,
                score_callback=score_callback,
                weight_callback=weight_callback,
                weight_mode=weight_mode,
                metrics_file=metrics_file,
            )
            return

        val_cpu = self._cpu_pair(val_data, val_label)

        models_datasets: list[tuple[torch.Tensor, torch.Tensor]] = []
        h: list[LightningModule] = []
        for j in range(self.number_of_models):
            x_i, y_i = failure_data, failure_label
            models_datasets.append((x_i, y_i))
            self._log(1, f"[CoTraining] Initial training of model {j} on {len(x_i)} failure samples...")
            h.append(self._fit_from_scratch(j, x_i, y_i, val_data, val_label))
        self._log(1, "[CoTraining] Initial training done.")

        if metrics_enabled:
            self._log_stage_metrics(
                stage="initial", h=h, models_datasets=models_datasets, test_data=test_data,
                test_label=test_label, val_data=val_data, val_label=val_label,
                score_callback=score_callback, weight_callback=weight_callback,
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

            unit_ids_int, unit_x, unit_lb = self._build_unit_pool_tensors(
                pool_ids, suspension_data, suspension_ids, suspension_lower_bounds, clip_bounds)
            xu_by_unit = {uid: xu for uid, xu in zip(unit_ids_int, unit_x)}

            # Phase 1 — per-model CPS scoring (inline; run_cps_score_job rebuilds the model from
            # a CPU state dict, matching the parallel path exactly).
            all_preds: dict[int, OrderedDict] = {}
            for j in range(self.number_of_models):
                self._log(2, f"[CoTraining]   Model {j}: CPS-scoring {len(pool_ids)} pooled units...")
                spec = self._make_cps_spec(
                    j, self._cpu_state_dict(h[j]), models_datasets[j][0], val_cpu,
                    unit_ids_int, unit_x, unit_lb)
                spec.accelerator = self._inline_accelerator
                spec.devices = self._inline_devices
                result = run_cps_score_job(spec)
                all_preds[j] = self._assemble_all_preds(result["units"], xu_by_unit)

            # Phase 2 — owner-based selection + label estimation (main process).
            selected_per_model, remaining_suspension_ids, added = self._select_and_label(
                all_preds=all_preds, h=h, models_datasets=models_datasets, pool_ids=pool_ids,
                remaining_suspension_ids=remaining_suspension_ids, suspension_ids=suspension_ids,
                suspension_time_steps=suspension_time_steps, add_ratio=add_ratio)

            if added == 0:
                self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: no unit selected.")
                break

            # Phase 3 — fine-tune every receiver on its accumulated data.
            for j in range(self.number_of_models):
                if selected_per_model[j]:
                    xj, yj = models_datasets[j]
                    new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)
                    xj = torch.cat([xj, new_xu], dim=0)
                    yj = torch.cat([yj, new_lu], dim=0)
                    models_datasets[j] = (xj, yj)
                    self._log(1, f"[CoTraining]   Fine-tuning model {j} | "
                                 f"added {len(selected_per_model[j])} unit(s) | dataset: {len(xj)} samples")
                    h[j] = self._fine_tune_inline(j, self._cpu_state_dict(h[j]), xj, yj, val_cpu)

            if metrics_enabled:
                self._log_stage_metrics(
                    stage=f"iteration_{i + 1}", h=h, models_datasets=models_datasets,
                    test_data=test_data, test_label=test_label, val_data=val_data,
                    val_label=val_label, score_callback=score_callback,
                    weight_callback=weight_callback, weight_mode=weight_mode, metrics_file=metrics_file,
                )

            del all_preds, selected_per_model
            gc.collect()

        if metrics_enabled:
            self._log_stage_metrics(
                stage="final", h=h, models_datasets=models_datasets, test_data=test_data,
                test_label=test_label, val_data=val_data, val_label=val_label,
                score_callback=score_callback, weight_callback=weight_callback,
                weight_mode=weight_mode, metrics_file=metrics_file,
            )

        self._log(1, "[CoTraining] Training complete.")
        self.lightning_modules = h

    def _train_parallel(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            suspension_time_steps: torch.Tensor | None,
            suspension_lower_bounds: torch.Tensor | None,
            iterations: int,
            pool_size: int,
            add_ratio: float,
            val_data: torch.Tensor,
            val_label: torch.Tensor,
            metrics_enabled: bool,
            test_data: torch.Tensor | None,
            test_label: torch.Tensor | None,
            score_callback: Callable[[torch.Tensor, torch.Tensor], float] | None,
            weight_callback: Callable[[torch.Tensor, torch.Tensor], float] | None,
            weight_mode: str,
            metrics_file: str | None,
    ) -> None:
        """Multi-GPU parallel version of :meth:`train`.

        Initial from-scratch training, per-model CPS scoring, and per-model fine-tuning each run
        concurrently across all ``gpu_ids`` via a :class:`CoTrainingGpuPool`. Owner-based
        selection and the latent k-NN label estimation stay in the main process (they use the
        rebuilt models ``h``).
        """
        n = self.number_of_models
        val_cpu = self._cpu_pair(val_data, val_label)
        clip_bounds = self.use_monotone_projection and suspension_lower_bounds is not None

        pool = CoTrainingGpuPool(self.gpu_ids)
        pool.start()
        try:
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]] = [
                (failure_data, failure_label) for _ in range(n)
            ]

            self._log(1, f"[CoTraining] Initial parallel training of {n} models...")
            job_ids = {
                j: pool.submit_job(pool.round_robin_gpu(j),
                                   self._make_fit_spec(j, failure_data, failure_label, val_cpu))
                for j in range(n)
            }
            results = pool.gather(list(job_ids.values()))
            h = [self._rebuild_module(j, results[job_ids[j]]["state_dict"]) for j in range(n)]
            self._log(1, "[CoTraining] Initial training done.")

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="initial", h=h, models_datasets=models_datasets, test_data=test_data,
                    test_label=test_label, val_data=val_data, val_label=val_label,
                    score_callback=score_callback, weight_callback=weight_callback,
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

                unit_ids_int, unit_x, unit_lb = self._build_unit_pool_tensors(
                    pool_ids, suspension_data, suspension_ids, suspension_lower_bounds, clip_bounds)
                xu_by_unit = {uid: xu for uid, xu in zip(unit_ids_int, unit_x)}

                # Phase 1 — per-model CPS scoring in parallel.
                score_jobs: dict[int, int] = {}
                for j in range(n):
                    self._log(2, f"[CoTraining]   Model {j}: submitting CPS scoring on GPU "
                                 f"{pool.round_robin_gpu(j)}...")
                    spec = self._make_cps_spec(
                        j, self._cpu_state_dict(h[j]), models_datasets[j][0], val_cpu,
                        unit_ids_int, unit_x, unit_lb)
                    score_jobs[j] = pool.submit_cps(pool.round_robin_gpu(j), spec)
                results = pool.gather(list(score_jobs.values()))

                all_preds: dict[int, OrderedDict] = {
                    j: self._assemble_all_preds(results[score_jobs[j]]["units"], xu_by_unit)
                    for j in range(n)
                }

                # Phase 2 — owner-based selection + label estimation (main process).
                selected_per_model, remaining_suspension_ids, added = self._select_and_label(
                    all_preds=all_preds, h=h, models_datasets=models_datasets, pool_ids=pool_ids,
                    remaining_suspension_ids=remaining_suspension_ids, suspension_ids=suspension_ids,
                    suspension_time_steps=suspension_time_steps, add_ratio=add_ratio)

                if added == 0:
                    self._log(1, f"[CoTraining] Early stop at iteration {i + 1}: no unit selected.")
                    break

                # Phase 3 — fine-tune every receiver in parallel.
                finetune_jobs: dict[int, int] = {}
                for j in range(n):
                    if selected_per_model[j]:
                        xj, yj = models_datasets[j]
                        new_xu, new_lu = self._concat_selected_units(selected_per_model[j], yj)
                        xj = torch.cat([xj, new_xu], dim=0)
                        yj = torch.cat([yj, new_lu], dim=0)
                        models_datasets[j] = (xj, yj)
                        self._log(1, f"[CoTraining]   Fine-tuning model {j} | "
                                     f"added {len(selected_per_model[j])} unit(s) | dataset: {len(xj)} samples")
                        finetune_jobs[j] = pool.submit_finetune(
                            pool.round_robin_gpu(j),
                            self._make_finetune_spec(j, self._cpu_state_dict(h[j]), xj, yj, val_cpu))
                results = pool.gather(list(finetune_jobs.values()))
                for j, jid in finetune_jobs.items():
                    h[j] = self._rebuild_module(j, results[jid]["state_dict"])

                if metrics_enabled:
                    self._log_stage_metrics(
                        stage=f"iteration_{i + 1}", h=h, models_datasets=models_datasets,
                        test_data=test_data, test_label=test_label, val_data=val_data,
                        val_label=val_label, score_callback=score_callback,
                        weight_callback=weight_callback, weight_mode=weight_mode, metrics_file=metrics_file,
                    )

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="final", h=h, models_datasets=models_datasets, test_data=test_data,
                    test_label=test_label, val_data=val_data, val_label=val_label,
                    score_callback=score_callback, weight_callback=weight_callback,
                    weight_mode=weight_mode, metrics_file=metrics_file,
                )

            self._log(1, "[CoTraining] Training complete.")
            self.lightning_modules = h
        finally:
            pool.shutdown()
