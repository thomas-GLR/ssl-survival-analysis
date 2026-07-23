import csv
import gc
import os
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
    * **Owner-based selection**: each model independently selects its own top ``add_ratio``
      fraction of the pool, ranked by its *own* conformal interval width. Selection is purely
      intra-model — a model is never compared to another — so no cross-model width normalization
      is needed. A unit's *owners* are the models that selected it; the label is injected into the
      *other* models (co-training). A unit selected by *every* model carries no information
      asymmetry and is skipped; a unit selected by no model is ignored.
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
        :param monotone_residual_weight: Weight of the residual term in the per-model ranking score
            (only used when ``use_monotone_projection`` is ``True``).
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

        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if not (0 <= model_pred_blend <= 1):
            raise ValueError("model_pred_blend must be in [0, 1].")
        if fine_tune_lr_factor <= 0:
            raise ValueError("fine_tune_lr_factor must be positive.")

        self.n_neighbors = n_neighbors
        self.model_pred_blend = model_pred_blend
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.fine_tune_max_epochs = fine_tune_max_epochs
        self.fine_tune_patience = fine_tune_patience

        # Set per-run by ``train``; gates the (no-op-by-default) label-estimator instrumentation.
        self._diagnostics: bool = False

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

    def _ranking_scores(
            self,
            all_preds: dict[int, OrderedDict],
    ) -> dict[int, dict[int, float]]:
        """Per-model, per-unit score used to rank each model's own units (smaller = better).

        Because selection is intra-model (each model ranks only against itself), no cross-model
        normalization is needed. With monotone projection off, the raw conformal interval width
        (tuple index 5) is used directly — dividing by a per-model constant would not change that
        model's own ranking. With projection on (and a non-zero residual weight), the width and the
        monotone residual live on different scales and must be combined, so the inherited
        :meth:`CoTrainingEnsemble_v2._selection_scores` (per-model width + residual blend) is used;
        that normalization is purely within a model, never a comparison between models.

        :param all_preds: ``{model_index: OrderedDict{unit_id_int: tuple}}`` with width at tuple
            index 5 and residual at index 6.
        :return: ``{model_index: {unit_id_int: score}}`` (smaller = more confident).
        """
        if self.use_monotone_projection and self.monotone_residual_weight != 0:
            return self._selection_scores(all_preds)
        return {
            j: {uid: tup[5] for uid, tup in preds.items()}
            for j, preds in all_preds.items()
        }

    def _select_owner_units(
            self,
            scores: dict[int, dict[int, float]],
            pool_uids: list[int],
            select_ratio: float,
    ) -> list[tuple[int, list[int], list[int]]]:
        """Per-model top-fraction selection with set-membership ownership.

        Each model independently selects its own top ``select_ratio`` fraction of the pool, ranked
        by its *own* score (smaller = more confident); this is a purely intra-model ranking, so no
        cross-model comparison or normalization is involved. A unit's *owners* are the models that
        selected it. A unit selected by *every* model is skipped (no information asymmetry to
        teach); a unit selected by no model is ignored. Owners teach the label to the remaining
        *receiver* models.

        :param scores: ``{model_index: {unit_id_int: score}}`` (smaller = more confident).
        :param pool_uids: The pooled unit ids (ints) available this iteration.
        :param select_ratio: Fraction in ``(0, 1]`` each model selects from the pool
            (``k = max(1, round(select_ratio * pool))``).
        :return: ``[(unit_id_int, owners, receivers), ...]``.
        """
        n_models = self.number_of_models
        k = max(1, round(select_ratio * len(pool_uids)))

        selected_sets: list[set[int]] = []
        for j in range(n_models):
            ranked_j = sorted(
                (uid for uid in pool_uids if uid in scores[j]), key=lambda uid: scores[j][uid])
            selected_sets.append(set(ranked_j[:k]))

        selected: list[tuple[int, list[int], list[int]]] = []
        for uid in pool_uids:
            owners = [j for j in range(n_models) if uid in selected_sets[j]]
            if not owners or len(owners) >= n_models:
                # Selected by nobody (ignore) or by everybody (no asymmetry to teach) -> skip.
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
            z_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
            h: list[LightningModule],
    ) -> float:
        """Estimate the pseudo-RUL of a unit's **last window** (owner-consensus, closed form).

        For each owner: embed the unit's last window, take the distance-weighted mean RUL of its
        ``n_neighbors`` nearest labelled neighbours in that model's latent space, optionally blend
        with the model's conformal median ``c`` (``model_pred_blend``), and clip to the conformal
        band ``[a, b]``. Multiple owners are combined by a simple (equal-weight) average — selection
        no longer compares confidence across models, so there is no cross-model weight to apply.
        """
        labels: list[float] = []
        for j in owners:
            _, xu, _, a, b, _, _, c = all_preds[j][uid]
            z_u = self._embed(h[j], xu[-1:])  # (1, D)
            z_label, y_label = z_cache[j]
            labels.append(self._knn_label(z_u, z_label, y_label, a, b, c))
        return sum(labels) / len(labels)

    def _knn_label(
            self,
            z_u: torch.Tensor,
            z_label: torch.Tensor,
            y_label: torch.Tensor,
            a: float,
            b: float,
            c: float,
    ) -> float:
        """Closed-form pseudo-RUL of one embedded last window from the labelled set's k-NN.

        Distance-weighted mean RUL of ``z_u``'s ``n_neighbors`` nearest labelled neighbours in
        latent space, optionally blended with the model's conformal median ``c``
        (``model_pred_blend``) and clipped to the conformal band ``[a, b]``. Factored out of
        :meth:`_estimate_label_last` so the label diagnostics can score the exact same rule.

        :param z_u: The embedded last window, shape ``(1, D)``, on the same device as ``z_label``.
        :param z_label: Labelled-set embeddings ``(N, D)``.
        :param y_label: Labelled-set RULs ``(N,)``, aligned with ``z_label``.
        :param a: Conformal band lower bound (clip floor).
        :param b: Conformal band upper bound (clip ceiling).
        :param c: Model conformal median (blended in when ``model_pred_blend > 0``).
        :return: The clipped, optionally blended pseudo-RUL.
        """
        k = min(self.n_neighbors, z_label.shape[0])
        dist = torch.cdist(z_u, z_label).view(-1)
        knn_dist, knn_idx = torch.topk(dist, k, largest=False)
        w = 1.0 / (knn_dist + 1e-8)
        rul_knn = float((w * y_label[knn_idx]).sum() / w.sum())
        label = self.model_pred_blend * c + (1.0 - self.model_pred_blend) * rul_knn
        return float(min(max(label, a), b))

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
            suspension_lower_bounds: torch.Tensor | None = None,
    ) -> tuple[list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], torch.Tensor, int,
               dict[str, float] | None]:
        """Phase 2 (path-independent): owner-based selection + closed-form pseudo-labelling.

        Returns ``(selected_per_model, remaining_suspension_ids, added, bias_stats)`` where
        ``selected_per_model[r]`` is the list of ``(unit_id, xu, per_window_label)`` assigned to
        receiver model ``r`` (ready for the inherited ``_concat_selected_units``). ``bias_stats``
        aggregates, over the units actually injected this iteration, the mean pseudo-label, the mean
        conformal median (p50), the mean model prediction and (when ``suspension_lower_bounds`` is
        given) the mean survival lower bound plus how many pseudo-labels / medians fall below it —
        it is ``None`` when diagnostics are off or nothing was selected.

        :param suspension_lower_bounds: Optional per-window survival lower bounds, row-aligned with
            ``suspension_ids``; used only for the injected-label bias diagnostics (never changes the
            injected label here).
        """
        n_models = self.number_of_models
        selected_per_model: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(n_models)
        ]

        scores = self._ranking_scores(all_preds)
        pool_uids = [int(u) for u in pool_ids]

        selected = self._select_owner_units(scores, pool_uids, add_ratio)
        if not selected:
            return selected_per_model, remaining_suspension_ids, 0, None

        owner_set: set[int] = set()
        for _, owners, _ in selected:
            owner_set.update(owners)
        z_cache = self._compute_label_caches(h, models_datasets, owner_set)

        self._log(1, f"[CoTraining]   Each model selecting its top {add_ratio} of "
                     f"{len(pool_uids)} pooled unit(s); {len(selected)} unit(s) have an owner "
                     f"asymmetry (selected by some but not all models).")

        inj_vals: list[float] = []
        p50_vals: list[float] = []
        raw_vals: list[float] = []
        lb_vals: list[float] = []

        added = 0
        for uid, owners, receivers in selected:
            label_last = self._estimate_label_last(uid, owners, all_preds, z_cache, h)

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

            if self._diagnostics:
                owner_c = [all_preds[j][uid][7] for j in owners]
                owner_raw = [float(all_preds[j][uid][2].view(-1)[-1]) for j in owners]
                inj_vals.append(label_last)
                p50_vals.append(sum(owner_c) / len(owner_c))
                raw_vals.append(sum(owner_raw) / len(owner_raw))
                if suspension_lower_bounds is not None:
                    lb_u = suspension_lower_bounds[suspension_ids == uid_tensor].view(-1)
                    lb_vals.append(float(lb_u[-1]))

            self._log(1, f"[CoTraining]   Unit {uid}: owners={owners} receivers={receivers} | "
                         f"last-window RUL={label_last:.4f}")

        # Free the per-model latent caches (each holds an (N, D) embedding of the labelled set).
        del z_cache
        gc.collect()

        bias_stats = self._pseudo_label_bias_stats(inj_vals, p50_vals, raw_vals, lb_vals)
        return selected_per_model, remaining_suspension_ids, added, bias_stats

    def _pseudo_label_bias_stats(
            self,
            inj_vals: list[float],
            p50_vals: list[float],
            raw_vals: list[float],
            lb_vals: list[float],
    ) -> dict[str, float] | None:
        """Aggregate this iteration's injected-label diagnostics (``None`` if nothing collected).

        Compares the mean injected pseudo-label against the mean conformal median (p50) and the
        mean model prediction, and — when survival lower bounds were available — reports the mean
        ``pseudo-label - lower_bound`` gap and how many pseudo-labels / medians fall *below* the
        lower bound (a censoring violation: predicting a RUL the unit is known to have exceeded).
        """
        if not inj_vals:
            return None
        inj = torch.tensor(inj_vals)
        p50 = torch.tensor(p50_vals)
        raw = torch.tensor(raw_vals)
        has_lb = len(lb_vals) == len(inj_vals)
        lb = torch.tensor(lb_vals) if has_lb else None
        stats = {
            "n_selected": float(len(inj_vals)),
            "inj_label_mean": float(inj.mean()),
            "p50_mean": float(p50.mean()),
            "model_pred_mean": float(raw.mean()),
            "lb_mean": float(lb.mean()) if has_lb else float("nan"),
            "inj_minus_lb_mean": float((inj - lb).mean()) if has_lb else float("nan"),
            "inj_below_lb": float((inj < lb).sum()) if has_lb else float("nan"),
            "p50_below_lb": float((p50 < lb).sum()) if has_lb else float("nan"),
        }
        msg = (f"[CoTraining]   [diag] injected {len(inj_vals)} label(s) | "
               f"mean pseudo-label={stats['inj_label_mean']:.4f} | mean p50={stats['p50_mean']:.4f} | "
               f"mean model pred={stats['model_pred_mean']:.4f}")
        if has_lb:
            msg += (f" | mean lower-bound={stats['lb_mean']:.4f} | "
                    f"pseudo<lb: {int(stats['inj_below_lb'])}/{len(inj_vals)} | "
                    f"p50<lb: {int(stats['p50_below_lb'])}/{len(inj_vals)}")
        self._log(1, msg)
        return stats

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
    # Label-estimator diagnostics (instrumentation, no effect on training)
    # ------------------------------------------------------------------ #

    def _estimator_accuracy_stats(
            self,
            h: list[LightningModule],
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]],
            eval_x: torch.Tensor,
            eval_y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
            max_samples: int | None,
    ) -> dict[int, dict[str, float]]:
        """Measure, per model, how accurate each last-window RUL estimator is on *labelled* data.

        This is the decisive check for whether semi-supervised injection can help: on a held-out
        set whose true RUL is known, it runs the exact scoring path used on censored units and
        compares three estimators against the ground truth:

        * ``raw``  — the model's own point prediction (what supervised training already gives),
        * ``p50``  — the conformal predictive median ``c`` (the 0.5 quantile),
        * ``knn``  — the latent k-NN estimate clipped to ``[a, b]`` (the injected pseudo-label rule).

        If ``knn`` is not clearly below ``raw`` here, the pseudo-labels carry no new information and
        injecting them can only add noise; comparing ``p50`` to ``knn`` answers directly whether the
        median quantile would be a better label than the current k-NN rule.

        :param h: Current models.
        :param models_datasets: Per-model accumulated ``(x, y)`` (the k-NN neighbour set + the
            ``DifficultyEstimator`` training set — identical to what scoring uses).
        :param eval_x: Held-out features ``(N, seq_len, n_features)`` with known labels.
        :param eval_y: Held-out RULs ``(N,)`` (or ``(N, 1)``).
        :param val_cpu: CPU ``(val_x, val_y)`` used to calibrate the conformal predictive system.
        :param max_samples: If set, evaluate on a random subsample of this many windows (cost cap).
        :return: ``{model_index: {"raw": rmse, "p50": rmse, "knn": rmse}}``.
        """
        eval_x = eval_x.detach().cpu()
        eval_y = eval_y.detach().cpu().view(-1).float()
        n = eval_x.shape[0]
        if max_samples is not None and n > max_samples:
            idx = torch.randperm(n)[:max_samples]
            eval_x, eval_y = eval_x[idx], eval_y[idx]
        m = eval_x.shape[0]

        # Each eval window is treated as its own single-window "unit" (its own last window), so the
        # CPS job returns that window's percentiles a / c / b and its raw prediction.
        unit_ids = list(range(m))
        unit_x = [eval_x[i:i + 1] for i in range(m)]

        z_cache = self._compute_label_caches(h, models_datasets, set(range(self.number_of_models)))
        stats: dict[int, dict[str, float]] = {}
        try:
            for j in range(self.number_of_models):
                spec = self._make_cps_spec(
                    j, self._cpu_state_dict(h[j]), models_datasets[j][0], val_cpu,
                    unit_ids, unit_x, None)
                spec.accelerator = self._inline_accelerator
                spec.devices = self._inline_devices
                result = run_cps_score_job(spec)

                z_all = self._embed(h[j], eval_x)  # (m, D)
                z_label, y_label = z_cache[j]
                by_uid = {u[0]: u for u in result["units"]}

                raw = torch.empty(m)
                p50 = torch.empty(m)
                knn = torch.empty(m)
                for i in range(m):
                    _, raw_preds, a, b, _, _, c = by_uid[i]
                    raw[i] = float(raw_preds.view(-1)[-1])
                    p50[i] = c
                    knn[i] = self._knn_label(z_all[i:i + 1], z_label, y_label, a, b, c)

                stats[j] = {
                    "raw": float(torch.sqrt(torch.mean((raw - eval_y) ** 2))),
                    "p50": float(torch.sqrt(torch.mean((p50 - eval_y) ** 2))),
                    "knn": float(torch.sqrt(torch.mean((knn - eval_y) ** 2))),
                }
                self._log(1, f"[CoTraining]   [diag] model {j} last-window RMSE on {m} labelled "
                             f"windows | raw(model)={stats[j]['raw']:.4f} | "
                             f"p50(median)={stats[j]['p50']:.4f} | knn(pseudo-label)={stats[j]['knn']:.4f}")
        finally:
            del z_cache
            gc.collect()
        return stats

    def _write_diagnostics_row(
            self,
            diagnostics_file: str,
            stage: str,
            acc_stats: dict[int, dict[str, float]],
            bias_stats: dict[str, float] | None,
    ) -> None:
        """Append one diagnostics row (per-model estimator RMSE + injected-label bias) to a CSV.

        Header is written only when the file does not yet exist, mirroring the append style of the
        metrics CSV. ``bias_stats`` is ``None`` for stages without a selection step (``initial`` /
        ``final``), in which case its columns are left blank.
        """
        n = self.number_of_models
        header = ["stage"]
        for j in range(n):
            header += [f"raw_rmse_{j}", f"p50_rmse_{j}", f"knn_rmse_{j}"]
        header += ["n_selected", "inj_label_mean", "p50_mean", "model_pred_mean", "lb_mean",
                   "inj_minus_lb_mean", "inj_below_lb", "p50_below_lb"]

        row: list[object] = [stage]
        for j in range(n):
            s = acc_stats.get(j, {})
            row += [s.get("raw", ""), s.get("p50", ""), s.get("knn", "")]
        if bias_stats is None:
            row += ["", "", "", "", "", "", "", ""]
        else:
            row += [bias_stats["n_selected"], bias_stats["inj_label_mean"], bias_stats["p50_mean"],
                    bias_stats["model_pred_mean"], bias_stats["lb_mean"],
                    bias_stats["inj_minus_lb_mean"], bias_stats["inj_below_lb"],
                    bias_stats["p50_below_lb"]]

        write_header = not os.path.exists(diagnostics_file)
        with open(diagnostics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    def _log_diagnostics(
            self,
            stage: str,
            h: list[LightningModule],
            models_datasets: list[tuple[torch.Tensor, torch.Tensor]],
            eval_x: torch.Tensor | None,
            eval_y: torch.Tensor | None,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
            bias_stats: dict[str, float] | None,
            diagnostics_file: str | None,
            max_samples: int | None,
    ) -> None:
        """Compute the estimator-accuracy stats for ``stage`` and write / log the diagnostics row."""
        acc_stats: dict[int, dict[str, float]] = {}
        if eval_x is not None and eval_y is not None:
            acc_stats = self._estimator_accuracy_stats(
                h, models_datasets, eval_x, eval_y, val_cpu, max_samples)
        if diagnostics_file is not None:
            self._write_diagnostics_row(diagnostics_file, stage, acc_stats, bias_stats)

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
            diagnostics: bool = False,
            diagnostics_file: str | None = None,
            diagnostics_max_samples: int = 200,
    ) -> None:
        """Train the v3 co-training ensemble.

        Each iteration: sample a random pool of censored units, score every pooled unit with each
        model's conformal predictive system, select units by owner-based confidence, estimate each
        selected unit's pseudo-label with the latent k-NN estimator + backward extrapolation,
        inject it into the receiver models, and fine-tune those receivers on their accumulated
        data.

        Args mirror :meth:`CoTrainingEnsemble_v2.train`, with these additions:

        :param suspension_time_steps: Optional per-window ``time_step`` for the censored data,
            row-aligned with ``suspension_data`` / ``suspension_ids`` (shape ``(N,)`` or
            ``(N, 1)``). Used to backward-extrapolate each unit's last-window RUL to its earlier
            windows. If ``None``, unit-spaced window indices are used instead.
        :param add_ratio: Per-model top fraction in ``(0, 1]`` — each model selects its top
            ``add_ratio`` of the pool by its own conformal width; ownership then follows from which
            models selected each unit.
        :param diagnostics: When ``True``, instrument the pseudo-label estimator without changing
            training. At every stage it measures, on a labelled held-out set, the last-window RMSE
            of three estimators — the model's own prediction (``raw``), the conformal median
            (``p50``) and the injected latent-kNN pseudo-label (``knn``) — so it can be checked
            whether the pseudo-labels carry usable information and whether the p50 quantile would be
            a better label than the k-NN rule. Each iteration it also logs the mean injected label
            vs mean p50 vs mean model prediction vs mean survival lower bound (and how many labels
            fall below it). Off by default (zero extra cost).
        :param diagnostics_file: Optional ``.csv`` destination for the diagnostics rows (one per
            stage). Header written once; blank bias columns on ``initial`` / ``final``.
        :param diagnostics_max_samples: Random subsample size of the labelled held-out set used for
            the estimator-accuracy diagnostics (caps their cost). Defaults to ``200``.
        """
        self._log_file_path = log_file
        self._diagnostics = diagnostics
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
                     f"per-model select ratio: {add_ratio} | "
                     f"mode: {'parallel(' + str(self.gpu_ids) + ')' if self._parallel else 'sequential'}")

        if self._parallel:
            self._train_parallel(
                failure_data=failure_data,
                failure_label=failure_label,
                suspension_data=suspension_data,
                suspension_ids=suspension_ids,
                suspension_time_steps=suspension_time_steps,
                # Pass the full bounds (not the clip-gated view) so the parallel-path bias
                # diagnostics can report them; the CPS censoring clip stays gated by clip_bounds,
                # which _train_parallel recomputes internally.
                suspension_lower_bounds=suspension_lower_bounds,
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
                diagnostics=diagnostics,
                diagnostics_file=diagnostics_file,
                diagnostics_max_samples=diagnostics_max_samples,
            )
            return

        val_cpu = self._cpu_pair(val_data, val_label)
        # Labelled held-out set for the estimator-accuracy diagnostics: the test set when
        # available (the honest target), else the validation set (a relative-only comparison,
        # since val also calibrates the conformal predictive systems).
        diag_eval_x, diag_eval_y = (
            (test_data, test_label) if test_data is not None else (val_data, val_label))

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
        if diagnostics:
            self._log_diagnostics(
                stage="initial", h=h, models_datasets=models_datasets, eval_x=diag_eval_x,
                eval_y=diag_eval_y, val_cpu=val_cpu, bias_stats=None,
                diagnostics_file=diagnostics_file, max_samples=diagnostics_max_samples)

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
            selected_per_model, remaining_suspension_ids, added, bias_stats = self._select_and_label(
                all_preds=all_preds, h=h, models_datasets=models_datasets, pool_ids=pool_ids,
                remaining_suspension_ids=remaining_suspension_ids, suspension_ids=suspension_ids,
                suspension_time_steps=suspension_time_steps, add_ratio=add_ratio,
                suspension_lower_bounds=suspension_lower_bounds)

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
            if diagnostics:
                self._log_diagnostics(
                    stage=f"iteration_{i + 1}", h=h, models_datasets=models_datasets,
                    eval_x=diag_eval_x, eval_y=diag_eval_y, val_cpu=val_cpu, bias_stats=bias_stats,
                    diagnostics_file=diagnostics_file, max_samples=diagnostics_max_samples)

            del all_preds, selected_per_model
            gc.collect()

        if metrics_enabled:
            self._log_stage_metrics(
                stage="final", h=h, models_datasets=models_datasets, test_data=test_data,
                test_label=test_label, val_data=val_data, val_label=val_label,
                score_callback=score_callback, weight_callback=weight_callback,
                weight_mode=weight_mode, metrics_file=metrics_file,
            )
        if diagnostics:
            self._log_diagnostics(
                stage="final", h=h, models_datasets=models_datasets, eval_x=diag_eval_x,
                eval_y=diag_eval_y, val_cpu=val_cpu, bias_stats=None,
                diagnostics_file=diagnostics_file, max_samples=diagnostics_max_samples)

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
            diagnostics: bool = False,
            diagnostics_file: str | None = None,
            diagnostics_max_samples: int = 200,
    ) -> None:
        """Multi-GPU parallel version of :meth:`train`.

        Initial from-scratch training, per-model CPS scoring, and per-model fine-tuning each run
        concurrently across all ``gpu_ids`` via a :class:`CoTrainingGpuPool`. Owner-based
        selection, the latent k-NN label estimation and the (optional) label-estimator diagnostics
        stay in the main process (they use the rebuilt models ``h``). The diagnostics args mirror
        :meth:`train`.
        """
        n = self.number_of_models
        val_cpu = self._cpu_pair(val_data, val_label)
        clip_bounds = self.use_monotone_projection and suspension_lower_bounds is not None
        # Labelled held-out set for the estimator-accuracy diagnostics (test when available, else
        # val — see the sequential path for the caveat).
        diag_eval_x, diag_eval_y = (
            (test_data, test_label) if test_data is not None else (val_data, val_label))

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
            if diagnostics:
                self._log_diagnostics(
                    stage="initial", h=h, models_datasets=models_datasets, eval_x=diag_eval_x,
                    eval_y=diag_eval_y, val_cpu=val_cpu, bias_stats=None,
                    diagnostics_file=diagnostics_file, max_samples=diagnostics_max_samples)

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
                selected_per_model, remaining_suspension_ids, added, bias_stats = self._select_and_label(
                    all_preds=all_preds, h=h, models_datasets=models_datasets, pool_ids=pool_ids,
                    remaining_suspension_ids=remaining_suspension_ids, suspension_ids=suspension_ids,
                    suspension_time_steps=suspension_time_steps, add_ratio=add_ratio,
                    suspension_lower_bounds=suspension_lower_bounds)

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
                if diagnostics:
                    self._log_diagnostics(
                        stage=f"iteration_{i + 1}", h=h, models_datasets=models_datasets,
                        eval_x=diag_eval_x, eval_y=diag_eval_y, val_cpu=val_cpu, bias_stats=bias_stats,
                        diagnostics_file=diagnostics_file, max_samples=diagnostics_max_samples)

            if metrics_enabled:
                self._log_stage_metrics(
                    stage="final", h=h, models_datasets=models_datasets, test_data=test_data,
                    test_label=test_label, val_data=val_data, val_label=val_label,
                    score_callback=score_callback, weight_callback=weight_callback,
                    weight_mode=weight_mode, metrics_file=metrics_file,
                )
            if diagnostics:
                self._log_diagnostics(
                    stage="final", h=h, models_datasets=models_datasets, eval_x=diag_eval_x,
                    eval_y=diag_eval_y, val_cpu=val_cpu, bias_stats=None,
                    diagnostics_file=diagnostics_file, max_samples=diagnostics_max_samples)

            self._log(1, "[CoTraining] Training complete.")
            self.lightning_modules = h
        finally:
            pool.shutdown()
