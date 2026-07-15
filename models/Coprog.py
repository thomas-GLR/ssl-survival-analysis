import copy
from typing import Callable

import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from models.coprog_gpu_pool import (
    CandidateContext,
    GpuTrainingPool,
    TrainingSpec,
    run_training_job,
)


class Coprog:
    """
    Co-training-based PROGnostics (COPROG) algorithm.

    Reference: "A co-training-based approach for prediction of remaining useful
    life utilizing both failure and suspension data." Chao Hu, Byeng D. Youn, Taejin Kim, Pingfeng Wang.

    Two models are trained on complementary views of the failure data. At each
    iteration, each model attempts to self-label the most informative suspension
    sample and passes it to the *other* model (cross-training). Training stops
    when neither model finds a beneficial sample or after `T` iterations.

    Training is delegated to PyTorch Lightning. Call **one** of the two setup methods to
    provide the training configuration before calling :meth:`train`:

    * :meth:`setup_training_legacy` (``lightning_modules`` + ``trainer_factories``): a
      Lightning module template is deep-copied and a fresh ``Trainer`` is built from a
      factory for every training. Runs sequentially in the current process.
    * :meth:`setup_training_builder` (``module_builders`` + ``max_epochs`` + ``patiences``):
      a fresh module is built from a *picklable* builder, its initial weights are pinned to a
      one-time snapshot, and the ``Trainer`` is built internally. This style additionally
      supports **multi-GPU parallel training**: when ``gpu_ids`` lists two or more GPUs,
      the independent trainings of each phase run concurrently across them (the GPU list
      is split between the two models — see :meth:`setup_training_builder`).

    :param first_model:  A torch.nn.Module for view 1.
    :param second_model: A torch.nn.Module for view 2.
    :param verbose: Verbosity level. 0 = silent, 1 = key decisions, 2 = full per-candidate details.
    """

    def __init__(
            self,
            first_model: nn.Module,
            second_model: nn.Module,
            verbose: int = 0,
    ):
        # Keep the originals so external code can still reference them.
        self.first_model = first_model
        self.second_model = second_model
        self.models = [first_model, second_model]

        self.w1 = None
        self.w2 = None

        self.verbose = verbose

        # Set through setup_training_legacy()
        self.lightning_modules: list[LightningModule] | None = None  # pristine templates
        self.trainer_factories: list[Callable[[], Trainer]] | None = None
        self.batch_sizes: list[int] | None = None
        self.shuffle_dataloaders: list[bool] | None = None

        # Set through setup_training_builder()
        self.module_builders: list[Callable[[], LightningModule]] | None = None
        self.max_epochs: list[int] | None = None
        self.patiences: list[int] | None = None
        self.gpu_ids: list[int] | None = None

        self._use_builders: bool = False
        self._parallel: bool = False
        self._initial_state_dicts: list[dict[str, torch.Tensor]] | None = None
        # Accelerator/devices used for inline (single-GPU / auto) training in builder style.
        self._inline_accelerator: str = "auto"
        self._inline_devices = None
        self._configured: bool = False

        # Trained Lightning modules (set after calling .train())
        self._h1: LightningModule | None = None
        self._h2: LightningModule | None = None

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message)

    def _store_dataloader_config(
            self,
            batch_sizes: list[int],
            shuffle_dataloaders: list[bool],
    ) -> None:
        """Validate and store the per-model ``DataLoader`` configuration shared by both setup styles.

        Args:
            batch_sizes (list[int]): Batch size used to train each model.
            shuffle_dataloaders (list[bool]): Whether to shuffle each training ``DataLoader``.

        Raises:
            ValueError: If either list is missing or does not have one entry per model.
        """
        model_number = len(self.models)
        if batch_sizes is None or shuffle_dataloaders is None:
            raise ValueError("batch_sizes and shuffle_dataloaders are required.")
        if len(batch_sizes) != model_number or len(shuffle_dataloaders) != model_number:
            raise ValueError(f"batch_sizes {len(batch_sizes)} and shuffle_dataloaders {len(shuffle_dataloaders)}"
                             f" must both have length {model_number}.")
        self.batch_sizes = batch_sizes
        self.shuffle_dataloaders = shuffle_dataloaders

    def setup_training_legacy(
            self,
            lightning_modules: list[LightningModule],
            trainer_factories: list[Callable[[], Trainer]],
            batch_sizes: list[int],
            shuffle_dataloaders: list[bool],
    ) -> None:
        r"""Setup **legacy-style** training from pre-built modules and trainer factories.

        Training runs sequentially in the current process. Each list has length two:
        index 0 configures ``first_model`` and index 1 configures ``second_model``.

        Args:
            lightning_modules (list[LightningModule]): Templates deep-copied for every
                training call (never mutated).
            trainer_factories (list[Callable[[], Trainer]]): Factories building a fresh
                ``Trainer`` per call; should include a ``ModelCheckpoint`` (monitoring
                ``val_loss``) so the best weights are reloaded.
            batch_sizes (list[int]): Batch size used to train each model.
            shuffle_dataloaders (list[bool]): Whether to shuffle each training ``DataLoader``.

        Raises:
            ValueError: If a list does not have one entry per model.
        """
        model_number = len(self.models)
        if len(lightning_modules) != model_number or len(trainer_factories) != model_number:
            raise ValueError(f"lightning_modules and trainer_factories must both have length {model_number}.")

        self._store_dataloader_config(batch_sizes, shuffle_dataloaders)

        self.lightning_modules = lightning_modules
        self.trainer_factories = trainer_factories
        self._use_builders = False
        self._parallel = False
        self._configured = True

    def setup_training_builder(
            self,
            module_builders: list[Callable[[], LightningModule]],
            max_epochs: list[int],
            patiences: list[int],
            batch_sizes: list[int],
            shuffle_dataloaders: list[bool],
            gpu_ids: list[int] | None = None,
    ) -> None:
        r"""Setup **builder-style** training, the style required for multi-GPU parallel training.

        A fresh module is built from a *picklable* builder, its initial weights are pinned to
        a one-time snapshot, and the ``Trainer`` is built internally. Each list has length
        two: index 0 configures ``first_model`` and index 1 configures ``second_model``.

        Args:
            module_builders (list[Callable[[], LightningModule]]): Picklable callables
                (module-level functions or ``functools.partial`` — no lambdas/closures) each
                returning a *fresh* ``LightningModule``. Must be picklable because parallel
                workers rebuild modules across a process boundary.
            max_epochs (list[int]): Max training epochs per model.
            patiences (list[int]): ``EarlyStopping`` patience per model.
            batch_sizes (list[int]): Batch size used to train each model.
            shuffle_dataloaders (list[bool]): Whether to shuffle each training ``DataLoader``.
            gpu_ids (list[int] | None): Physical GPU ids to train on.
                ``None`` → auto (Lightning picks one GPU); ``[g]`` → pin to GPU ``g``,
                sequential; ``[g0, g1, ...]`` (>=2) → parallel: the list is split in half
                between the two models (model 1 gets the first half, model 2 the second;
                an odd extra GPU goes to model 2), each model's trainings running on its own
                GPU subset so the two models never share a GPU.

        Raises:
            ValueError: If a list does not have one entry per model.
        """
        model_number = len(self.models)
        if len(module_builders) != model_number or len(max_epochs) != model_number or len(patiences) != model_number:
            raise ValueError(f"module_builders, max_epochs and patiences must all have length {model_number}.")

        self._store_dataloader_config(batch_sizes, shuffle_dataloaders)

        self.module_builders = module_builders
        self.max_epochs = max_epochs
        self.patiences = patiences
        self.gpu_ids = list(gpu_ids) if gpu_ids else None
        self._use_builders = True

        # Snapshot one initial weight set per model so every from-scratch training
        # starts from identical weights (matching the legacy deep-copy-of-template
        # behaviour) and so workers can reproduce that init across process boundaries.
        self._initial_state_dicts = []
        for builder in module_builders:
            template = builder()
            self._initial_state_dicts.append(
                {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
            )

        # Decide inline accelerator/devices and whether to run in parallel.
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
            # Inline fallback (unused while parallel) kept sane just in case.
            self._inline_accelerator = "gpu"
            self._inline_devices = [self.gpu_ids[0]]

        self._configured = True

    def _check_if_training_is_possible(self) -> None:
        if not self._configured:
            raise ValueError("You need to call setup_training_legacy or setup_training_builder before calling train.")

    def train(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: int,
            val_data: torch.Tensor | None = None,
            val_label: torch.Tensor | None = None,
    ) -> None:
        """
        Full COPROG training procedure (Algorithm 1 in the paper).

        Dispatches to the sequential implementation (single GPU / auto / legacy) or the
        multi-GPU parallel implementation depending on the setup-method config.

        :param failure_data:        Shape (N, *feature_dims) – labelled failure set L.
        :param failure_label:       Shape (N,) or (N, 1)     – RUL labels for L.
        :param suspension_data:     Shape (M, *feature_dims) – unlabelled suspension set U.
        :param suspension_ids:      Shape (M,) – unit id of each suspension sequence.
        :param iterations:          Maximum number of co-training rounds T.
        :param suspension_pool_size: Size u of the random sub-pool U' drawn each round. If -1 then all censored are selected
        :param val_data:            Optional validation features used for early stopping /
                                    best-checkpoint selection during every training call.
        :param val_label:           Optional validation labels associated with ``val_data``.
                                    Must be provided together with ``val_data``.
        """
        self._check_if_training_is_possible()

        if (val_data is None) != (val_label is None):
            raise ValueError("val_data and val_label must both be provided or both be None.")

        total_suspension_units = len(torch.unique(suspension_ids))
        self._log(1, f"[Coprog] Starting training | failure samples: {len(failure_data)} | "
                     f"censored units: {total_suspension_units} | "
                     f"max iterations: {iterations} | pool size: {suspension_pool_size} | "
                     f"validation: {'yes' if val_data is not None else 'no'} | "
                     f"mode: {'parallel(' + str(self.gpu_ids) + ')' if self._parallel else 'sequential'}")

        if self._parallel:
            self._train_parallel(
                failure_data, failure_label, suspension_data, suspension_ids,
                iterations, suspension_pool_size, val_data, val_label,
            )
        else:
            self._train_sequential(
                failure_data, failure_label, suspension_data, suspension_ids,
                iterations, suspension_pool_size, val_data, val_label,
            )

    def _train_sequential(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: int,
            val_data: torch.Tensor | None,
            val_label: torch.Tensor | None,
    ) -> None:
        """Sequential COPROG training (single GPU / auto / legacy style)."""
        # Line 1 – L1 = L2 = L  (we split L into two views)
        x1, y1 = failure_data, failure_label
        x2, y2 = failure_data, failure_label

        # Line 2 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
        self._log(1, f"[Coprog] Initial training of h1 on {len(x1)} failure samples...")
        h1 = self._fit_one(0, x1, y1, val_data, val_label)
        self._log(1, f"[Coprog] Initial training of h2 on {len(x2)} failure samples...")
        h2 = self._fit_one(1, x2, y2, val_data, val_label)
        self._log(1, f"[Coprog] Initial training done.")

        remaining_suspension_ids = torch.unique(suspension_ids)

        # Line 3 – Repeat for T times
        for i in range(iterations):

            # Line 4 – Create pool U' of u suspension units
            if len(remaining_suspension_ids) == 0:
                self._log(1, f"[Coprog] Early stop at iteration {i}: no remaining censored units.")
                break

            if suspension_pool_size == -1:
                pool_size = len(remaining_suspension_ids)
            else:
                pool_size = min(suspension_pool_size, len(remaining_suspension_ids))
            shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
            pool_ids = shuffled_ids[:pool_size]  # U'

            self._log(1, f"[Coprog] --- Iteration {i + 1}/{iterations} | "
                         f"remaining censored units: {len(remaining_suspension_ids)} | "
                         f"pool: {pool_ids.tolist()} ---")

            pi = [None, None]  # π1, π2

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

                    # Line 8 – train a temporary model h'j from scratch on the augmented set
                    x_aug = torch.cat([xj, xu], dim=0)

                    lu_p_reshaped = lu_p.view(-1, yj.shape[1]) if yj.dim() > 1 else lu_p.view(-1)
                    y_aug = torch.cat([yj, lu_p_reshaped], dim=0)

                    hj_prime = self._fit_one(j, x_aug, y_aug, val_data, val_label)

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
            x1, y1 = self._apply_cross_label(x1, y1, pi[1])
            x2, y2 = self._apply_cross_label(x2, y2, pi[0])

            # Remove newly labelled samples from U (global pool)
            remaining_suspension_ids = self._drop_selected(remaining_suspension_ids, pi)

            # Line 20 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
            self._log(1, f"[Coprog]   Retraining h1 | dataset size: {len(x1)} samples")
            h1 = self._fit_one(0, x1, y1, val_data, val_label)
            self._log(1, f"[Coprog]   Retraining h2 | dataset size: {len(x2)} samples")
            h2 = self._fit_one(1, x2, y2, val_data, val_label)

        self._log(1, f"[Coprog] Training complete.")

        # Save final trained models
        self._h1 = h1
        self._h2 = h2

    @staticmethod
    def _select_best_candidate(
            ranked: list[tuple[float, dict]],
            excluded_unit: torch.Tensor | None = None,
    ) -> tuple[float, dict] | None:
        """Pick the best eligible candidate from a delta-sorted ranking.

        Because ``ranked`` is sorted by delta descending, the best eligible candidate is
        simply the first entry with a positive delta whose unit is not ``excluded_unit``.

        :param ranked: List of ``(delta, candidate_info)`` tuples sorted by delta descending.
        :param excluded_unit: Unit id to skip so model 2 avoids model 1's pick, or None.

        :return: The chosen ``(delta, candidate_info)`` tuple, or None if no candidate has a
                 positive delta (or the only positive one is excluded).
        """
        for delta, candidate in ranked:
            if delta <= 0:
                return None
            if excluded_unit is not None and bool(candidate["unit_id"] == excluded_unit):
                continue
            return delta, candidate
        return None

    def _train_parallel(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            suspension_ids: torch.Tensor,
            iterations: int,
            suspension_pool_size: int,
            val_data: torch.Tensor | None,
            val_label: torch.Tensor | None,
    ) -> None:
        """Multi-GPU parallel COPROG training.

        Same algorithm as :meth:`_train_sequential`, but the independent trainings of each
        phase run concurrently on separate GPUs via a :class:`GpuTrainingPool`:

        * initial fit of h1/h2 → one job per model, on each model's GPU subset,
        * candidate search → both models' candidate trainings submitted together, each on
          its own GPU subset (round-robin), then gathered,
        * end-of-iteration retrain of h1/h2 → one job per model.

        The two models' candidate searches run in parallel, so (unlike the sequential
        version) model 2 cannot exclude model 1's pick *before* evaluating. Instead both
        evaluate the full pool and the conflict is resolved afterwards with model-1 priority
        (model 2 falls back to its best non-conflicting unit), matching the sequential
        "one distinct unit per model per iteration" outcome.
        """
        split = len(self.gpu_ids) // 2
        subsets = [self.gpu_ids[:split], self.gpu_ids[split:]]
        self._log(1, f"[Coprog] GPU split | model1 -> {subsets[0]} | model2 -> {subsets[1]}")

        val_cpu = self._cpu_pair(val_data, val_label)

        pool = GpuTrainingPool(self.gpu_ids)
        pool.start()
        try:
            xs = [failure_data, failure_data]
            ys = [failure_label, failure_label]

            # Line 2 – initial training of both models, in parallel.
            self._log(1, f"[Coprog] Initial parallel training of h1 and h2...")
            job_ids = {
                j: pool.submit_job(subsets[j][0], self._make_fit_spec(j, xs[j], ys[j], val_cpu))
                for j in range(2)
            }
            results = pool.gather(list(job_ids.values()))
            h = [self._rebuild_module(j, results[job_ids[j]]["state_dict"]) for j in range(2)]
            self._log(1, f"[Coprog] Initial training done.")

            remaining_suspension_ids = torch.unique(suspension_ids)

            for i in range(iterations):
                if len(remaining_suspension_ids) == 0:
                    self._log(1, f"[Coprog] Early stop at iteration {i}: no remaining censored units.")
                    break

                if suspension_pool_size == -1:
                    pool_size = len(remaining_suspension_ids)
                else:
                    pool_size = min(suspension_pool_size, len(remaining_suspension_ids))
                shuffled_ids = remaining_suspension_ids[torch.randperm(len(remaining_suspension_ids))]
                pool_ids = shuffled_ids[:pool_size]

                self._log(1, f"[Coprog] --- Iteration {i + 1}/{iterations} | "
                             f"remaining censored units: {len(remaining_suspension_ids)} | "
                             f"pool: {pool_ids.tolist()} ---")

                # Cache each model's static data on its GPU subset, then submit every
                # candidate training for both models so they run concurrently.
                candidate_info: dict[int, list[dict]] = {0: [], 1: []}
                ctx_ids = [f"ctx_{i}_{j}" for j in range(2)]
                for j in range(2):
                    context = CandidateContext(
                        module_builder=self.module_builders[j],
                        initial_state_dict=self._initial_state_dicts[j],
                        max_epochs=self.max_epochs[j],
                        patience=self.patiences[j],
                        batch_size=self.batch_sizes[j],
                        shuffle=self.shuffle_dataloaders[j],
                        labelled_x=xs[j].detach().cpu(),
                        labelled_y=ys[j].detach().cpu(),
                        val_x=val_cpu[0],
                        val_y=val_cpu[1],
                    )
                    pool.set_context(subsets[j], ctx_ids[j], context)

                    self._log(2, f"[Coprog]   Model h{j + 1}: submitting {len(pool_ids)} candidates "
                                 f"across GPUs {subsets[j]}...")
                    for k, unit_id in enumerate(pool_ids):
                        mask = (suspension_ids == unit_id)
                        xu = suspension_data[mask].detach().cpu()
                        lu_p = self._predict(h[j], xu).detach().cpu()
                        gpu_id = subsets[j][k % len(subsets[j])]
                        job_id = pool.submit_candidate(gpu_id, ctx_ids[j], xu, lu_p)
                        candidate_info[j].append(
                            {"unit_id": unit_id, "xu": xu, "lu_p": lu_p, "job_id": job_id}
                        )

                all_job_ids = [c["job_id"] for j in range(2) for c in candidate_info[j]]
                results = pool.gather(all_job_ids)
                for j in range(2):
                    pool.clear_context(subsets[j], ctx_ids[j])

                # Compute Δ per candidate: mse_orig (current model on L) minus the worker's
                # mse_aug on L. Then pick one distinct unit per model (model-1 priority).
                ranked: list[list[tuple[float, dict]]] = [[], []]
                for j in range(2):
                    mse_orig = self._summed_squared_error(h[j], xs[j], ys[j])
                    for c in candidate_info[j]:
                        delta = mse_orig - results[c["job_id"]]["sse"]
                        ranked[j].append((delta, c))
                    ranked[j].sort(key=lambda t: t[0], reverse=True)

                # Model 1 has priority; model 2 falls back to its best non-conflicting unit.
                best_1 = self._select_best_candidate(ranked[0])
                excluded_unit = best_1[1]["unit_id"] if best_1 is not None else None
                best_2 = self._select_best_candidate(ranked[1], excluded_unit=excluded_unit)

                pi = [None, None]
                for j, best in enumerate((best_1, best_2)):
                    if best is None:
                        self._log(1, f"[Coprog]   Model h{j + 1}: no unit selected "
                                     f"(no positive delta found).")
                        continue
                    delta, c = best
                    pi[j] = (c["unit_id"], c["xu"], c["lu_p"])
                    self._log(1, f"[Coprog]   Model h{j + 1}: selected unit "
                                 f"{c['unit_id'].item()} (delta = {delta:.4f})")

                if pi[0] is None and pi[1] is None:
                    self._log(1, f"[Coprog] Early stop at iteration {i + 1}: "
                                 f"no model found a beneficial censored unit.")
                    break

                # Cross-labelling and global-pool bookkeeping.
                xs[0], ys[0] = self._apply_cross_label(xs[0], ys[0], pi[1])
                xs[1], ys[1] = self._apply_cross_label(xs[1], ys[1], pi[0])
                remaining_suspension_ids = self._drop_selected(remaining_suspension_ids, pi)

                # Retrain both models in parallel.
                self._log(1, f"[Coprog]   Retraining h1 ({len(xs[0])} samples) and "
                             f"h2 ({len(xs[1])} samples) in parallel...")
                job_ids = {
                    j: pool.submit_job(subsets[j][0], self._make_fit_spec(j, xs[j], ys[j], val_cpu))
                    for j in range(2)
                }
                results = pool.gather(list(job_ids.values()))
                h = [self._rebuild_module(j, results[job_ids[j]]["state_dict"]) for j in range(2)]

            self._log(1, f"[Coprog] Training complete.")
            self._h1 = h[0]
            self._h2 = h[1]
        finally:
            pool.shutdown()

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

        p1 = self._predict(self._h1, x).view(-1)
        p2 = self._predict(self._h2, x).view(-1)
        return self.w1 * p1 + self.w2 * p2

    def prediction_for_first_model(self, x: torch.Tensor) -> torch.Tensor:
        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .prediction_for_first_model().")
        return self._predict(self._h1, x).view(-1)

    def prediction_for_second_model(self, x: torch.Tensor) -> torch.Tensor:
        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .prediction_for_second_model().")
        return self._predict(self._h2, x).view(-1)

    def _fit_one(
            self,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_x: torch.Tensor | None = None,
            val_y: torch.Tensor | None = None,
    ) -> LightningModule:
        """Train one model from scratch on ``(x, y)`` and return it (inline, this process).

        Uses the builder path (:func:`run_training_job` + rebuild from the returned CPU
        state dict) when configured with ``module_builders``, otherwise the legacy path
        (:meth:`_train_fun` on a deep-copied template). Equivalent to lines 2 / 8 / 20 of
        the pseudo-code.

        :param model_index: 0 for ``first_model``, 1 for ``second_model``.
        :param x: Training features.
        :param y: Training targets.
        :param val_x: Optional validation features.
        :param val_y: Optional validation targets.
        :return: The trained ``LightningModule``.
        """
        if self._use_builders:
            spec = self._make_fit_spec(model_index, x, y, self._cpu_pair(val_x, val_y))
            # Inline: run in this process with the configured accelerator/devices.
            spec.accelerator = self._inline_accelerator
            spec.devices = self._inline_devices
            result = run_training_job(spec)
            return self._rebuild_module(model_index, result["state_dict"])

        return self._train_fun(
            copy.deepcopy(self.lightning_modules[model_index]),
            model_index,
            x,
            y,
            val_x,
            val_y,
        )

    def _make_fit_spec(
            self,
            model_index: int,
            x: torch.Tensor,
            y: torch.Tensor,
            val_cpu: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> TrainingSpec:
        """Build a picklable :class:`TrainingSpec` for a full (return-state) training.

        Defaults to worker settings (``accelerator="gpu", devices=1``); the inline caller
        overrides these fields for in-process training.

        :param model_index: 0 for ``first_model``, 1 for ``second_model``.
        :param x: Training features.
        :param y: Training targets.
        :param val_cpu: ``(val_x, val_y)`` already moved to CPU (either may be ``None``).
        :return: A self-contained training spec.
        """
        return TrainingSpec(
            module_builder=self.module_builders[model_index],
            initial_state_dict=self._initial_state_dicts[model_index],
            max_epochs=self.max_epochs[model_index],
            patience=self.patiences[model_index],
            batch_size=self.batch_sizes[model_index],
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
        """Rebuild a model in this (main) process from a CPU state dict, for inference only.

        :param model_index: 0 for ``first_model``, 1 for ``second_model``.
        :param state_dict: CPU ``state_dict`` returned by a training job.
        :return: A fresh ``LightningModule`` (on CPU) with the trained weights loaded.
        """
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

    def _apply_cross_label(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            selection: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the *other* model's self-labelled unit to this model's set (line 19).

        :param x: This model's current features.
        :param y: This model's current labels.
        :param selection: ``(unit_id, xu, lu_p)`` from the other model, or ``None``.
        :return: The (possibly) augmented ``(x, y)``.
        """
        if selection is None:
            return x, y
        _, xu, lu = selection
        xu = xu.to(x.device)
        if y.dim() > 1:
            lu_reshaped = lu.view(-1, y.shape[1])
        else:
            lu_reshaped = lu.view(-1)
        lu_reshaped = lu_reshaped.to(y.device)
        return torch.cat([x, xu], dim=0), torch.cat([y, lu_reshaped], dim=0)

    @staticmethod
    def _drop_selected(
            remaining_suspension_ids: torch.Tensor,
            pi: list,
    ) -> torch.Tensor:
        """Remove the units selected this iteration from the global suspension pool."""
        selected_ids = []
        if pi[0] is not None:
            selected_ids.append(pi[0][0].item())
        if pi[1] is not None:
            selected_ids.append(pi[1][0].item())
        for s_id in set(selected_ids):
            remaining_suspension_ids = remaining_suspension_ids[remaining_suspension_ids != s_id]
        return remaining_suspension_ids

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
        TrainFun (legacy style): fit ``model`` on (x, y) with Lightning and return it.

        A fresh Trainer is built from the factory for this model index. If a
        validation set is provided it is used for early stopping / checkpointing.
        After ``trainer.fit`` the best checkpoint (as tracked by the trainer's
        ``ModelCheckpoint`` callback) is reloaded so we never keep the potentially
        worse last-epoch weights.
        """
        # A Trainer keeps internal state, so we always build a fresh one.
        trainer = self.trainer_factories[model_index]()
        batch_size = self.batch_sizes[model_index]

        train_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=self.shuffle_dataloaders[model_index],
        )

        val_loader = None
        if val_x is not None and val_y is not None:
            val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Reload the best checkpoint (based on the monitored validation metric) so we
        # use the best model after training instead of the last-epoch weights.
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", "") if checkpoint_callback else ""
        if best_model_path:
            self._log(2, f"[Coprog]     Reloading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=model.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            self._log(2, f"[Coprog]     No best model find the model with last epoch is used")

        return model

    def _predict(self, model: LightningModule, x: torch.Tensor) -> torch.Tensor:
        """
        Return model predictions (shape (N, *output_dims)) without tracking gradients.
        """
        model.eval()
        with torch.no_grad():
            x = x.to(next(model.parameters()).device)
            return model(x)

    def _summed_squared_error(self, model: LightningModule, x: torch.Tensor, y: torch.Tensor) -> float:
        """Summed squared error of ``model`` on ``(x, y)`` — the MSE term of the confidence measure.

        :param model: A trained ``LightningModule`` (``forward`` returns real-unit predictions).
        :param x: Evaluation features.
        :param y: Evaluation targets.
        :return: ``sum((y - pred) ** 2)`` as a Python float.
        """
        y_flat = y.view(-1).to(next(model.parameters()).device)
        pred = self._predict(model, x).view(-1)
        return ((y_flat - pred) ** 2).sum().item()

    def _confidence_measure(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            original_model: LightningModule,
            augmented_model: LightningModule,
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
        mse_orig = self._summed_squared_error(original_model, x, y)
        mse_aug = self._summed_squared_error(augmented_model, x, y)
        return mse_orig - mse_aug  # > 0 means augmented model is better

    def calculate_weights(
            self,
            x_test: torch.Tensor,
            target: torch.Tensor,
            criteria_callback: Callable[[torch.Tensor, torch.Tensor], float],
            mode: str,
    ):
        """
        Compute the ensemble weights from each model's score on a held-out set.

        Args:
            x_test: Features of the held-out (ideally validation) set.
            target: Labels of the held-out set.
            criteria_callback: Callable returning a scalar score from (prediction, target).
            mode: value can be "min" or "max".
                "min" mean that more the score is little more the model is good.
                "max" mean that more the score is high more the model is good.
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")

        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .calculate_weights().")

        scores = []

        # Flatten both sides so the criteria callback compares aligned (N,) vectors
        # instead of broadcasting (N, 1) against (N,) into an (N, N) matrix.
        target_flat = target.view(-1).float()

        pred_h1 = self._predict(self._h1, x_test).view(-1).to(target_flat.device)
        scores.append(criteria_callback(pred_h1, target_flat))

        pred_h2 = self._predict(self._h2, x_test).view(-1).to(target_flat.device)
        scores.append(criteria_callback(pred_h2, target_flat))

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
