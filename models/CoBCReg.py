import csv
import math
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader, TensorDataset

from models.RBFNetwork import RBFNetwork
from scania.lightning_module.BasicLightningModule import BasicLightningModule


def _make_generator(seed: Optional[int], index: int) -> torch.Generator:
    """Builds a per-committee-member random generator.

    Args:
        seed: Base seed for reproducibility, or ``None`` for a non-deterministic
            generator seeded from OS entropy.
        index: Committee member index, added to ``seed`` so every member gets a
            distinct (but reproducible) stream when ``seed`` is set.

    Returns:
        A ``torch.Generator`` dedicated to this committee member.
    """
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed + index)
    return generator


def _unit_windows(x: torch.Tensor, unit_ids: torch.Tensor, unit_id: int) -> torch.Tensor:
    """Gathers every row belonging to one unit (vehicle).

    Args:
        x: Rows to select from, shape ``(n_rows, ...features)``.
        unit_ids: Unit id per row of ``x``, shape ``(n_rows,)``.
        unit_id: The unit to gather.

    Returns:
        The unit's rows, in their original (assumed chronological) order.
    """
    return x[unit_ids == unit_id]


def _backward_extrapolate_labels(last_label: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Backward-extrapolates a unit's earlier-window labels from its last-window label.

    ``RUL_i = last_label + (t_last - t_i)``, using each window's **real** time
    step rather than its index within the unit — the gap between consecutive
    windows can vary a lot (irregular reporting/sampling), so an index-based
    proxy (assuming evenly-spaced windows) would distort the extrapolated RUL.

    Args:
        last_label: The unit's last-window label, shape ``(1, 1)``.
        time_steps: The unit's per-window time steps, shape ``(n_windows,)``,
            chronologically ordered (last entry = the scored last window).

    Returns:
        Per-window labels, shape ``(n_windows, 1)``, decreasing towards the last window.
    """
    offsets = (time_steps[-1] - time_steps).view(-1, 1).to(dtype=last_label.dtype, device=last_label.device)
    return last_label + offsets


def _label_unit(
        unit_id: int,
        x_unlabeled: torch.Tensor,
        unlabeled_unit_ids: torch.Tensor,
        unlabeled_time_steps: torch.Tensor,
        last_window_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gathers a unit's full window group and labels every window.

    Args:
        unit_id: The unit to gather and label.
        x_unlabeled: All unlabeled windows, shape ``(n_rows, ...features)``.
        unlabeled_unit_ids: Unit id per row of ``x_unlabeled``, shape ``(n_rows,)``.
        unlabeled_time_steps: Time step per row of ``x_unlabeled``, shape ``(n_rows,)``.
        last_window_label: The label to assign to the unit's last window (e.g. the
            committee's averaged prediction), shape ``(1, 1)``.

    Returns:
        A tuple ``(unit_x, unit_y)``: the unit's windows and their per-window
        backward-extrapolated labels, both with ``n_windows`` rows.
    """
    unit_x = _unit_windows(x_unlabeled, unlabeled_unit_ids, unit_id)
    unit_time_steps = _unit_windows(unlabeled_time_steps, unlabeled_unit_ids, unit_id)
    unit_y = _backward_extrapolate_labels(last_window_label, unit_time_steps)
    return unit_x, unit_y


class CoBCReg:
    """CoBC for Regression: co-training by committee with RBF-network regressors.

    Reference: Hady, M. F. A., Schwenker, F., & Palm, G. (2009). "Semi-supervised
    Learning for Regression with Co-training by Committee". The committee is a set
    of :class:`~models.RBFNetwork.RBFNetwork` regressors, each wrapped in
    :class:`~scania.lightning_module.BasicLightningModule.BasicLightningModule` for
    Lightning-based training. Diversity across committee members comes from three
    orthogonal sources: a different bootstrap sample of the labeled set per member,
    a different random initialization of the RBF centers (via a per-member
    ``torch.Generator``), and a different Minkowski distance order ``p`` used inside
    each member's Gaussian basis (``distance_orders``).

    At each co-training iteration, every member samples a fresh pool of **units**
    (vehicles) from the shared unlabeled set and, following
    ``SelectRelevantExamples`` (Algorithm 2 of the paper), retrains a candidate
    RBFNN once per pooled unit to measure how much adding it (all its windows,
    labeled from the other members' averaged prediction at its last window) would
    improve that member's out-of-bag validation error. This makes training
    expensive by design (``pool_size`` candidate retrains per member per
    iteration) — it is a faithful implementation of the paper's algorithm, not an
    optimized approximation.
    """

    def __init__(
            self,
            distance_orders: list[float],
            n_centers: int,
            width_scale: float = 1.0,
            width_neighbors: int = 2,
            trainable_centers: bool = True,
            trainable_widths: bool = True,
            max_iterations: int = 10,
            pool_size: int = 50,
            growth_rate: int = 1,
            lr: float = 1e-3,
            max_epochs: int = 100,
            patience: int = 10,
            batch_size: int = 32,
            accelerator: str = "auto",
            devices: int = 1,
            seed: Optional[int] = None,
            verbose: int = 0,
    ) -> None:
        """Initializes the CoBCReg committee.

        Args:
            distance_orders: Minkowski distance order ``p_i`` for each committee
                member. Its length sets the committee size ``N`` (must be >= 2, so
                "the other models' average prediction" is defined).
            n_centers: Number of RBF hidden units ``k``, shared by every member.
            width_scale: RBF width parameter ``alpha`` (multiplicative scale applied
                to each member's heuristic width init), shared by every member.
            width_neighbors: Number of nearest neighboring centers averaged over
                when initializing RBF widths (see :class:`~models.RBFNetwork.RBFNetwork`).
            trainable_centers: Whether RBF centers are updated by backprop during
                training, for every member.
            trainable_widths: Whether RBF widths are updated by backprop during
                training, for every member.
            max_iterations: Maximum number of co-training iterations ``T``.
            pool_size: Number of unlabeled units sampled per member per
                iteration ``u``.
            growth_rate: Maximum number of units a member can add to its labeled
                bag per iteration ``gr`` (each unit contributes all of its windows).
            lr: Learning rate used to train every RBFNN (main and candidate fits).
            max_epochs: Maximum training epochs per RBFNN fit.
            patience: ``EarlyStopping`` patience (monitored on ``val_loss``) per fit.
            batch_size: Batch size used for both the train and validation dataloaders
                of every RBFNN fit.
            accelerator: Lightning accelerator passed to every ``Trainer``.
            devices: Lightning devices passed to every ``Trainer``.
            seed: Optional base seed. When set, committee member ``i`` gets a
                reproducible generator seeded with ``seed + i``; otherwise each
                member's generator is seeded from OS entropy.
            verbose: Verbosity level. ``0`` = silent, ``1`` = key decisions,
                ``2`` = full per-candidate details.

        Raises:
            ValueError: If ``distance_orders`` has fewer than 2 entries, or if
                ``n_centers``, ``pool_size``, ``growth_rate``, ``max_iterations``,
                ``max_epochs`` or ``batch_size`` is not a positive integer.
        """
        if len(distance_orders) < 2:
            raise ValueError("distance_orders must have at least 2 entries (a committee needs >= 2 members).")
        if n_centers < 1:
            raise ValueError("n_centers must be >= 1.")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1.")
        if growth_rate < 1:
            raise ValueError("growth_rate must be >= 1.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if max_epochs < 1:
            raise ValueError("max_epochs must be >= 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.distance_orders = distance_orders
        self.n_models = len(distance_orders)
        self.n_centers = n_centers
        self.width_scale = width_scale
        self.width_neighbors = width_neighbors
        self.trainable_centers = trainable_centers
        self.trainable_widths = trainable_widths
        self.max_iterations = max_iterations
        self.pool_size = pool_size
        self.growth_rate = growth_rate
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.devices = devices
        self.verbose = verbose

        self._generators = [_make_generator(seed, i) for i in range(self.n_models)]
        self._log_file_path: Optional[str] = None

        self.n_features: Optional[int] = None
        self.labeled_bags: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self.validation_sets: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self.models: Optional[list[BasicLightningModule]] = None
        self.validation_errors: Optional[list[float]] = None
        self.weights: Optional[list[float]] = None

    def _log(self, level: int, message: str) -> None:
        """Prints a message if verbose allows it, and always appends it to the log file if set.

        Args:
            level: Minimum verbosity level required to print ``message``.
            message: The message to print / append.
        """
        if self.verbose >= level:
            print(message)
        if self._log_file_path is not None:
            with open(self._log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")

    def _bootstrap_sample(self, row_number: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Draws a bootstrap bag and its out-of-bag complement from ``row_number`` examples.

        Standard bagging: ``torch.randint`` draws each of the ``row_number`` bag
        indices independently, so rows are sampled **with replacement** (expect
        duplicates, and roughly 1/e of rows never drawn). This is unrelated to the
        unlabeled-pool sampling in :meth:`train`, which is a plain, intentionally
        **not**-bootstrapped subsample.

        Args:
            row_number: Number of labeled examples to sample from.
            generator: Random generator controlling the sampling.

        Returns:
            A tuple ``(bag_indices, out_of_bag_indices)``, indices into ``0..row_number-1``.
        """
        bag_indices = torch.randint(0, row_number, (row_number,), generator=generator)
        in_bag = torch.zeros(row_number, dtype=torch.bool)
        in_bag[bag_indices] = True
        out_of_bag_indices = torch.nonzero(~in_bag, as_tuple=True)[0]
        return bag_indices, out_of_bag_indices

    def _fit_rbfnn(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            x_val: torch.Tensor,
            y_val: torch.Tensor,
            model_idx: int,
    ) -> BasicLightningModule:
        """Trains a fresh RBFNN (via Lightning) for one committee member.

        Args:
            x: Training data, shape ``(n_train, ...features)`` (flat vectors or
                sequences — see :class:`~models.RBFNetwork.RBFNetwork`).
            y: Training targets of shape ``(n_train, 1)``.
            x_val: Validation data, shape ``(n_val, ...features)``.
            y_val: Validation targets of shape ``(n_val, 1)``.
            model_idx: Index of the committee member this RBFNN belongs to (selects
                its distance order and random generator).

        Returns:
            The trained :class:`BasicLightningModule` wrapping the new RBFNN.
        """
        net = RBFNetwork(
            in_features=self.n_features,
            n_centers=self.n_centers,
            distance_order=self.distance_orders[model_idx],
            trainable_centers=self.trainable_centers,
            trainable_widths=self.trainable_widths,
            width_neighbors=self.width_neighbors,
            width_scale=self.width_scale,
            init_data=x,
            generator=self._generators[model_idx],
        )
        module = BasicLightningModule(lr=self.lr, model=net)

        train_loader = DataLoader(
            TensorDataset(x, y), batch_size=min(self.batch_size, x.shape[0]), shuffle=True)
        val_loader = DataLoader(
            TensorDataset(x_val, y_val), batch_size=min(self.batch_size, x_val.shape[0]), shuffle=False)

        early_stop_callback = callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.patience,
            verbose=False,
            mode="min",
        )
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=[early_stop_callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return module

    def _predict_model(self, module: BasicLightningModule, x: torch.Tensor) -> torch.Tensor:
        """Runs a single committee member's forward pass in eval mode, without grad.

        Args:
            module: The committee member to run.
            x: Input data, shape ``(batch, ...features)``.

        Returns:
            Predictions of shape ``(batch, 1)``.
        """
        module.eval()
        with torch.no_grad():
            return module(x)

    def _validation_error(
            self,
            module: BasicLightningModule,
            x_val: torch.Tensor,
            y_val: torch.Tensor,
    ) -> float:
        """Computes a model's mean squared error on a validation set.

        Args:
            module: The trained committee member to evaluate.
            x_val: Validation data, shape ``(n_val, ...features)``.
            y_val: Validation targets of shape ``(n_val, 1)``.

        Returns:
            The scalar MSE.
        """
        predictions = self._predict_model(module, x_val).view(-1)
        return F.mse_loss(predictions, y_val.view(-1)).item()

    def _select_relevant_examples(
            self,
            model_idx: int,
            pool_unit_ids: torch.Tensor,
            x_unlabeled: torch.Tensor,
            unlabeled_unit_ids: torch.Tensor,
            unlabeled_time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements Algorithm 2 (``SelectRelevantExamples``) for one committee member.

        Operates at unit granularity: each pooled unit is scored via its **last**
        window (matching the convention already used by ``CoTrainingEnsemble_v2``),
        but the retrain probe used to compute ``Delta`` adds the unit's **entire**
        window group (backward-extrapolated from that last-window prediction using
        real per-window time steps), since that is what actually happens if the
        unit ends up selected.

        Args:
            model_idx: Index ``j`` of the committee member excluded from the
                averaged prediction (the one being evaluated for growth).
            pool_unit_ids: Unit ids sampled for this iteration's pool, shape ``(pool_size,)``.
            x_unlabeled: All unlabeled windows, shape ``(n_rows, ...features)``.
            unlabeled_unit_ids: Unit id per row of ``x_unlabeled``, shape ``(n_rows,)``.
            unlabeled_time_steps: Time step per row of ``x_unlabeled``, shape ``(n_rows,)``.

        Returns:
            A tuple ``(selected_unit_ids, selected_last_window_labels,
            selected_positions)``: the selected units' ids, the committee-averaged
            prediction at each selected unit's last window (used to label its full
            window group), and their positions within ``pool_unit_ids``. All three
            are empty (0-length) when no unit has positive ``Delta``.
        """
        x_val, y_val = self.validation_sets[model_idx]
        epsilon = self._validation_error(self.models[model_idx], x_val, y_val)

        last_windows = torch.stack(
            [_unit_windows(x_unlabeled, unlabeled_unit_ids, uid.item())[-1] for uid in pool_unit_ids],
            dim=0,
        )
        other_predictions = torch.stack(
            [self._predict_model(self.models[k], last_windows) for k in range(self.n_models) if k != model_idx],
            dim=0,
        )
        committee_prediction = other_predictions.mean(dim=0)

        x_bag, y_bag = self.labeled_bags[model_idx]
        deltas = torch.empty(pool_unit_ids.shape[0])
        for u in range(pool_unit_ids.shape[0]):
            unit_x, unit_y = _label_unit(
                pool_unit_ids[u].item(), x_unlabeled, unlabeled_unit_ids, unlabeled_time_steps,
                committee_prediction[u:u + 1])
            candidate_x = torch.cat([x_bag, unit_x], dim=0)
            candidate_y = torch.cat([y_bag, unit_y], dim=0)
            candidate_module = self._fit_rbfnn(candidate_x, candidate_y, x_val, y_val, model_idx)
            epsilon_prime = self._validation_error(candidate_module, x_val, y_val)
            deltas[u] = (epsilon - epsilon_prime) / epsilon

        positive_positions = torch.nonzero(deltas > 0, as_tuple=True)[0]
        if positive_positions.numel() == 0:
            empty_ids = pool_unit_ids.new_empty(0)
            empty_labels = committee_prediction.new_empty((0, committee_prediction.shape[1]))
            return empty_ids, empty_labels, torch.empty(0, dtype=torch.long)

        order = torch.argsort(deltas[positive_positions], descending=True)
        top_positions = positive_positions[order][: self.growth_rate]

        self._log(
            2,
            f"[CoBCReg] model {model_idx} | epsilon={epsilon:.4f} | "
            f"selected {top_positions.numel()} unit(s), deltas={deltas[top_positions].tolist()}",
        )

        return pool_unit_ids[top_positions], committee_prediction[top_positions], top_positions

    def _compute_weights(self, validation_errors: list[float]) -> list[float]:
        """Turns per-model validation errors into normalized, inverse-error weights.

        The paper's prediction phase (``H(x) = sum_i w_i h_i(x)``) does not specify
        how ``w_i`` is derived; this uses each member's out-of-bag validation MSE,
        inverse-weighted and normalized to sum to 1, so more accurate members
        contribute more to the ensemble prediction.

        Args:
            validation_errors: Per-model validation MSE.

        Returns:
            Normalized weights, one per model, summing to 1.
        """
        if any(error == 0.0 for error in validation_errors):
            return [1.0 if error == 0.0 else 0.0 for error in validation_errors]
        inverse_errors = [1.0 / error for error in validation_errors]
        total = sum(inverse_errors)
        return [inverse_error / total for inverse_error in inverse_errors]

    def _log_stage_metrics(
            self,
            stage: str,
            test_data: torch.Tensor,
            test_label: torch.Tensor,
            score_callback: Callable[[torch.Tensor, torch.Tensor], float],
            metrics_file: str,
    ) -> None:
        """Appends one row of per-model + averaged/weighted metrics to a CSV file.

        Mirrors ``CoTrainingEnsemble_v2``'s per-stage metrics CSV: a header row is
        written only if ``metrics_file`` doesn't already exist, then one data row
        per call (append mode).

        Args:
            stage: Label for this row (e.g. ``"initial"``, ``"iteration_2"``, ``"final"``).
            test_data: Held-out test features.
            test_label: Held-out test targets.
            score_callback: Computes a domain score from ``(predictions, targets)``,
                e.g. the C-MAPSS score.
            metrics_file: CSV path to append to.
        """
        test_label_flat = test_label.view(-1).float()

        train_rmses, val_rmses, test_rmses, test_scores = [], [], [], []
        for i in range(self.n_models):
            x_bag, y_bag = self.labeled_bags[i]
            x_val, y_val = self.validation_sets[i]

            train_pred = self._predict_model(self.models[i], x_bag).view(-1)
            train_rmses.append(F.mse_loss(train_pred, y_bag.view(-1)).sqrt().item())
            val_rmses.append(self._validation_error(self.models[i], x_val, y_val) ** 0.5)

            test_pred = self._predict_model(self.models[i], test_data).view(-1)
            test_rmses.append(F.mse_loss(test_pred, test_label_flat).sqrt().item())
            test_scores.append(score_callback(test_pred, test_label_flat))

        avg_test_rmse = sum(test_rmses) / self.n_models
        avg_test_score = sum(test_scores) / self.n_models

        weights = self._compute_weights([val_rmse ** 2 for val_rmse in val_rmses])
        weighted_test_rmse = sum(weight * rmse for weight, rmse in zip(weights, test_rmses))
        weighted_test_score = sum(weight * score for weight, score in zip(weights, test_scores))

        columns = ["stage"]
        row: list = [stage]
        for i in range(self.n_models):
            columns += [f"train_rmse_{i}", f"val_rmse_{i}", f"test_rmse_{i}", f"test_score_{i}"]
            row += [train_rmses[i], val_rmses[i], test_rmses[i], test_scores[i]]
        columns += ["avg_test_rmse", "avg_test_score", "weighted_test_rmse", "weighted_test_score"]
        row += [avg_test_rmse, avg_test_score, weighted_test_rmse, weighted_test_score]
        for i in range(self.n_models):
            columns += [f"weight_{i}"]
            row += [weights[i]]

        write_header = not os.path.exists(metrics_file)
        with open(metrics_file, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(columns)
            writer.writerow(row)

    def train(
            self,
            x_labeled: torch.Tensor,
            y_labeled: torch.Tensor,
            x_unlabeled: torch.Tensor,
            unlabeled_unit_ids: torch.Tensor,
            unlabeled_time_steps: torch.Tensor,
            log_file: Optional[str] = None,
            test_data: Optional[torch.Tensor] = None,
            test_label: Optional[torch.Tensor] = None,
            score_callback: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
            metrics_file: Optional[str] = None,
    ) -> None:
        """Trains the committee following Algorithm 1 (CoBC for Regression).

        Unlabeled data is unit-based: ``x_unlabeled`` holds every window of every
        censored unit (vehicle), and ``unlabeled_unit_ids`` carries each row's unit
        id (mirroring how ``ScaniaDataset``/``CoTrainingEnsemble_v2`` carry
        ``vehicle_id`` as a tensor parallel to the features, never embedded in
        them). Rows of the same unit must be contiguous and in chronological order
        (its last row is that unit's most recent window) — the same precondition
        ``CoTrainingEnsemble_v2`` relies on. When a unit is selected, **all** of its
        windows are added to the bag together: the unit's last window is scored
        and given the committee's averaged prediction as its label, and earlier
        windows are backward-extrapolated using their **real** time steps
        (``RUL_i = RUL_last + (t_last - t_i)``, via ``unlabeled_time_steps`` —
        not window position, since the gap between windows can vary a lot).
        ``growth_rate`` therefore counts **units** selected per iteration, not
        individual rows.

        Two independent, optional logging outputs are available, matching
        ``Coprog``/``CoTrainingEnsemble``/``CoTrainingEnsemble_v2``: a free-text
        ``log_file`` (every internal log line is appended to it, regardless of
        ``verbose``), and a per-stage metrics ``metrics_file`` CSV (one row at
        ``"initial"``, after each iteration, and ``"final"``) enabled by passing
        ``test_data``.

        Note on sampling: the labeled bootstrap bag samples **with replacement**
        (standard bagging, see :meth:`_bootstrap_sample`). The per-iteration
        unlabeled **pool** is a separate, intentionally-not-bootstrapped subsample
        **without** replacement — the paper's "create a pool U' of u examples by
        random sampling from U" — so a unit isn't scored twice within the same pool.

        Args:
            x_labeled: Labeled training data, shape ``(row_number, ...features)``.
            y_labeled: Labeled training targets of shape ``(row_number,)`` or ``(row_number, 1)``.
            x_unlabeled: All unlabeled/censored windows, shape ``(n_rows, ...features)``.
            unlabeled_unit_ids: Unit id per row of ``x_unlabeled``, shape ``(n_rows,)``.
            unlabeled_time_steps: Time step per row of ``x_unlabeled``, shape
                ``(n_rows,)``, used for backward-extrapolating earlier windows of
                a selected unit from its scored last window.
            log_file: Optional path; every internal log line is appended to it.
            test_data: Optional held-out test features. Passing this enables the
                per-stage metrics CSV (``test_label``, ``score_callback`` and
                ``metrics_file`` then become required).
            test_label: Held-out test targets, required if ``test_data`` is given.
            score_callback: Computes a domain score from ``(predictions, targets)``
                (e.g. the C-MAPSS score), required if ``test_data`` is given.
            metrics_file: CSV path to append per-stage metrics to, required if
                ``test_data`` is given.

        Raises:
            ValueError: If ``x_labeled`` has fewer rows than ``n_centers``; if
                ``x_unlabeled`` and ``unlabeled_unit_ids`` have mismatched row
                counts; if a bootstrap sample leaves a committee member with an
                empty out-of-bag validation set; or if ``test_data`` is given
                without ``test_label``, ``score_callback`` and ``metrics_file``.
        """
        self._log_file_path = log_file

        metrics_enabled = test_data is not None
        if metrics_enabled and (test_label is None or score_callback is None or metrics_file is None):
            raise ValueError("test_label, score_callback and metrics_file are required when test_data is given.")

        x_labeled = x_labeled.float()
        y_labeled = y_labeled.view(-1, 1).float()
        x_unlabeled = x_unlabeled.float()
        unlabeled_unit_ids = unlabeled_unit_ids.long()
        unlabeled_time_steps = unlabeled_time_steps.float()

        if x_unlabeled.shape[0] != unlabeled_unit_ids.shape[0] or x_unlabeled.shape[0] != unlabeled_time_steps.shape[0]:
            raise ValueError(
                f"x_unlabeled has {x_unlabeled.shape[0]} rows, unlabeled_unit_ids has "
                f"{unlabeled_unit_ids.shape[0]}, and unlabeled_time_steps has "
                f"{unlabeled_time_steps.shape[0]}; they must all be aligned 1-to-1.")

        row_number = x_labeled.shape[0]
        if row_number < self.n_centers:
            raise ValueError(f"x_labeled has {row_number} rows, need at least n_centers={self.n_centers}.")

        self.n_features = math.prod(x_labeled.shape[1:])
        self.labeled_bags = []
        self.validation_sets = []
        self.models = []

        for i in range(self.n_models):
            bag_indices, out_of_bag_indices = self._bootstrap_sample(row_number, self._generators[i])
            if out_of_bag_indices.numel() == 0:
                raise ValueError(
                    f"Bootstrap sample for model {i} left an empty out-of-bag validation set; "
                    f"increase the labeled set size.")
            x_bag, y_bag = x_labeled[bag_indices], y_labeled[bag_indices]
            x_val, y_val = x_labeled[out_of_bag_indices], y_labeled[out_of_bag_indices]
            self.labeled_bags.append((x_bag, y_bag))
            self.validation_sets.append((x_val, y_val))
            self.models.append(self._fit_rbfnn(x_bag, y_bag, x_val, y_val, i))

        if metrics_enabled:
            self._log_stage_metrics("initial", test_data, test_label, score_callback, metrics_file)

        available_units = torch.unique(unlabeled_unit_ids)
        unit_available_mask = torch.ones(available_units.shape[0], dtype=torch.bool)

        for t in range(self.max_iterations):
            if not unit_available_mask.any():
                self._log(1, f"[CoBCReg] Unlabeled pool exhausted after {t} iteration(s); stopping early.")
                break

            selections: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * self.n_models
            for i in range(self.n_models):
                available_positions = torch.nonzero(unit_available_mask, as_tuple=True)[0]
                if available_positions.numel() == 0:
                    continue
                pool_n = min(self.pool_size, available_positions.numel())
                perm = torch.randperm(available_positions.numel(), generator=self._generators[i])[:pool_n]
                pool_positions = available_positions[perm]
                pool_unit_ids = available_units[pool_positions]

                selected_unit_ids, selected_labels, selected_positions = self._select_relevant_examples(
                    i, pool_unit_ids, x_unlabeled, unlabeled_unit_ids, unlabeled_time_steps)
                if selected_positions.numel() > 0:
                    unit_x_parts, unit_y_parts = [], []
                    for uid, label in zip(selected_unit_ids.tolist(), selected_labels):
                        unit_x, unit_y = _label_unit(
                            uid, x_unlabeled, unlabeled_unit_ids, unlabeled_time_steps, label.view(1, 1))
                        unit_x_parts.append(unit_x)
                        unit_y_parts.append(unit_y)
                    selections[i] = (torch.cat(unit_x_parts, dim=0), torch.cat(unit_y_parts, dim=0))
                    unit_available_mask[pool_positions[selected_positions]] = False

            for i in range(self.n_models):
                if selections[i] is None:
                    continue
                selected_x, selected_y = selections[i]
                x_bag, y_bag = self.labeled_bags[i]
                x_bag = torch.cat([x_bag, selected_x], dim=0)
                y_bag = torch.cat([y_bag, selected_y], dim=0)
                self.labeled_bags[i] = (x_bag, y_bag)
                x_val, y_val = self.validation_sets[i]
                self.models[i] = self._fit_rbfnn(x_bag, y_bag, x_val, y_val, i)

            self._log(1, f"[CoBCReg] Iteration {t + 1}/{self.max_iterations} done | "
                         f"added {[0 if s is None else s[0].shape[0] for s in selections]} window(s) per model.")

            if metrics_enabled:
                self._log_stage_metrics(f"iteration_{t + 1}", test_data, test_label, score_callback, metrics_file)

        self.validation_errors = [
            self._validation_error(self.models[i], *self.validation_sets[i]) for i in range(self.n_models)
        ]
        self.weights = self._compute_weights(self.validation_errors)
        self._log(1, f"[CoBCReg] Final validation errors: {self.validation_errors} | weights: {self.weights}")

        if metrics_enabled:
            self._log_stage_metrics("final", test_data, test_label, score_callback, metrics_file)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts with the weighted committee: ``H(x) = sum_i w_i h_i(x)``.

        Args:
            x: Input data, shape ``(batch, ...features)``.

        Returns:
            Weighted ensemble predictions of shape ``(batch, 1)``.

        Raises:
            ValueError: If called before :meth:`train`.
        """
        if self.models is None or self.weights is None:
            raise ValueError("CoBCReg must be trained via train() before calling predict().")

        x = x.float()
        predictions = torch.stack([self._predict_model(model, x) for model in self.models], dim=0)
        weights = torch.tensor(self.weights, dtype=predictions.dtype, device=predictions.device).view(-1, 1, 1)
        return (predictions * weights).sum(dim=0)
