from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F

from models.CoBCReg import CoBCReg
from scania.dataset import ScaniaDataModule
from scania.utils.utils_scania import (
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
    save_train_parameters,
    generate_and_save_model_prediction,
    _scania_score,
)
from shared.utils import ModelVersion


def train_model(
    checkpoints_path: str,
    results_path: str,
    model_version: ModelVersion,
    # Model params
    distance_orders: list[float],
    n_centers: int,
    width_scale: float,
    width_neighbors: int,
    trainable_centers: bool,
    trainable_widths: bool,
    # Dataset params
    dataset_root: str,
    sequence_len: int,
    seed: int | None,
    data_fraction: float,
    val_rate: float,
    test_rate: float,
    stratify: bool,
    norm_type: str | None,
    shuffle_loader: bool,
    cache_dir: str | None,
    num_workers: int,
    pin_memory: bool,
    return_sequence_label: bool,
    batch_size: int,
    counter_mode: str,
    include_histograms: bool,
    histogram_mode: str,
    # Training params
    lr: float,
    max_epochs: int,
    patience: int,
    max_iterations: int,
    pool_size: int,
    growth_rate: int,
    rul_target_standardization: bool,
    # Others
    gpu_ids: list[int] | None = None,
    datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train a CoBCReg committee of RBF networks on the Scania dataset.

    Mirrors :func:`scania.utils.utils_coprog.train_model` but for CoBCReg: the committee
    builds and trains its own RBF networks internally (no ``setup_training_builder``
    handshake). Censored (suspension) data is fed unit-aware — every censored window plus
    a parallel vehicle-id tensor and a per-window time reference — and CoBCReg self-labels
    whole units, backward-extrapolating each unit's earlier windows.

    Args:
        checkpoints_path: Directory for the per-committee-member ``.pth`` files.
        results_path: Directory for prediction/score CSVs, the run log and per-stage metrics.
        model_version: The model version (``ModelVersion.COBCREG``), used for file naming.
        distance_orders: Minkowski distance order ``p`` per committee member; its length
            sets the committee size.
        n_centers: Number of RBF hidden units per member.
        width_scale: RBF width scale (``alpha``).
        width_neighbors: Nearest neighbours averaged over when initializing RBF widths.
        trainable_centers: Whether RBF centers are learned by backprop.
        trainable_widths: Whether RBF widths are learned by backprop.
        dataset_root: Root directory of the Scania data files.
        sequence_len: Window length.
        seed: Base RNG seed (shared by the data module and the committee generators).
        data_fraction: Fraction of the data to use.
        val_rate: Validation split rate (unused by CoBCReg's OOB validation, but the data
            module still builds the split).
        test_rate: Test split rate.
        stratify: Whether to stratify the vehicle-level split.
        norm_type: Feature normalization type (e.g. ``"z-score"``) or ``None``.
        shuffle_loader: Whether the data module shuffles its loaders.
        cache_dir: Processed-split cache directory (injected from the CLI), or ``None``.
        num_workers: Dataloader worker count.
        pin_memory: Whether dataloaders pin memory.
        return_sequence_label: Whether the dataset returns per-step labels.
        batch_size: Batch size (shared by the data module and each RBFNN fit).
        counter_mode: Scania counter feature mode.
        include_histograms: Whether histogram features are included.
        histogram_mode: Histogram aggregation mode.
        lr: Learning rate for every RBFNN fit.
        max_epochs: Maximum epochs per RBFNN fit.
        patience: Early-stopping patience per RBFNN fit.
        max_iterations: Maximum co-training iterations ``T``.
        pool_size: Number of unlabeled units sampled per member per iteration ``u``.
        growth_rate: Maximum units a member adds to its bag per iteration ``gr``.
        rul_target_standardization: Whether each RBFNN learns in standardized RUL space.
        gpu_ids: GPU id(s) to train on. ``None`` → auto (single GPU/CPU); a list pins to
            those device ids (CoBCReg trains sequentially, so extra ids are not used in
            parallel — the list is forwarded to Lightning's ``devices``).
        datetime_for_folders: Timestamp used to name the run's output folders.

    Returns:
        The weighted-ensemble ``(rmse, score)`` on the test set.
    """
    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
    )

    checkpoints_path, results_path = create_and_get_checkpoints_results_path(
        model_version=model_version.value,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    if len(distance_orders) < 2:
        raise ValueError(
            f"distance_orders must have at least 2 entries (a committee needs >= 2 members), "
            f"got {distance_orders!r}.")

    dataset_kwargs = {
        'data_dir': dataset_root,
        'seed': seed,
        'data_fraction': data_fraction,
        'val_rate': val_rate,
        'test_rate': test_rate,
        'stratify': stratify,
        'norm_type': norm_type,
        'shuffle_loader': shuffle_loader,
        'cache_dir': cache_dir,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'return_sequence_label': return_sequence_label,
        'batch_size': batch_size,
        'sequence_len': sequence_len,
        'counter_mode': counter_mode,
        'include_histograms': include_histograms,
        'histogram_mode': histogram_mode,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    scania_data_module = ScaniaDataModule(**dataset_kwargs)
    scania_data_module.setup()

    feature_num = len(scania_data_module.feature_cols)

    features_uncensored, targets_uncensored, features_censored, ids_censored = \
        scania_data_module.get_cotraining_tensors("train")
    # Per-censored-window time reference for backward extrapolation. The dataset exposes
    # ``lower_bound = length_of_study - time_step`` (evaluated at each window's last row),
    # already row-aligned with features_censored/ids_censored. CoBCReg only uses time
    # DIFFERENCES within a unit, and t_last - t_i == lb_i - lb_last exactly (length_of_study
    # is constant per vehicle), so ``-lower_bound`` is a valid increasing time axis (equal to
    # the real time_step up to a per-vehicle constant that cancels in within-unit differences).
    _, _, lower_bounds_censored = scania_data_module.get_censored_lower_bounds("train")
    time_steps_censored = -lower_bounds_censored.float()

    test_features, test_targets, _, _ = scania_data_module.get_cotraining_tensors("test")

    # gpu_ids (from --gpu-ids): None -> auto (single GPU/CPU); a list pins to those device ids.
    if gpu_ids:
        accelerator = "gpu"
        devices: int | list[int] = gpu_ids
    else:
        accelerator = "auto"
        devices = 1
    print(f"CoBCReg GPU selection: {gpu_ids if gpu_ids else 'auto (single GPU/CPU)'}")

    training_kwargs = {
        "lr": lr,
        "max_epochs": max_epochs,
        "patience": patience,
        "max_iterations": max_iterations,
        "pool_size": pool_size,
        "growth_rate": growth_rate,
        "rul_target_standardization": rul_target_standardization,
    }

    model_kwargs = {
        "distance_orders": distance_orders,
        "n_centers": n_centers,
        "width_scale": width_scale,
        "width_neighbors": width_neighbors,
        "trainable_centers": trainable_centers,
        "trainable_widths": trainable_widths,
    }

    save_train_parameters(
        results_path=results_path,
        dataset_parameters=dataset_kwargs,
        training_parameters=training_kwargs,
        model_parameters=model_kwargs,
    )

    model = CoBCReg(
        distance_orders=distance_orders,
        n_centers=n_centers,
        width_scale=width_scale,
        width_neighbors=width_neighbors,
        trainable_centers=trainable_centers,
        trainable_widths=trainable_widths,
        max_iterations=max_iterations,
        pool_size=pool_size,
        growth_rate=growth_rate,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        rul_target_standardization=rul_target_standardization,
        accelerator=accelerator,
        devices=devices,
        seed=seed,
        verbose=1,
    )

    print("Training CoBCReg model...")

    model.train(
        x_labeled=features_uncensored,
        y_labeled=targets_uncensored,
        x_unlabeled=features_censored,
        unlabeled_unit_ids=ids_censored,
        unlabeled_time_steps=time_steps_censored,
        log_file=f"{results_path}/{model_version.value}_run.txt",
        # Per-stage metrics (initial / iteration_k / final). Score columns use the Scania
        # score, consistent with generate_and_save_model_prediction below.
        test_data=test_features,
        test_label=test_targets,
        score_callback=_score_callback_for_cobcreg,
        metrics_file=f"{results_path}/{model_version.value}-per-stage-scania.csv",
    )

    number_of_models = len(model.models)

    print("Saving trained committee members...")
    for i in range(number_of_models):
        torch.save(model.models[i], f"{checkpoints_path}/{model_version.value}_h{i}.pth")

    targets_flat = test_targets.detach().cpu().view(-1)

    # Weighted-ensemble prediction.
    weighted_predictions = model.predict(test_features).detach().cpu().view(-1)
    rmse_weighted, score_weighted = generate_and_save_model_prediction(
        predictions=weighted_predictions,
        targets=targets_flat,
        model_version=model_version.value,
        prediction_type="test_weighted",
        results_path=results_path,
    )
    print(f"Weighted ensemble | Test RMSE: {rmse_weighted} | Score: {score_weighted}")

    # Per-member (unweighted) predictions.
    per_model_predictions = model.predict_per_model(test_features)
    per_model_rmse: list[float] = []
    per_model_score: list[float] = []
    for i in range(number_of_models):
        prediction_i = per_model_predictions[i].detach().cpu().view(-1)
        rmse_i, score_i = generate_and_save_model_prediction(
            predictions=prediction_i,
            targets=targets_flat,
            model_version=model_version.value,
            prediction_type=f"test_h{i}",
            results_path=results_path,
        )
        per_model_rmse.append(rmse_i)
        per_model_score.append(score_i)

    # Summary scores table with dynamic per-member columns.
    columns: list[str] = []
    row: list[float] = []
    for i in range(number_of_models):
        columns += [f"test_rmse_h{i}", f"test_score_h{i}"]
        row += [per_model_rmse[i], per_model_score[i]]
    columns += ["test_rmse_weighted", "test_score_weighted"]
    row += [rmse_weighted, score_weighted]
    for i in range(number_of_models):
        columns += [f"weight_h{i}"]
        row += [model.weights[i]]

    scores = pd.DataFrame(columns=columns)
    scores.loc[0] = row
    scores.to_csv(f"{results_path}/{model_version.value}-scania.csv", index=False)

    return rmse_weighted, score_weighted


def _score_callback_for_cobcreg(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Scania score (a1=13, a2=10) for a prediction/target tensor pair.

    Wraps the numpy-based :func:`_scania_score` so it can be used as CoBCReg's
    ``score_callback`` for the per-stage metrics (the ``test_score`` columns), matching the
    score reported by :func:`generate_and_save_model_prediction`.

    Args:
        predictions: Predicted RUL, shape ``(N,)``.
        targets: True RUL, shape ``(N,)``.

    Returns:
        The Scania score as a Python float.
    """
    return _scania_score(
        predictions.detach().cpu().numpy().flatten(),
        targets.detach().cpu().numpy().flatten(),
    )
