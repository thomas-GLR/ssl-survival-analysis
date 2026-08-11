"""Score a trained Scania run on the official Component X held-out test set.

Every ``train_model`` in :mod:`scania.utils` reports RMSE and the Scania cost on an *internal*
test split -- a slice of ``train_operational_readouts.csv`` vehicles held out by
``ScaniaDataModule``. That is not the challenge number. This module adds the challenge number:
after training, the run predicts on ``test_operational_readouts.csv`` and is scored against
``test_labels.csv``.

The ground truth there is a **failure-urgency class**, not an RUL, so the models' predicted RUL is
binned with :func:`scania.metrics.rul_to_class` -- verified to be the official binning, since
applying it to ``length_of_study_time_step - max(time_step)`` with censored vehicles forced to
class 0 reproduces ``train_label.csv`` for all 23550 training vehicles. The three reported metrics
are accuracy, macro-F1 and the Scania cost (:func:`scania.metrics.scania_classification_metrics`).

Two entry points, because the models consume the data in two shapes:

- :func:`run_challenge_evaluation` for the window models (the supervised Lightning models, COPROG
  and the co-training ensembles), which predict from ``(N, sequence_len, n_features)`` windows;
- :func:`run_challenge_evaluation_rowwise` for RSF, which treats every readout row as an
  individual and integrates a survival function.

Both take a ``split`` argument (default ``"test"``) so the same run is additionally scored
against a second, disjoint held-out population: ``validation_operational_readouts.csv`` /
``validation_labels.csv``, via ``split="validation"``. The two splits never share vehicles, are
scored identically, and write to distinct file names (see :func:`_score_and_save`).

Both write the same two artifacts into the run's results folder (per split) and both are
**failure-tolerant**:
they run after every other artifact of the run is already on disk, so a missing challenge CSV or a
bug here degrades to a printed warning instead of destroying a finished multi-hour training run.

This module lives directly under ``scania/`` rather than in ``scania/utils/`` for the same reason
:mod:`scania.metrics` does: ``scania/utils/__init__.py`` imports the training orchestrators, which
import ``scania.lightning_module``, so importing it from there would be a circular import.
"""

import os
import traceback
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch

from constants.scania_component_x_columns import VEHICLE_ID
from scania.dataset import ScaniaDataModule
from scania.metrics import rul_to_class, scania_classification_metrics

# The official held-out labels: one failure-urgency class per vehicle, no time-to-event.
CHALLENGE_LABELS_FILE = "test_labels.csv"
# A second, disjoint held-out population, scored the same way. See the module docstring.
VALIDATION_LABELS_FILE = "validation_labels.csv"
CLASS_LABEL = "class_label"

# Column names of the two artifacts this module writes.
PREDICTION_COLUMNS = ["vehicle_id", "target_class", "predicted_class", "predicted_rul"]
METRIC_COLUMNS = ["model", "n_vehicles", "accuracy", "f1_macro", "scania_score"]

_LOG_PREFIX = "[Scania][challenge]"


def load_challenge_labels(
        data_dir: str, split: Literal["test", "validation"] = "test") -> pd.DataFrame:
    """Read the official held-out labels.

    Args:
        data_dir: Directory holding the Scania data files.
        split: Which held-out population to read labels for -- ``"test"`` (default) or
            ``"validation"``.

    Returns:
        A frame with ``vehicle_id`` and ``class_label`` columns.

    Raises:
        FileNotFoundError: If the matching labels file is not in ``data_dir``.
    """
    labels_file = CHALLENGE_LABELS_FILE if split == "test" else VALIDATION_LABELS_FILE
    labels_path = os.path.join(data_dir, labels_file)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} does not exist; the official Scania held-out labels are required to "
            f"score a run on the challenge {split} set.")
    return pd.read_csv(labels_path, usecols=[VEHICLE_ID, CLASS_LABEL])


def get_challenge_features(
        data_module: ScaniaDataModule,
        split: Literal["test", "validation"] = "test",
) -> tuple[torch.Tensor, np.ndarray]:
    """Windows and vehicle ids of the official held-out set.

    Args:
        data_module: A ``ScaniaDataModule`` whose ``setup()`` has already run (the challenge set is
            normalized with that run's train-fitted params).
        split: Which held-out population to build -- ``"test"`` (default) or ``"validation"``.

    Returns:
        ``(features, vehicle_ids)``: a ``(N, sequence_len, n_features)`` float tensor and the
        matching ``(N,)`` vehicle-id array, one entry per held-out vehicle.

        ``vehicle_ids`` is the **alignment key** and must be used as such:
        ``ScaniaDataset._gen_sequence`` emits every long vehicle before every short one, so the row
        order is not the input order and labels have to be joined on the id, never zipped
        positionally.
    """
    challenge_set = (
        data_module.build_challenge_dataset() if split == "test"
        else data_module.build_validation_challenge_dataset())
    features, _ = challenge_set.get_features_targets()
    return features, np.asarray(challenge_set.id_array)


def run_challenge_evaluation(
        data_module: ScaniaDataModule,
        predict_fn: Callable[[torch.Tensor], dict[str, Any]],
        model_version: str,
        results_path: str,
        split: Literal["test", "validation"] = "test",
) -> pd.DataFrame | None:
    """Score a trained window model on an official held-out set and write the results.

    Args:
        data_module: A ``ScaniaDataModule`` whose ``setup()`` has already run.
        predict_fn: Called **once** with the ``(N, sequence_len, n_features)`` challenge feature
            tensor, returning ``{name: predicted_rul}`` -- so an ensemble runs one forward pass for
            all of its predictors instead of one per name. Each value is a tensor or array of
            ``N`` de-normalized RUL predictions. ``name`` becomes the ``model`` column and the file
            suffix (e.g. ``"test"``, ``"weighted"``, ``"h0"``).
        model_version: Model version string, used in the output file names.
        results_path: The run's results directory, where the two artifacts are written.
        split: Which held-out population to score against -- ``"test"`` (default) or
            ``"validation"``. See the module docstring.

    Returns:
        The metrics frame (one row per predictor), or ``None`` if the evaluation was skipped or
        failed -- this never raises, see the module docstring.
    """
    try:
        features, vehicle_ids = get_challenge_features(data_module, split=split)
        predictions_by_name = predict_fn(features)
        return _score_and_save(
            predictions_by_name=predictions_by_name,
            vehicle_ids=vehicle_ids,
            data_dir=data_module.data_dir,
            model_version=model_version,
            results_path=results_path,
            split=split,
        )
    except Exception as error:  # noqa: BLE001 - never let this kill a finished training run
        _report_failure(error)
        return None


def run_challenge_evaluation_rowwise(
        data_module: ScaniaDataModule,
        predict_rul_fn: Callable[[Any], np.ndarray],
        model_version: str,
        results_path: str,
        split: Literal["test", "validation"] = "test",
) -> pd.DataFrame | None:
    """Score a trained row-wise model (RSF) on an official held-out set.

    RSF cannot consume the windows: it treats every readout as an individual and predicts a total
    lifetime. Only each vehicle's **last** readout is scored here, because the challenge label
    describes the vehicle's state at the end of its readout series -- the row-wise equivalent of
    the ``only_final=True`` used for the window models.

    Args:
        data_module: A ``ScaniaDataModule`` whose ``setup()`` has already run.
        predict_rul_fn: Called once with the challenge ``ScikitDataset`` (one row per vehicle,
            ``.X`` features and ``.Y["Time"]`` elapsed times), returning that many de-normalized
            RUL predictions.
        model_version: Model version string, used in the output file names.
        results_path: The run's results directory, where the two artifacts are written.
        split: Which held-out population to score against -- ``"test"`` (default) or
            ``"validation"``. See the module docstring.

    Returns:
        The metrics frame (one row), or ``None`` if the evaluation was skipped or failed.
    """
    try:
        # Local import: dataset.ScikitDataset pulls in the C-MAPSS loader at module level, which
        # the window-model paths through this module have no reason to load.
        from dataset.ScikitDataset import ScikitDataset

        challenge_set = (
            data_module.build_challenge_dataset() if split == "test"
            else data_module.build_validation_challenge_dataset())
        scikit_dataset = ScikitDataset.from_scania_challenge(
            challenge_set, feature_cols=list(data_module.feature_cols))

        predictions = np.asarray(predict_rul_fn(scikit_dataset), dtype=np.float64).reshape(-1)

        return _score_and_save(
            predictions_by_name={"test": predictions},
            vehicle_ids=np.asarray(scikit_dataset.ids),
            data_dir=data_module.data_dir,
            model_version=model_version,
            results_path=results_path,
            split=split,
        )
    except Exception as error:  # noqa: BLE001 - never let this kill a finished training run
        _report_failure(error)
        return None


def _score_and_save(
        predictions_by_name: dict[str, Any],
        vehicle_ids: np.ndarray,
        data_dir: str,
        model_version: str,
        results_path: str,
        split: Literal["test", "validation"] = "test",
) -> pd.DataFrame:
    """Bin the predictions, score them against the official labels and write both artifacts.

    Args:
        predictions_by_name: ``{name: predicted_rul}``; every value must hold one prediction per
            entry of ``vehicle_ids``, in the same order.
        vehicle_ids: Vehicle id per prediction, used to join the official labels.
        data_dir: Directory holding the labels file for ``split``.
        model_version: Model version string, used in the output file names.
        results_path: Destination directory.
        split: Which held-out population these predictions describe -- ``"test"`` (default) or
            ``"validation"``. Selects the labels file and the output file names, so the two splits
            never collide on disk.

    Returns:
        The metrics frame, one row per predictor, in the order of ``predictions_by_name``.

    Raises:
        ValueError: If a prediction vector does not match ``vehicle_ids`` in length, or if the
            labels do not cover every predicted vehicle.
    """
    labels_file = CHALLENGE_LABELS_FILE if split == "test" else VALIDATION_LABELS_FILE
    file_tag = "" if split == "test" else "_validation"
    metrics_tag = "" if split == "test" else "-validation"

    labels = load_challenge_labels(data_dir, split=split)
    label_by_vehicle = labels.set_index(VEHICLE_ID)[CLASS_LABEL]

    vehicle_ids = np.asarray(vehicle_ids).reshape(-1)
    # Join on the vehicle id rather than zipping: the window order is not the input order.
    target_classes = label_by_vehicle.reindex(vehicle_ids)
    if target_classes.isna().any():
        missing = target_classes[target_classes.isna()].index.tolist()
        raise ValueError(
            f"{len(missing)} predicted vehicle(s) have no entry in {labels_file} "
            f"(first few: {missing[:5]}); the predictions and the labels describe different sets.")
    target_classes = target_classes.to_numpy(dtype=np.float64)

    metric_rows: list[dict[str, float | str]] = []
    for name, predictions in predictions_by_name.items():
        predicted_rul = _to_numpy(predictions)
        if predicted_rul.size != vehicle_ids.size:
            raise ValueError(
                f"predictor {name!r} returned {predicted_rul.size} predictions for "
                f"{vehicle_ids.size} vehicles.")

        predicted_classes = rul_to_class(predicted_rul)

        predictions_frame = pd.DataFrame({
            PREDICTION_COLUMNS[0]: vehicle_ids,
            PREDICTION_COLUMNS[1]: target_classes.astype(np.int64),
            PREDICTION_COLUMNS[2]: predicted_classes,
            PREDICTION_COLUMNS[3]: predicted_rul,
        })
        predictions_csv_path = os.path.join(
            results_path, f"predictions_{model_version}_challenge{file_tag}_{name}_scania.csv")
        predictions_frame.to_csv(predictions_csv_path, index=False)

        metrics = scania_classification_metrics(predicted_classes, target_classes)
        metric_rows.append({
            "model": name,
            "n_vehicles": int(metrics["n_samples"]),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "scania_score": metrics["scania_score"],
        })
        print(f"{_LOG_PREFIX} [{split}] {name:<10} accuracy {metrics['accuracy']:.4f} | "
              f"f1_macro {metrics['f1_macro']:.4f} | score {metrics['scania_score']:.1f}")

    metrics_frame = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    metrics_csv_path = os.path.join(results_path, f"{model_version}-challenge{metrics_tag}-scania.csv")
    metrics_frame.to_csv(metrics_csv_path, index=False)
    print(f"{_LOG_PREFIX} results saved under: {metrics_csv_path}")

    return metrics_frame


def _to_numpy(predictions: Any) -> np.ndarray:
    """Flatten a prediction vector to a 1-D float64 array, whatever its container.

    Args:
        predictions: A tensor or an array-like of RUL predictions.

    Returns:
        The predictions as a flat ``float64`` array.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    return np.asarray(predictions, dtype=np.float64).reshape(-1)


def _report_failure(error: Exception) -> None:
    """Print a challenge-evaluation failure without propagating it.

    A missing data file is an expected, benign condition (a Colab/RunPod volume may carry only the
    training files), so it is reported as a one-line skip. Anything else prints a full traceback --
    it is a bug worth seeing -- but still lets the run return its training results.

    Args:
        error: The exception raised by the evaluation.

    Returns:
        None.
    """
    if isinstance(error, FileNotFoundError):
        print(f"{_LOG_PREFIX} skipped: {error}")
        return

    print(f"{_LOG_PREFIX} FAILED, the run's other results are unaffected: {error}")
    traceback.print_exc()
