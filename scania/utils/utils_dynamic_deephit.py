"""Train Dynamic DeepHit (TensorFlow rewrite) on full Scania vehicle histories and report a RUL.

Dynamic DeepHit (`dynamic_deephit/class_DeepLongitudinal.py::Model_Longitudinal_Attention`) is a
competing-risks *survival* model: it predicts a joint probability mass over ``(event, discretized
absolute time)`` cells, from which a survival function ``S(t)`` can be derived. Every other model
in this repository is a **RUL regressor**, so this module bridges the two worlds -- and, unlike the
supervised models (which drop censored vehicles entirely), Dynamic DeepHit consumes censoring
natively via its log-likelihood loss.

Two things set this integration apart from every other Scania trainer:

- **One sample is one vehicle's entire readout history**, not a fixed-length sliding window. This
  is how the original paper trains the architecture (landmark-time evaluation over a patient's full
  visit history), and it needs zero new dataset-layer code: ``ScaniaDataset._gen_sequence`` puts a
  vehicle in its single-window, front-padded branch whenever ``sequence_len`` exceeds that vehicle's
  own row count, so setting ``sequence_len`` to the longest vehicle's readout count (computed here,
  never configured) turns every vehicle into exactly one ``pad_mode='nan'`` window covering its full
  history. See :func:`_compute_max_length` and :func:`_build_data_module`.
- **Time is discretized on the absolute axis**, not residual RUL. The model's masks
  (``import_data.f_get_fc_mask1/2/3``) index a fixed ``num_Category`` grid by literal time, and
  ``ScaniaDataset`` already exposes everything needed to build it without new accessors:
  ``time_step_array`` (each vehicle's last observed absolute time, defined for every row) and
  ``lower_bound_array`` (``length_of_study_time_step - time_step``, *never* NaN'd for censored rows,
  unlike ``label_array``/``rul``) sum to the absolute failure/censoring time for every vehicle,
  censored or not. See :func:`_build_time_bin_edges` and :func:`_digitize`.

Nothing under ``dynamic_deephit/`` is modified. ``dynamic_deephit/main.py`` and
``dynamic_deephit/get_main_CF.py`` are *scripts* (the former runs a full PBC2 training pipeline at
import time, the latter is a known-broken CysFib variant per its own docstring) and are never
imported; only the pure building blocks are (:func:`import_dynamic_deephit`).

``scania/hpo/optuna_search_dynamic_deephit.py`` runs Optuna HPO against this module, optimizing the
validation C-index (:func:`_validation_c_index`) that would otherwise only be used internally for
checkpoint selection. It imports :func:`_build_run_context`, :func:`_build_network_settings` and
:func:`_fit` -- the same three building blocks :func:`train_model` itself is assembled from -- so
that an Optuna trial can reuse the (searched-hyperparameter-independent) data/mask/time-bin setup
across every trial while only rebuilding a fresh model and re-running :func:`_fit` per trial.
:func:`_fit` never imports Optuna itself: pruning is layered in purely through its optional
``on_eval`` callback, keeping this module framework-agnostic.
"""

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from constants.scania_component_x_columns import VEHICLE_ID
from scania.challenge import run_challenge_evaluation
from scania.dataset import ScaniaDataModule
from scania.dataset.ScaniaDataset import ScaniaDataset
from scania.metrics import scania_score
from scania.utils.utils_scania import (
    assert_data_is_valid,
    create_and_get_checkpoints_results_path,
    generate_and_save_model_prediction,
    save_train_parameters,
)
from shared.utils import ModelVersion, set_seed

# Repository root, from ``scania/utils/utils_dynamic_deephit.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DYNAMIC_DEEPHIT_ROOT = os.path.join(_REPO_ROOT, "dynamic_deephit")

_READOUTS_FILE = "train_operational_readouts.csv"
_LOG_PREFIX = "[Scania][ddh]"

_ACTIVATION_NAMES = ("relu", "tanh")


def import_dynamic_deephit() -> tuple[type, Any, Any, Any]:
    """Import the TF Dynamic DeepHit building blocks without modifying the package.

    ``dynamic_deephit/`` is a flat script dump, not an installable package: its modules import
    each other by bare sibling name (``class_DeepLongitudinal.py`` does ``import utils_network as
    utils``), so it only needs to be put on ``sys.path`` -- there is no ``auton_survival``-style
    broken ``__init__.py`` to work around here, unlike the retired PyTorch integration.

    ``main.py`` and ``get_main_CF.py`` must never be imported: the former is a script that runs a
    full PBC2 training pipeline at import time, and the latter's own module docstring says it is
    broken (it calls a data-loading function that does not exist). Only the pure building blocks
    used here are imported: the model class, the three mask builders and the minibatch sampler
    (``import_data``/``utils_helper``, pure functions, safe to call unmodified), and ``c_index``
    (``utils_eval``, also pure, needs ``lifelines`` only for its unused-here weighted variant).

    Returns:
        ``(Model_Longitudinal_Attention, import_data, utils_helper, utils_eval)``.
    """
    if _DYNAMIC_DEEPHIT_ROOT not in sys.path:
        sys.path.insert(0, _DYNAMIC_DEEPHIT_ROOT)

    from class_DeepLongitudinal import Model_Longitudinal_Attention
    import import_data
    import utils_helper
    import utils_eval

    return Model_Longitudinal_Attention, import_data, utils_helper, utils_eval


def _compute_max_length(dataset_root: str) -> int:
    """Return the largest number of readouts any vehicle has in the raw training file.

    Read *before* building the ``ScaniaDataModule`` so that value can become this run's
    ``sequence_len`` -- the knob that turns every vehicle's window into its own full history (see
    the module docstring). Only the vehicle id column is read (``usecols``), so this is cheap even
    though the file itself is over a gigabyte.

    Args:
        dataset_root: Root directory of the Scania data files.

    Returns:
        The maximum row count of any single vehicle in ``train_operational_readouts.csv``.
    """
    vehicle_ids = pd.read_csv(
        os.path.join(dataset_root, _READOUTS_FILE), usecols=[VEHICLE_ID])[VEHICLE_ID]
    return int(vehicle_ids.value_counts().max())


def _build_data_module(
        dataset_root: str,
        max_length: int,
        seed: int | None,
        data_fraction: float,
        val_rate: float,
        test_rate: float,
        stratify: bool,
        norm_type: str | None,
        counter_mode: str,
        include_histograms: bool,
        histogram_mode: str,
        cache_dir: str | None,
        force_load_from_cache: bool,
) -> ScaniaDataModule:
    """Build and set up a ``ScaniaDataModule`` configured for one-window-per-vehicle histories.

    ``sequence_len=max_length`` and ``pad_mode='nan'`` are hard-coded here, not exposed as config:
    they are not modelling choices for this run, they are what makes "one window == one vehicle's
    full history" true (see the module docstring). Every DataLoader-only knob (``batch_size``,
    ``num_workers``, ``pin_memory``, ``shuffle_loader``, ``return_sequence_label``) is fixed too --
    this model never touches a ``DataLoader``.

    Args:
        dataset_root: Root directory of the Scania data files.
        max_length: The window length that makes every vehicle single-windowed (see
            :func:`_compute_max_length`).
        seed: Seed forwarded to the dataset split/subsample.
        data_fraction: Fraction of vehicles to keep.
        val_rate: Fraction of vehicles held out for validation.
        test_rate: Fraction of vehicles held out for test.
        stratify: Whether the vehicle split is stratified by series length.
        norm_type: ``'z-score'`` or ``None``.
        counter_mode: ``'cumulative'``, ``'delta'`` or ``'both'``.
        include_histograms: Whether histogram features are added.
        histogram_mode: ``'sum'`` or ``'zhist'``.
        cache_dir: Dataset cache directory, or ``None`` for the default. Because ``max_length``
            differs from every other model's ``sequence_len``, this run gets its own cache
            sub-directory and cannot usefully share ``--force-load-from-cache`` with them.
        force_load_from_cache: Pin the run to an existing dataset cache.

    Returns:
        A ``ScaniaDataModule`` whose ``setup()`` has already run.
    """
    data_module = ScaniaDataModule(
        data_dir=dataset_root,
        batch_size=None,
        sequence_len=max_length,
        seed=seed,
        data_fraction=data_fraction,
        val_rate=val_rate,
        test_rate=test_rate,
        stratify=stratify,
        norm_type=norm_type,
        shuffle_loader=False,
        cache_dir=cache_dir,
        num_workers=0,
        pin_memory=False,
        return_sequence_label=False,
        counter_mode=counter_mode,
        include_histograms=include_histograms,
        histogram_mode=histogram_mode,
        pad_mode="nan",
        force_load_from_cache=force_load_from_cache,
    )
    data_module.setup()
    return data_module


def _windows_to_x_and_missing(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert NaN-tail-padded windows into the model's ``(x, x_mi)`` input pair.

    Dynamic DeepHit's ``get_seq_length`` treats a timestep as padding iff *every* feature at that
    step is exactly zero, and needs its column 0 to be "time since the previous reading". Scania
    exposes no per-row time delta through any existing accessor, so column 0 is a constant stand-in:
    ``1.0`` on real steps, ``0.0`` on padding. This is not just a simplification -- it also
    guarantees no real row is ever mistaken for padding, which is more robust than relying on the
    z-scored features themselves (a real row could otherwise coincidentally sum to zero).

    ``x_mi`` (Dynamic DeepHit's *within-row* missing-value mask, orthogonal to padding) is all
    zeros: Scania's engineered features have no missing values at an observed readout.

    Args:
        windows: ``(N, max_length, n_features)`` windows, NaN in every feature on padding steps
            (``pad_mode='nan')``, finite elsewhere.

    Returns:
        ``(x, x_mi)``, both ``(N, max_length, 1 + n_features)`` ``float32`` arrays.
    """
    real_step = ~np.isnan(windows[:, :, 0])
    delta = real_step.astype(np.float32)[..., np.newaxis]
    features = np.nan_to_num(windows, nan=0.0).astype(np.float32)
    x = np.concatenate([delta, features], axis=-1)
    x_mi = np.zeros_like(x)
    return x, x_mi


def _build_time_bin_edges(*absolute_times: np.ndarray, num_bins: int) -> np.ndarray:
    """Build an explicit, strictly increasing grid of absolute-time bin edges.

    Quantiles of the pooled input times give ``num_bins`` roughly-equal-mass bins (fewer if ties
    collapse some quantiles, via ``np.unique``). The first edge sits below the smallest observed
    time and the last at the largest, so :func:`_digitize` never needs a fallback for in-range
    values -- and clips gracefully for out-of-range ones (a held-out vehicle older than anything
    seen in train/val).

    Args:
        *absolute_times: One or more ``(N,)`` arrays of absolute times to pool together.
        num_bins: Target number of bins (``len(edges) - 1``, at most this after deduplication).

    Returns:
        The bin edges, length ``num_bins + 1`` or fewer.
    """
    assert num_bins >= 2, f"num_bins must be at least 2, got {num_bins}"
    pooled = np.concatenate(absolute_times)
    quantile_points = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    interior = np.quantile(pooled, quantile_points)
    edges = np.unique(np.concatenate([[pooled.min() - 1.0], interior, [pooled.max()]]))
    return edges


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map absolute times to bin indices in ``[0, len(edges) - 2]``, clipped at both ends.

    ``np.digitize(..., right=True)`` alone would return ``-1`` for a value at or below ``edges[0]``
    and ``len(edges) - 1`` (one past the last bin) for a value above ``edges[-1]`` -- both silent
    corruptions if used as a raw array index. Clipping here makes any out-of-range value (including
    a held-out vehicle's time exceeding the training range) land in the nearest valid bin instead.

    Args:
        values: Absolute times to digitize.
        edges: Bin edges from :func:`_build_time_bin_edges`.

    Returns:
        Integer bin indices, same shape as ``values``.
    """
    num_category = len(edges) - 1
    index = np.digitize(values, edges, right=True) - 1
    return np.clip(index, 0, num_category - 1).astype(np.int64)


def _build_masks(
        meas_time_bin: np.ndarray,
        event_time_bin: np.ndarray,
        event_indicator: np.ndarray,
        num_category: int,
        import_data_module: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three masks ``Model_Longitudinal_Attention``'s losses need, via the vendored
    ``import_data`` helpers (single risk: ``num_Event=1``).

    Args:
        meas_time_bin: ``(N,)`` bin index of each vehicle's last real observation.
        event_time_bin: ``(N,)`` bin index of each vehicle's absolute failure/censoring time.
        event_indicator: ``(N,)`` ``1`` if the vehicle failed, ``0`` if censored.
        num_category: Number of time bins.
        import_data_module: The vendored ``dynamic_deephit.import_data`` module.

    Returns:
        ``(mask1, mask2, mask3)``, shapes ``(N, 1, num_category)``, ``(N, 1, num_category)`` and
        ``(N, num_category)``.
    """
    meas_time_col = meas_time_bin[:, np.newaxis].astype(float)
    event_time_col = event_time_bin[:, np.newaxis].astype(float)
    event_col = event_indicator[:, np.newaxis].astype(float)

    mask1 = import_data_module.f_get_fc_mask1(meas_time_col, 1, num_category)
    mask2 = import_data_module.f_get_fc_mask2(event_time_col, event_col, 1, num_category)
    mask3 = import_data_module.f_get_fc_mask3(event_time_col, meas_time_col, num_category)
    return mask1, mask2, mask3


def _extract_vehicle_arrays(dataset: ScaniaDataset) -> dict[str, np.ndarray]:
    """Pull every per-vehicle quantity this module needs out of one split's ``ScaniaDataset``.

    All of it is already public, populated by ``_gen_sequence`` -- no new accessor needed.
    ``lower_bound_array`` is used rather than ``label_array``/``rul`` because it is defined for
    *every* row (``count_rul`` NaNs the RUL of censored rows, never the lower bound), so
    ``time_step_array + lower_bound_array`` is the absolute failure/censoring time for censored and
    uncensored vehicles alike, with no branching.

    Args:
        dataset: A ``ScaniaDataset`` built with ``sequence_len`` at least the longest vehicle's row
            count (so every vehicle is exactly one window).

    Returns:
        A dict with keys ``windows`` (``(N, max_length, n_features)``, NaN-padded),
        ``meas_time_abs``, ``event_time_abs``, ``event_indicator`` (all ``(N,)`` ``float64``), and
        ``vehicle_ids``.
    """
    windows, _ = dataset.get_features_targets()
    meas_time_abs = dataset.time_step_array.astype(np.float64)
    event_time_abs = meas_time_abs + dataset.lower_bound_array.astype(np.float64)
    event_indicator = (1 - dataset.is_censored_array).astype(np.float64)
    return {
        "windows": windows.numpy(),
        "meas_time_abs": meas_time_abs,
        "event_time_abs": event_time_abs,
        "event_indicator": event_indicator,
        "vehicle_ids": dataset.id_array,
    }


@dataclass
class RunContext:
    """Everything about a Dynamic DeepHit run that is independent of the searched hyperparameters.

    None of ``mb_size``, ``iteration_burn_in``, ``iteration``, ``keep_prob``, ``lr_train``,
    ``alpha``, ``beta``, ``gamma``, ``reg_W``, ``reg_W_out``, ``FC_active_fn``, ``RNN_active_fn``,
    ``h_dim_RNN``, ``h_dim_FC``, ``num_layers_RNN``, ``num_layers_ATT``, ``num_layers_CS`` or
    ``RNN_type`` affects the data module, the per-split arrays, the time-bin discretization or the
    training-split masks -- so a single ``RunContext`` is built once and reused, unchanged, across
    every :func:`train_model` call and every Optuna trial in ``scania/hpo/optuna_search_dynamic_deephit.py``.

    Attributes:
        scania_data_module: The set-up ``ScaniaDataModule``.
        dataset_kwargs: The resolved keyword arguments :func:`_build_data_module` was called with
            (including the computed ``max_length``), kept around for :func:`save_train_parameters`.
        train_arrays: Train split arrays from :func:`_extract_vehicle_arrays`, plus ``x``/``x_mi``/
            ``meas_time_bin``/``event_time_bin``.
        val_arrays: Same, for the validation split.
        test_arrays: Same, for the test split.
        train_masks: ``(mask1, mask2, mask3)`` from :func:`_build_masks`, built from the train split.
        edges: Time-bin edges from :func:`_build_time_bin_edges`.
        num_category: Number of realized time bins (``len(edges) - 1``).
        input_dims: The model's ``input_dims`` dict (``x_dim``, ``x_dim_cont``, ``x_dim_bin``,
            ``num_Event``, ``num_Category``, ``max_length``).
    """

    scania_data_module: ScaniaDataModule
    dataset_kwargs: dict[str, Any]
    train_arrays: dict[str, np.ndarray]
    val_arrays: dict[str, np.ndarray]
    test_arrays: dict[str, np.ndarray]
    train_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    edges: np.ndarray
    num_category: int
    input_dims: dict[str, int]


def _build_run_context(
        dataset_kwargs: dict[str, Any],
        num_category_bins: int,
        import_data_module: Any,
) -> RunContext:
    """Build everything a Dynamic DeepHit run needs that does not depend on the searched
    training/model hyperparameters (see :class:`RunContext`).

    Args:
        dataset_kwargs: Keyword arguments for :func:`_build_data_module` -- ``dataset_root``,
            ``seed``, ``data_fraction``, ``val_rate``, ``test_rate``, ``stratify``, ``norm_type``,
            ``counter_mode``, ``include_histograms``, ``histogram_mode``, ``cache_dir``,
            ``force_load_from_cache`` -- without ``max_length``, which is computed here.
        num_category_bins: Target number of absolute-time discretization bins.
        import_data_module: The vendored ``dynamic_deephit.import_data`` module (mask builders).

    Returns:
        A populated :class:`RunContext`.
    """
    max_length = _compute_max_length(dataset_kwargs["dataset_root"])
    print(f"{_LOG_PREFIX} sequence_len (=max vehicle length) = {max_length}")

    dataset_kwargs = {**dataset_kwargs, "max_length": max_length}
    print("Creating data module with the following parameters :")
    print(dataset_kwargs)
    scania_data_module = _build_data_module(**dataset_kwargs)

    # ``force_load_from_cache`` adopts every split-defining param from the cache manifest,
    # including ``sequence_len`` (see ``ScaniaDataModule._apply_cached_config``), silently
    # overwriting ``scania_data_module.sequence_len`` out from under the freshly-computed
    # ``max_length`` above whenever the cache was built for a different model. Continuing would
    # build the model's TF graph for ``max_length`` while every window actually has
    # ``scania_data_module.sequence_len`` steps -- and, worse, would silently break "one window ==
    # one vehicle's full history", the invariant this whole integration depends on (see the module
    # docstring). Fail loudly instead of training on truncated windows.
    if scania_data_module.sequence_len != max_length:
        raise ValueError(
            f"{_LOG_PREFIX} --force-load-from-cache pinned sequence_len to "
            f"{scania_data_module.sequence_len} (from the cache manifest at "
            f"{dataset_kwargs['cache_dir']}), but Dynamic DeepHit needs sequence_len == the longest "
            f"vehicle's row count ({max_length}) for its one-window-per-vehicle-history design. This "
            f"cache was built for a different model's sequence_len and cannot be shared with Dynamic "
            f"DeepHit. Point --dataset-cache-dir at a cache built for this model (run once without "
            f"--force-load-from-cache to create it), or drop --force-load-from-cache for this run."
        )

    train_arrays = _extract_vehicle_arrays(scania_data_module.get_full_dataset("train"))
    val_arrays = _extract_vehicle_arrays(scania_data_module.get_full_dataset("val"))
    test_arrays = _extract_vehicle_arrays(scania_data_module.get_full_dataset("test"))

    edges = _build_time_bin_edges(
        train_arrays["meas_time_abs"], train_arrays["event_time_abs"],
        val_arrays["meas_time_abs"], val_arrays["event_time_abs"],
        num_bins=num_category_bins,
    )
    num_category = len(edges) - 1
    print(f"{_LOG_PREFIX} {num_category} time bins, edges [{edges[0]:.2f}, {edges[-1]:.2f}]")

    for split_arrays in (train_arrays, val_arrays, test_arrays):
        split_arrays["x"], split_arrays["x_mi"] = _windows_to_x_and_missing(split_arrays["windows"])
        split_arrays["meas_time_bin"] = _digitize(split_arrays["meas_time_abs"], edges)
        split_arrays["event_time_bin"] = _digitize(split_arrays["event_time_abs"], edges)

    n_features = train_arrays["x"].shape[-1] - 1
    input_dims = {
        "x_dim": n_features + 1,
        "x_dim_cont": n_features,
        "x_dim_bin": 0,
        "num_Event": 1,
        "num_Category": num_category,
        "max_length": max_length,
    }

    train_masks = _build_masks(
        train_arrays["meas_time_bin"], train_arrays["event_time_bin"],
        train_arrays["event_indicator"], num_category, import_data_module,
    )

    return RunContext(
        scania_data_module=scania_data_module,
        dataset_kwargs=dataset_kwargs,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_arrays=test_arrays,
        train_masks=train_masks,
        edges=edges,
        num_category=num_category,
        input_dims=input_dims,
    )


def _build_network_settings(
        tf_module: Any,
        h_dim_RNN: int,
        h_dim_FC: int,
        num_layers_RNN: int,
        num_layers_ATT: int,
        num_layers_CS: int,
        RNN_type: str,
        FC_active_fn: str,
        RNN_active_fn: str,
        reg_W: float,
        reg_W_out: float,
) -> dict[str, Any]:
    """Validate and resolve ``model_params`` into the ``network_settings`` dict
    ``Model_Longitudinal_Attention`` expects.

    ``FC_active_fn``/``RNN_active_fn`` arrive as plain strings (config-JSON-friendly, and what
    Optuna's categorical search suggests); the model class expects actual TF function references
    (see the module docstring), so the string -> function mapping happens here, once, shared by
    :func:`train_model` and every Optuna trial.

    Args:
        tf_module: The imported ``tensorflow`` module.
        h_dim_RNN: RNN hidden size.
        h_dim_FC: Hidden width of the attention and cause-specific FC networks.
        num_layers_RNN: Number of stacked RNN layers.
        num_layers_ATT: Number of hidden layers in the attention network.
        num_layers_CS: Number of hidden layers in the cause-specific network.
        RNN_type: ``'LSTM'`` or ``'GRU'``.
        FC_active_fn: ``'relu'`` or ``'tanh'`` -- activation of the FC networks.
        RNN_active_fn: ``'relu'`` or ``'tanh'`` -- activation of the RNN cells.
        reg_W: L1 regularization on the cause-specific FC layers.
        reg_W_out: L1 regularization on the output layer.

    Returns:
        The ``network_settings`` dict, with activations resolved to ``tf_module`` functions and
        ``initial_W`` added.
    """
    assert RNN_type in ("LSTM", "GRU"), f"Unsupported RNN_type: {RNN_type}"
    assert FC_active_fn in _ACTIVATION_NAMES, f"Unsupported FC_active_fn: {FC_active_fn}"
    assert RNN_active_fn in _ACTIVATION_NAMES, f"Unsupported RNN_active_fn: {RNN_active_fn}"

    activation_functions = {"relu": tf_module.nn.relu, "tanh": tf_module.nn.tanh}
    return {
        "h_dim_RNN": h_dim_RNN,
        "h_dim_FC": h_dim_FC,
        "num_layers_RNN": num_layers_RNN,
        "num_layers_ATT": num_layers_ATT,
        "num_layers_CS": num_layers_CS,
        "RNN_type": RNN_type,
        "FC_active_fn": activation_functions[FC_active_fn],
        "RNN_active_fn": activation_functions[RNN_active_fn],
        "initial_W": tf_module.keras.initializers.GlorotUniform(),
        "reg_W": reg_W,
        "reg_W_out": reg_W_out,
    }


def _predict_pmf(model: Any, x: np.ndarray, x_mi: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the model's forward pass in chunks.

    ``Model_Longitudinal_Attention``'s traced functions use a free batch dimension (``[None,
    max_length, x_dim]``), so any chunk size runs without retracing; chunking here only bounds peak
    memory on large splits.

    Args:
        model: A built (and, at inference time, trained) ``Model_Longitudinal_Attention``.
        x: ``(N, max_length, x_dim)`` model input.
        x_mi: ``(N, max_length, x_dim)`` missing-value mask.
        batch_size: Number of vehicles per forward pass.

    Returns:
        ``(N, num_Event, num_Category)`` joint probability mass.
    """
    if len(x) == 0:
        return np.empty((0, model.num_Event, model.num_Category))
    chunks = []
    for start in range(0, len(x), batch_size):
        chunks.append(model.predict(x[start:start + batch_size], x_mi[start:start + batch_size]))
    return np.concatenate(chunks, axis=0)


def _predicted_rul(pmf: np.ndarray, meas_time_bin: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert a predicted PMF into a RUL at each vehicle's own last real observation.

    ``pmf`` is the model's *unconditional* joint distribution over the whole absolute-time axis
    (event 0, since Scania is single-risk). Integrating it as-is would give the expected *total*
    lifetime from time zero, double-counting everything already observed -- what is wanted instead
    is the expected *remaining* life, i.e. the restricted mean of the survival function
    *conditioned on having survived to the vehicle's last observation*:
    ``S_cond(t) = S_uncond(t) / S_uncond(now)`` for ``t >= now`` (the same conditioning
    ``loss_Log_Likelihood``'s ``denom`` and ``main.py``'s ``f_get_risk_predictions`` use, just
    applied at inference time instead of during training/landmarking).

    The per-vehicle "now" (``meas_time_bin``, which bin edge the last observation falls in) differs
    per row, so a single shared integration grid does not work; instead each row's x-axis is
    ``max(edge - now, 0)`` -- edges before "now" collapse to a repeated ``0``, and the trapezoid
    rule assigns exactly zero area to any segment with no width, whatever ``y`` is there. This
    avoids a per-row Python loop while still integrating "from now" for every vehicle.

    Args:
        pmf: ``(N, 1, num_category)`` predicted joint mass (single risk).
        meas_time_bin: ``(N,)`` bin index of each vehicle's last real observation.
        edges: The bin edges the model was discretized on, length ``num_category + 1``.

    Returns:
        A ``(N,)`` array of non-negative predicted RUL, in raw time units.
    """
    cif = np.cumsum(pmf[:, 0, :], axis=1)
    survival_uncond = np.concatenate(
        [np.ones((len(cif), 1)), np.clip(1.0 - cif, 0.0, 1.0)], axis=1)  # (N, num_category + 1)

    now_survival = np.clip(survival_uncond[np.arange(len(cif)), meas_time_bin], 1e-6, 1.0)
    survival_cond = np.clip(survival_uncond / now_survival[:, np.newaxis], 0.0, 1.0)

    now_time = edges[meas_time_bin]
    x_from_now = np.clip(edges[np.newaxis, :] - now_time[:, np.newaxis], 0.0, None)

    return np.clip(np.trapezoid(survival_cond, x_from_now, axis=1), 0.0, None)


def _validation_c_index(
        model: Any,
        val_arrays: dict[str, np.ndarray],
        edges: np.ndarray,
        time_quantiles: list[float],
        max_vehicles: int,
        seed: int | None,
        utils_eval_module: Any,
        inference_batch_size: int,
) -> float:
    """Average C-index of the model's one-shot prediction over a few absolute-time horizons.

    This deliberately skips ``main.py``'s landmark truncation (predicting from a history cut off
    before the last observation): every vehicle here is already used at its full observed length
    for the RUL/RMSE/Scania-cost reporting, so scoring checkpoints the same way keeps the selection
    metric consistent with what is actually reported. It is an internal proxy only -- never
    reported -- so the simplification is acceptable here in exchange for reusing
    ``utils_eval.c_index`` unmodified and avoiding a second, more complex evaluation path.

    Args:
        model: The ``Model_Longitudinal_Attention`` being trained.
        val_arrays: The validation split's arrays from :func:`_extract_vehicle_arrays` plus ``x``/
            ``x_mi``.
        edges: Bin edges from :func:`_build_time_bin_edges`.
        time_quantiles: Quantiles (in ``(0, 1)``) of the validation event-time distribution to
            evaluate C-index at.
        max_vehicles: Cap on the number of validation vehicles used (the metric is ``O(n^2)``).
        seed: Seed for the subsample.
        utils_eval_module: The vendored ``dynamic_deephit.utils_eval`` module.
        inference_batch_size: Chunk size for the forward pass.

    Returns:
        The mean C-index across ``time_quantiles``, or ``nan`` if none could be computed.
    """
    event_time = val_arrays["event_time_abs"]
    death = val_arrays["event_indicator"]
    x, x_mi = val_arrays["x"], val_arrays["x_mi"]

    if max_vehicles < len(x):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(x), size=max_vehicles, replace=False)
        event_time, death, x, x_mi = event_time[keep], death[keep], x[keep], x_mi[keep]

    pmf = _predict_pmf(model, x, x_mi, inference_batch_size)
    cif = np.cumsum(pmf[:, 0, :], axis=1)

    scores = []
    for threshold in np.quantile(event_time, time_quantiles):
        bin_index = int(_digitize(np.array([threshold]), edges)[0])
        risk = cif[:, bin_index]
        score = utils_eval_module.c_index(risk, event_time, death, float(threshold))
        if score >= 0:  # c_index returns -1 when it could not be computed
            scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")


def _fit(
        model: Any,
        checkpoint: Any | None,
        checkpoint_prefix: str | None,
        context: RunContext,
        mb_size: int,
        burn_in_mode: bool,
        iteration_burn_in: int,
        iteration: int,
        keep_prob: float,
        lr_train: float,
        alpha: float,
        beta: float,
        gamma: float,
        eval_every: int,
        c_index_time_quantiles: list[float],
        c_index_max_vehicles: int,
        seed: int | None,
        utils_helper_module: Any,
        utils_eval_module: Any,
        inference_batch_size: int,
        on_eval: Callable[[int, float], None] | None = None,
) -> tuple[float, float]:
    """Run Dynamic DeepHit's burn-in + main training loop, tracking the best validation C-index.

    There is no early stopping: the full ``iteration`` budget always runs, matching
    ``dynamic_deephit/main.py``. Every ``eval_every`` main-phase iterations, :func:`_validation_c_index`
    is computed and the checkpoint is overwritten whenever it improves.

    ``checkpoint``/``checkpoint_prefix`` are optional so this function serves both call sites:
    :func:`train_model` passes a real ``tf.train.Checkpoint`` (its final scoring needs the
    best-by-C-index weights restored), while an Optuna trial (``scania/hpo/optuna_search_dynamic_deephit.py``)
    passes ``None`` for both to skip all disk I/O -- a trial only needs the scalar ``best_c_index``.

    ``on_eval``, if given, is called with ``(step, c_index)`` at every ``eval_every`` checkpoint. This
    is how Optuna pruning (``trial.report``/``should_prune``/``raise TrialPruned()``) is layered in
    from the HPO module without this module importing Optuna itself.

    Args:
        model: The (freshly constructed, untrained) ``Model_Longitudinal_Attention``.
        checkpoint: A ``tf.train.Checkpoint`` wrapping ``model``, or ``None`` to skip checkpointing.
        checkpoint_prefix: Path prefix to write/restore the checkpoint at. Required iff ``checkpoint``
            is not ``None``.
        context: The run's :class:`RunContext` (train/val arrays, masks, edges).
        mb_size: Minibatch size for both training phases.
        burn_in_mode: Whether to run the burn-in phase (RNN-reconstruction loss only) first.
        iteration_burn_in: Number of burn-in minibatch steps.
        iteration: Number of main-phase minibatch steps.
        keep_prob: Dropout keep probability.
        lr_train: Learning rate for both phases.
        alpha: Weight of the log-likelihood loss.
        beta: Weight of the ranking loss.
        gamma: Weight of the RNN-prediction (longitudinal reconstruction) loss.
        eval_every: Validate (and possibly checkpoint) every this many main-phase iterations.
        c_index_time_quantiles: Quantiles of the validation event-time distribution at which the
            checkpoint-selection C-index is computed.
        c_index_max_vehicles: Cap on validation vehicles used for the (quadratic) C-index.
        seed: Seed for the C-index validation subsample.
        utils_helper_module: The vendored ``dynamic_deephit.utils_helper`` module (minibatch sampler).
        utils_eval_module: The vendored ``dynamic_deephit.utils_eval`` module (``c_index``).
        inference_batch_size: Chunk size for every forward-pass-only evaluation.
        on_eval: Optional callback invoked with ``(step, c_index)`` at every ``eval_every`` checkpoint.

    Returns:
        ``(best_c_index, training_time_seconds)``.
    """
    train_arrays = context.train_arrays
    val_arrays = context.val_arrays
    train_mask1, train_mask2, train_mask3 = context.train_masks

    training_start = time.perf_counter()

    if burn_in_mode:
        print(f"{_LOG_PREFIX} burn-in training ({iteration_burn_in} iterations)...")
        for step in tqdm(range(iteration_burn_in)):
            # mask1/mask2/mask3 are unused by train_burn_in (RNN-reconstruction loss only) but
            # f_get_minibatch always returns all seven arrays, so they still have to be passed in.
            x_mb, x_mi_mb, k_mb, t_mb, _, _, _ = utils_helper_module.f_get_minibatch(
                mb_size, train_arrays["x"], train_arrays["x_mi"],
                train_arrays["event_indicator"][:, np.newaxis],
                train_arrays["event_time_bin"][:, np.newaxis],
                train_mask1, train_mask2, train_mask3,
            )
            _, loss = model.train_burn_in((x_mb, k_mb, t_mb), x_mi_mb, keep_prob, lr_train)
            if (step + 1) % 1000 == 0:
                print(f"{_LOG_PREFIX} burn-in itr {step + 1:6d} | loss {loss:.4f}")

    if checkpoint is not None:
        checkpoint.write(checkpoint_prefix)  # baseline, so a checkpoint always exists
    best_c_index = float("-inf")

    print(f"{_LOG_PREFIX} main training ({iteration} iterations)...")
    for step in tqdm(range(iteration)):
        x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = utils_helper_module.f_get_minibatch(
            mb_size, train_arrays["x"], train_arrays["x_mi"],
            train_arrays["event_indicator"][:, np.newaxis],
            train_arrays["event_time_bin"][:, np.newaxis],
            train_mask1, train_mask2, train_mask3,
        )
        _, loss = model.train(
            (x_mb, k_mb, t_mb), (m1_mb, m2_mb, m3_mb), x_mi_mb, (alpha, beta, gamma),
            keep_prob, lr_train)

        if (step + 1) % 1000 == 0:
            print(f"{_LOG_PREFIX} itr {step + 1:6d} | loss {loss:.4f}")

        if (step + 1) % eval_every == 0:
            c_index = _validation_c_index(
                model, val_arrays, context.edges, c_index_time_quantiles, c_index_max_vehicles, seed,
                utils_eval_module, inference_batch_size)
            print(f"{_LOG_PREFIX} itr {step + 1:6d} | val C-index {c_index:.4f}")
            if c_index > best_c_index:
                best_c_index = c_index
                if checkpoint is not None:
                    checkpoint.write(checkpoint_prefix)
                print(f"{_LOG_PREFIX} updated... best val C-index = {best_c_index:.4f}")
            if on_eval is not None:
                on_eval(step + 1, c_index)

    training_time_seconds = time.perf_counter() - training_start

    if checkpoint is not None:
        checkpoint.restore(checkpoint_prefix).expect_partial()

    return best_c_index, training_time_seconds


def train_model(
        checkpoints_path: str,
        results_path: str,
        model_version: ModelVersion,
        # Dataset params
        dataset_root: str,
        seed: int | None,
        data_fraction: float,
        val_rate: float,
        test_rate: float,
        stratify: bool,
        norm_type: str | None,
        counter_mode: str,
        include_histograms: bool,
        histogram_mode: str,
        # Training params
        mb_size: int,
        burn_in_mode: bool,
        iteration_burn_in: int,
        iteration: int,
        keep_prob: float,
        lr_train: float,
        alpha: float,
        beta: float,
        gamma: float,
        eval_every: int,
        num_category_bins: int,
        use_gpu: bool,
        c_index_time_quantiles: list[float],
        c_index_max_vehicles: int,
        inference_batch_size: int,
        # Model params
        h_dim_RNN: int,
        h_dim_FC: int,
        num_layers_RNN: int,
        num_layers_ATT: int,
        num_layers_CS: int,
        RNN_type: str,
        FC_active_fn: str,
        RNN_active_fn: str,
        reg_W: float,
        reg_W_out: float,
        # Others
        cache_dir: str | None = None,
        force_load_from_cache: bool = False,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Train Dynamic DeepHit on full Scania vehicle histories and write the benchmark artifacts.

    See the module docstring for the two departures from every other Scania trainer: one sample is
    one vehicle's full history (``sequence_len`` is computed, not configured), and time is
    discretized on the absolute axis. Hyperparameter names and defaults mirror
    ``dynamic_deephit/main.py``'s own ``new_parser``/``network_settings`` dicts directly.

    Args:
        checkpoints_path: Root directory for checkpoints; a per-run subfolder is created inside.
        results_path: Root directory for results; a per-run subfolder is created inside.
        model_version: The model version being trained, used in every output file name.
        dataset_root: Root directory of the Scania data files.
        seed: Seed for python/numpy/torch/TF, and for the C-index validation subsample.
        data_fraction: Fraction of vehicles to keep.
        val_rate: Fraction of vehicles held out for validation.
        test_rate: Fraction of vehicles held out for test.
        stratify: Whether the vehicle split is stratified by series length.
        norm_type: ``'z-score'`` or ``None``.
        counter_mode: ``'cumulative'``, ``'delta'`` or ``'both'``.
        include_histograms: Whether histogram features are added.
        histogram_mode: ``'sum'`` or ``'zhist'``.
        mb_size: Minibatch size for both training phases.
        burn_in_mode: Whether to run the burn-in phase (RNN-reconstruction loss only) first.
        iteration_burn_in: Number of burn-in minibatch steps.
        iteration: Number of main-phase minibatch steps. There is no early stopping (matching
            ``main.py``): the full budget always runs, and the best-by-C-index checkpoint is
            restored afterwards.
        keep_prob: Dropout keep probability.
        lr_train: Learning rate for both phases.
        alpha: Weight of the log-likelihood loss.
        beta: Weight of the ranking loss.
        gamma: Weight of the RNN-prediction (longitudinal reconstruction) loss.
        eval_every: Validate (and possibly checkpoint) every this many main-phase iterations.
        num_category_bins: Target number of absolute-time discretization bins.
        use_gpu: Train on GPU when available.
        c_index_time_quantiles: Quantiles of the validation event-time distribution at which the
            checkpoint-selection C-index is computed.
        c_index_max_vehicles: Cap on validation vehicles used for the (quadratic) C-index.
        inference_batch_size: Chunk size for every forward-pass-only evaluation.
        h_dim_RNN: RNN hidden size.
        h_dim_FC: Hidden width of the attention and cause-specific FC networks.
        num_layers_RNN: Number of stacked RNN layers.
        num_layers_ATT: Number of hidden layers in the attention network.
        num_layers_CS: Number of hidden layers in the cause-specific network.
        RNN_type: ``'LSTM'`` or ``'GRU'``.
        FC_active_fn: ``'relu'`` or ``'tanh'`` -- activation of the FC networks.
        RNN_active_fn: ``'relu'`` or ``'tanh'`` -- activation of the RNN cells.
        reg_W: L1 regularization on the cause-specific FC layers.
        reg_W_out: L1 regularization on the output layer.
        cache_dir: Dataset cache directory, or ``None`` for the default.
        force_load_from_cache: Pin the run to an existing dataset cache.
        datetime_for_folders: Timestamp used in the run's folder name.

    Returns:
        ``(test_rmse, test_score)`` on the uncensored vehicles of the internal test split, in raw
        RUL time units.
    """
    set_seed(seed)

    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
    )

    model_class, import_data, utils_helper, utils_eval = import_dynamic_deephit()

    import tensorflow as tf
    tf.random.set_seed(seed)
    _configure_gpu(tf, use_gpu)

    checkpoints_path, results_path = create_and_get_checkpoints_results_path(
        model_version=model_version.value,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    dataset_kwargs = {
        "dataset_root": dataset_root,
        "seed": seed,
        "data_fraction": data_fraction,
        "val_rate": val_rate,
        "test_rate": test_rate,
        "stratify": stratify,
        "norm_type": norm_type,
        "counter_mode": counter_mode,
        "include_histograms": include_histograms,
        "histogram_mode": histogram_mode,
        "cache_dir": cache_dir,
        "force_load_from_cache": force_load_from_cache,
    }
    context = _build_run_context(dataset_kwargs, num_category_bins, import_data)

    network_settings = _build_network_settings(
        tf,
        h_dim_RNN=h_dim_RNN,
        h_dim_FC=h_dim_FC,
        num_layers_RNN=num_layers_RNN,
        num_layers_ATT=num_layers_ATT,
        num_layers_CS=num_layers_CS,
        RNN_type=RNN_type,
        FC_active_fn=FC_active_fn,
        RNN_active_fn=RNN_active_fn,
        reg_W=reg_W,
        reg_W_out=reg_W_out,
    )

    training_kwargs = {
        "mb_size": mb_size,
        "burn_in_mode": burn_in_mode,
        "iteration_burn_in": iteration_burn_in,
        "iteration": iteration,
        "keep_prob": keep_prob,
        "lr_train": lr_train,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "eval_every": eval_every,
        "num_category_bins": num_category_bins,
        "num_category_realized": context.num_category,
        "use_gpu": use_gpu,
        "c_index_time_quantiles": c_index_time_quantiles,
        "c_index_max_vehicles": c_index_max_vehicles,
        "inference_batch_size": inference_batch_size,
    }
    model_kwargs = {
        "h_dim_RNN": h_dim_RNN,
        "h_dim_FC": h_dim_FC,
        "num_layers_RNN": num_layers_RNN,
        "num_layers_ATT": num_layers_ATT,
        "num_layers_CS": num_layers_CS,
        "RNN_type": RNN_type,
        "FC_active_fn": FC_active_fn,
        "RNN_active_fn": RNN_active_fn,
        "reg_W": reg_W,
        "reg_W_out": reg_W_out,
        "input_dims": context.input_dims,
    }
    print(f"Models parameters : {model_kwargs}")
    print(f"Training parameters : {training_kwargs}")
    save_train_parameters(
        results_path=results_path,
        dataset_parameters=context.dataset_kwargs,
        training_parameters=training_kwargs,
        model_parameters=model_kwargs,
    )

    model = model_class("dynamic_deephit_scania", context.input_dims, network_settings)

    checkpoint_prefix = os.path.join(checkpoints_path, "dynamic_deephit_ckpt")
    checkpoint = tf.train.Checkpoint(model=model)

    best_c_index, training_time_seconds = _fit(
        model=model,
        checkpoint=checkpoint,
        checkpoint_prefix=checkpoint_prefix,
        context=context,
        mb_size=mb_size,
        burn_in_mode=burn_in_mode,
        iteration_burn_in=iteration_burn_in,
        iteration=iteration,
        keep_prob=keep_prob,
        lr_train=lr_train,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        eval_every=eval_every,
        c_index_time_quantiles=c_index_time_quantiles,
        c_index_max_vehicles=c_index_max_vehicles,
        seed=seed,
        utils_helper_module=utils_helper,
        utils_eval_module=utils_eval,
        inference_batch_size=inference_batch_size,
    )
    print(f"{model_version.value} trained in {training_time_seconds:.1f}s")

    train_rmse, _ = _score_split(context.train_arrays, model, context.edges, inference_batch_size)
    val_rmse, _ = _score_split(context.val_arrays, model, context.edges, inference_batch_size)
    test_rmse, test_score, test_predictions, test_targets = _score_split(
        context.test_arrays, model, context.edges, inference_batch_size, return_predictions=True)

    scores = pd.DataFrame(
        columns=["train_rmse", "val_rmse", "test_rmse", "test_score", "training_time_seconds"])
    scores.loc[0] = [train_rmse, val_rmse, test_rmse, test_score, training_time_seconds]
    scores.to_csv(f"{results_path}/{model_version.value}-scania.csv", index=False)
    print(f"Scores from train and test :\n{scores}")

    def _predict_challenge_rul(
            features: Any, split: Literal["test", "validation"] = "test") -> dict[str, np.ndarray]:
        challenge_set = (
            context.scania_data_module.build_challenge_dataset() if split == "test"
            else context.scania_data_module.build_validation_challenge_dataset())
        features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)
        x_challenge, x_mi_challenge = _windows_to_x_and_missing(features_np)
        meas_time_bin_challenge = _digitize(
            challenge_set.time_step_array.astype(np.float64), context.edges)
        pmf_challenge = _predict_pmf(model, x_challenge, x_mi_challenge, inference_batch_size)
        return {"test": _predicted_rul(pmf_challenge, meas_time_bin_challenge, context.edges)}

    # Additionally score the trained model on the official Scania Component X held-out sets (test
    # and validation). Never raises: the run's own results are already written by the time this
    # runs. _predict_challenge_rul must be re-invoked per split: it needs that split's own
    # challenge_set (time_step_array) to digitize its measurement time bins, not just its features.
    run_challenge_evaluation(
        data_module=context.scania_data_module,
        predict_fn=lambda features: _predict_challenge_rul(features, split="test"),
        model_version=model_version.value,
        results_path=results_path,
    )
    run_challenge_evaluation(
        data_module=context.scania_data_module,
        predict_fn=lambda features: _predict_challenge_rul(features, split="validation"),
        model_version=model_version.value,
        results_path=results_path,
        split="validation",
    )

    return generate_and_save_model_prediction(
        predictions=torch.from_numpy(test_predictions).float(),
        targets=torch.from_numpy(test_targets).float(),
        model_version=model_version.value,
        prediction_type="test",
        results_path=results_path,
    )


def _configure_gpu(tf_module: Any, use_gpu: bool) -> None:
    """Enable or disable GPU visibility for training, mirroring ``main.py``.

    Args:
        tf_module: The imported ``tensorflow`` module.
        use_gpu: If ``False``, GPUs are hidden entirely; if ``True``, memory growth is enabled on
            every visible GPU (so this run does not pre-allocate the whole device).

    Returns:
        None.
    """
    if not use_gpu:
        tf_module.config.set_visible_devices([], "GPU")
        return
    for gpu in tf_module.config.list_physical_devices("GPU"):
        tf_module.config.experimental.set_memory_growth(gpu, True)


def _score_split(
        split_arrays: dict[str, np.ndarray],
        model: Any,
        edges: np.ndarray,
        inference_batch_size: int,
        return_predictions: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray, np.ndarray]:
    """Score the model on the uncensored vehicles of one split.

    Censored vehicles are excluded: they have no true RUL, so neither RMSE nor the Scania cost is
    defined on them. This mirrors what every other Scania trainer reports.

    Args:
        split_arrays: One split's arrays, including ``x``/``x_mi``/``meas_time_bin`` (from
            :func:`train_model`'s per-split conversion) and ``event_indicator``/``event_time_abs``.
        model: The (trained) ``Model_Longitudinal_Attention``.
        edges: Bin edges from :func:`_build_time_bin_edges`.
        inference_batch_size: Chunk size for the forward pass.
        return_predictions: If ``True``, also return the raw predictions/targets arrays.

    Returns:
        ``(rmse, scania_cost)``, or ``(rmse, scania_cost, predictions, targets)`` when
        ``return_predictions`` is set. All in raw RUL time units.
    """
    uncensored = split_arrays["event_indicator"] == 1
    x = split_arrays["x"][uncensored]
    x_mi = split_arrays["x_mi"][uncensored]
    meas_time_bin = split_arrays["meas_time_bin"][uncensored]
    targets = split_arrays["event_time_abs"][uncensored] - split_arrays["meas_time_abs"][uncensored]

    pmf = _predict_pmf(model, x, x_mi, inference_batch_size)
    predictions = _predicted_rul(pmf, meas_time_bin, edges)

    rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
    score = float(scania_score(predictions, targets))

    if return_predictions:
        return rmse, score, predictions, targets
    return rmse, score
