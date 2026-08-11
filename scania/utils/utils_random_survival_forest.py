import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
from numpy import ndarray
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sksurv.ensemble import RandomSurvivalForest

from dataset.ScikitDataset import ScikitDataset
from scania.challenge import run_challenge_evaluation_rowwise
from scania.dataset import ScaniaDataModule
from scania.metrics import scania_score
from scania.utils.utils_scania import assert_data_is_valid, create_and_get_checkpoints_results_path
from shared.utils import ModelVersion, set_seed


def train_model(
        checkpoints_path: str,
        results_path: str,
        dataset_root: str,
        model_version: str,
        # Models params
        n_estimators: list[int] | None,
        max_depth: list[int] | None,
        min_samples_split: list[int] | None,
        min_samples_leaf: list[int] | None,
        cv_for_grid_search: int,
        # Dataset params
        seed: int,
        data_fraction: float,
        val_rate: float,
        test_rate: float,
        stratify: bool,
        norm_type: str | None,
        cache_dir: str | None,
        return_sequence_label: bool,
        counter_mode: str = "delta",
        include_histograms: bool = False,
        histogram_mode: str = "sum",
        # Others params
        force_load_from_cache: bool = False,
        model_kwargs=None,
        variance_warning_threshold: float = 0.05,  # Threshold for stability warning
        datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
):
    set_seed(seed)

    assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
    )



    print("Loading dataset...")

    scania_data_module = ScaniaDataModule(
        data_dir=dataset_root,
        batch_size=None,
        sequence_len=1,
        seed=seed,
        data_fraction=data_fraction,
        val_rate=val_rate,
        test_rate=test_rate,
        stratify=stratify,
        norm_type=norm_type,
        cache_dir=cache_dir,
        return_sequence_label=return_sequence_label,
        counter_mode=counter_mode,
        include_histograms=include_histograms,
        histogram_mode=histogram_mode,
        # RSF consumes ScaniaDataset.df row-wise (see ScikitDataset.from_scania), never the
        # windows, so adopting the cache's sequence_len over the sequence_len=1 above changes
        # nothing it reads -- it only lets RSF share the benchmark splits instead of rebuilding
        # (and overwriting) the cache with its own.
        force_load_from_cache=force_load_from_cache,
    )

    # Transform the ScaniaDataModule into the scikit-survival format expected by RSF:
    # every readout row is one individual (Time=time_step, Status=failure). Training
    # keeps all rows (censoring handled via Status); the test set keeps uncensored
    # (failure) rows only so a true RUL exists for RMSE/score.
    train_dataset, test_dataset, _ = ScikitDataset.from_scania(scania_data_module)
    train_X = train_dataset.X
    train_Y = train_dataset.Y
    test_X = test_dataset.X
    test_Y = test_dataset.Y

    has_multiple_params = (
            (n_estimators and len(n_estimators) > 1)
            or (max_depth and len(max_depth) > 1)
            or (min_samples_leaf and len(min_samples_leaf) > 1)
            or (min_samples_split and len(min_samples_split) > 1)
    )

    rsf = RandomSurvivalForest(
        n_jobs=-1,
        random_state=seed,
    )

    if has_multiple_params:
        print("Multiple params detected, the program will perform a grid search...")
        best_rsf = select_best_params(
            rsf,
            train_X,
            train_Y,
            cv_for_grid_search,
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            seed,
            variance_warning_threshold,
        )
    else:
        print("No grid search needed. Evaluating model stability via Cross-Validation...")
        best_params = {}

        if n_estimators is not None and len(n_estimators) > 0:
            best_params["n_estimators"] = n_estimators[0]
        if max_depth is not None and len(max_depth) > 0:
            best_params["max_depth"] = max_depth[0]
        if min_samples_split is not None and len(min_samples_split) > 0:
            best_params["min_samples_split"] = min_samples_split[0]
        if min_samples_leaf is not None and len(min_samples_leaf) > 0:
            best_params["min_samples_leaf"] = min_samples_leaf[0]

        print(f"The parameters are: {best_params}")

        best_rsf = RandomSurvivalForest(
            **best_params,
            n_jobs=-1,
            random_state=seed,
        )

        status = train_Y["Status"]

        # Dynamically adjust folds based on failure density
        safe_cv_folds = calculate_safe_cv_folds(status, cv_for_grid_search)

        stratified_k_fold = StratifiedKFold(n_splits=safe_cv_folds, shuffle=True, random_state=seed)
        custom_splits = list(stratified_k_fold.split(train_X, status))

        print(f"Running {safe_cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            best_rsf,
            train_X,
            train_Y,
            cv=custom_splits,
            n_jobs=-1
        )

        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        print(f"CV Concordance Index: {mean_score:.4f} (+/- {std_score:.4f})")

        # --- Robustness Warning for Standard CV ---
        if std_score > variance_warning_threshold:
            warnings.warn(
                f"\n[ROBUSTNESS WARNING] High variance detected in cross-validation (std: {std_score:.4f} > {variance_warning_threshold}). "
                "The model's performance fluctuates significantly across folds. This indicates potential instability, overfitting, or heterogeneous data distribution across folds.\n"
            )

        print("Fitting the final model on the entire training dataset...")
        best_rsf.fit(train_X, train_Y)

    # --- Predictions and evaluations ---
    predicted_ruls = _predict_rul(best_rsf, test_X, test_Y['Time'])

    rmse = float(np.sqrt(mean_squared_error(test_dataset.rul, predicted_ruls)))
    score = float(scania_score(predicted_ruls, test_dataset.rul))

    print(f'Test RMSE : {rmse}')
    print(f'Score for : {score}')

    # --- File Management ---
    final_checkpoints_path, final_results_path = create_and_get_checkpoints_results_path(
        model_version=model_version,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    model_metadata_path = os.path.join(final_checkpoints_path, "best_rsf_model.joblib")
    joblib.dump(best_rsf, model_metadata_path)

    if results_path:
        run_summary = {
            "timestamp": datetime_for_folders,
            "metrics": {"test_rmse": rmse, "score": score},
            "parameters": best_rsf.get_params()
        }
        summary_file_path = os.path.join(final_results_path, "results.json")
        with open(summary_file_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=4)

    # `reproduce_result` passes a ModelVersion here despite the `str` annotation, and it is a
    # plain Enum, so formatting it directly yields "ModelVersion.RSF". Take the value so the
    # challenge files are named like every other model's ("rsf-challenge-scania.csv").
    model_version_name = (
        model_version.value if isinstance(model_version, ModelVersion) else model_version)

    # Additionally score the forest on the official Scania Component X held-out sets (test and
    # validation), one row per vehicle. Never raises: the run's own results are already written by
    # the time this runs.
    challenge_predict_rul_fn = lambda challenge: _predict_rul(
        best_rsf, challenge.X, challenge.Y['Time'])
    run_challenge_evaluation_rowwise(
        data_module=scania_data_module,
        predict_rul_fn=challenge_predict_rul_fn,
        model_version=model_version_name,
        results_path=final_results_path,
    )
    run_challenge_evaluation_rowwise(
        data_module=scania_data_module,
        predict_rul_fn=challenge_predict_rul_fn,
        model_version=model_version_name,
        results_path=final_results_path,
        split="validation",
    )

    return rmse, score


def _predict_rul(
        model: RandomSurvivalForest,
        features: ndarray,
        elapsed_times: ndarray,
) -> ndarray:
    """Predict a remaining useful life per row from a fitted Random Survival Forest.

    RSF models the survival function, not the RUL: the area under a row's predicted survival curve
    is its expected **total** lifetime, so the remaining life is that minus the time already
    observed.

    Args:
        model: A fitted ``RandomSurvivalForest``.
        features: ``(n_rows, n_features)`` feature matrix.
        elapsed_times: ``(n_rows,)`` observed time of each row (the survival ``Time`` field).

    Returns:
        The predicted RUL of every row, in time steps.
    """
    survival_funcs = model.predict_survival_function(features)
    predicted_total_times = [np.trapezoid(fn.y, fn.x) for fn in survival_funcs]

    return np.array(predicted_total_times) - elapsed_times


def select_best_params(
        model: RandomSurvivalForest,
        train_X: ndarray,
        train_Y: ndarray,
        cv_for_grid_search: int,
        n_estimators: list[int] | None,
        max_depth: list[int] | None,
        min_samples_split: list[int] | None,
        min_samples_leaf: list[int] | None,
        random_state: int | None,
        variance_warning_threshold: float,  # Pass threshold to grid search
) -> RandomSurvivalForest:
    param_grid = {}
    if n_estimators is not None: param_grid["n_estimators"] = n_estimators
    if max_depth is not None: param_grid["max_depth"] = max_depth
    if min_samples_split is not None: param_grid["min_samples_split"] = min_samples_split
    if min_samples_leaf is not None: param_grid["min_samples_leaf"] = min_samples_leaf

    status = train_Y["Status"]

    # Dynamically adjust folds based on failure density
    safe_cv_folds = calculate_safe_cv_folds(status, cv_for_grid_search)

    stratified_k_fold = StratifiedKFold(n_splits=safe_cv_folds, shuffle=True, random_state=random_state)
    # We need a custom split when there is no enough failure data and we wan't failure data for each fold
    custom_splits = list(stratified_k_fold.split(train_X, status))

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=custom_splits,
        n_jobs=-1,
        error_score='raise',
        verbose=1
    )

    print("Performing grid search...")
    grid_search.fit(train_X, train_Y)

    best_index = grid_search.best_index_
    best_std_score = grid_search.cv_results_['std_test_score'][best_index]

    print(f"The best params for RSF are: {grid_search.best_params_}")
    print(f"Best CV Score (Concordance Index): {grid_search.best_score_:.4f} (+/- {best_std_score:.4f})")

    # --- Robustness Warning for Grid Search ---
    if best_std_score > variance_warning_threshold:
        warnings.warn(
            f"\n[ROBUSTNESS WARNING] High variance detected for the best parameter set in Grid Search (std: {best_std_score:.4f} > {variance_warning_threshold}). "
            "Even though these are the 'best' parameters, their performance is unstable across folds. Consider checking data splits or increasing regularization.\n"
        )

    return grid_search.best_estimator_


def calculate_safe_cv_folds(
        status_array: np.ndarray,
        requested_folds: int,
        min_failures_per_fold: int = 2
) -> int:
    """
    Dynamically reduces the number of cross-validation folds if there are not enough
    failure events to support reliable Concordance Index calculations.
    """
    # Count the total number of actual failures (where Status is True or 1)
    total_failures = int(np.sum(status_array))

    # Extreme edge case: barely any failures in the entire dataset
    if total_failures < min_failures_per_fold * 2:
        warnings.warn(
            f"\n[CRITICAL DATA WARNING] Only {total_failures} failures detected in the entire dataset. "
            "Cross-validation will be highly unstable. Forcing folds to 2, but expect potential C-index calculation errors.\n"
        )
        return 2

    # Calculate the maximum folds we can safely create
    max_possible_folds = total_failures // min_failures_per_fold

    # Reduce folds if requested amount exceeds the safe limit
    if requested_folds > max_possible_folds:
        safe_folds = max_possible_folds
        warnings.warn(
            f"\n[FOLD REDUCTION] Requested {requested_folds} folds, but only {total_failures} failures are available. "
            f"To guarantee at least {min_failures_per_fold} failures per fold, "
            f"the number of CV folds has been automatically reduced to {safe_folds}.\n"
        )
        return safe_folds

    return requested_folds