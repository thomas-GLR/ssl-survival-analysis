import os
import json
import joblib
import warnings
import numpy as np
from datetime import datetime
from numpy import ndarray
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sksurv.ensemble import RandomSurvivalForest

from dataset.ScikitDataset import ScikitDataset
from utils import utils_cmapss
from utils.utils import cmapss_score


def train_model(
        checkpoints_path: str,
        results_path: str,
        n_estimators: list[int] | None,
        max_depth: list[int] | None,
        min_samples_split: list[int] | None,
        min_samples_leaf: list[int] | None,
        cv_for_grid_search: int,
        # Dataset params
        dataset_root: str,
        sub_dataset: str,
        max_rul: int,
        norm_type: str,
        include_cols: list[str],
        exclude_cols: list[str],
        cluster_operations: bool,
        norm_by_operations: bool,
        use_max_rul_on_test: bool,
        use_max_rul_on_valid: bool,
        percent_of_censored_data: float,
        percent_of_broken_data: float | None,
        summarize_features: bool,
        seed: int = 42,
        model_version: str = "rsf",
        device: str | None = None,
        variance_warning_threshold: float = 0.05,  # Threshold for stability warning
        datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float| None, float| None]:
    rsf = RandomSurvivalForest(
        n_jobs=-1,
        random_state=seed,
    )
    utils_cmapss.assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )
    if percent_of_censored_data > 0.80:
        print("There is no enough failure data for rsf on CMAPSS")
        return None, None

    print("Loading dataset...")

    cluster_operations = False if summarize_features else cluster_operations
    norm_by_operations = False if summarize_features else norm_by_operations

    train_dataset, test_dataset, _ = ScikitDataset.from_cmapss(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        seed=seed,
        max_rul=max_rul,
        norm_type=norm_type,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
        summarize_features=summarize_features,
    )

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
    survival_funcs = best_rsf.predict_survival_function(test_X)
    predicted_total_times = []
    for fn in survival_funcs:
        predicted_total_times.append(np.trapezoid(fn.y, fn.x))

    predicted_ruls = np.array(predicted_total_times) - test_Y['Time']

    rmse = float(np.sqrt(mean_squared_error(test_dataset.rul, predicted_ruls)))
    score = float(cmapss_score(predicted_ruls, test_dataset.rul))

    print(f'Test RMSE for {sub_dataset}: {rmse}')
    print(f'Score for {sub_dataset}: {score}')

    # --- File Management ---
    broken_percentage = percent_of_broken_data if percent_of_broken_data is not None else 0.0
    folder_for_current_training = (
        f"model-{model_version}-turbofan-{sub_dataset}-{datetime_for_folders}/"
        f"censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"
    )

    final_checkpoints_path = os.path.join(checkpoints_path, folder_for_current_training)
    os.makedirs(final_checkpoints_path, exist_ok=True)
    model_metadata_path = os.path.join(final_checkpoints_path, "best_rsf_model.joblib")
    joblib.dump(best_rsf, model_metadata_path)

    if results_path:
        final_results_path = os.path.join(results_path, folder_for_current_training)
        os.makedirs(final_results_path, exist_ok=True)
        run_summary = {
            "dataset": sub_dataset,
            "timestamp": datetime_for_folders,
            "metrics": {"test_rmse": rmse, "cmapss_score": score},
            "parameters": best_rsf.get_params()
        }
        summary_file_path = os.path.join(final_results_path, "results.json")
        with open(summary_file_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=4)

    return rmse, score


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
