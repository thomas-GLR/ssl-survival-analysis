import os
import json
import joblib
import warnings
import numpy as np
from datetime import datetime
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from dataset.PyclusDataset import PyclusDataset
from models import SslPCT
from C_MAPSS.utils import utils_cmapss


def train_model(
        checkpoints_path: str,
        results_path: str,
        # Model Parameters
        min_leaf_size: list[int] | None,
        n_trees: list[int] | None,
        max_depth: list[int] | None,
        pruning_method: str,
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
        seed: int | None = 42,
        model_version: str = "ssl-pct",
        variance_warning_threshold: float = 5.0,  # Adjusted scale for RMSE variance
        device: str | None = None,
        datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float| None, float| None]:
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

    train_dataset, test_dataset, _ = PyclusDataset.from_cmapss(
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
        summarize_features=summarize_features
    )

    X_train, Y_train = train_dataset.X, train_dataset.Y
    X_test, Y_test = test_dataset.X, test_dataset.Y

    percentage_labeled = int((1 - percent_of_censored_data) * 100)

    # Determine if Grid Search is required
    has_multiple_params = (
            (min_leaf_size and len(min_leaf_size) > 1)
            or (n_trees and len(n_trees) > 1)
            or (max_depth and len(max_depth) > 1)
    )

    if has_multiple_params:
        print("Multiple parameters detected. Executing custom grid search for SslPCT...")
        best_model, best_params = custom_ssl_grid_search(
            X_train, Y_train, train_dataset.time_grid,
            cv_for_grid_search,
            min_leaf_size, n_trees, max_depth,
            pruning_method, percentage_labeled,
            seed, variance_warning_threshold
        )
        ssl_pct = best_model
    else:
        print("No grid search needed. Evaluating model stability via Cross-Validation...")
        best_params = {
            "min_leaf_size": min_leaf_size[0] if min_leaf_size else 5,
            "n_trees": n_trees[0] if n_trees else 100,
            "max_depth": max_depth[0] if max_depth else 10,
        }
        print(f"The parameters are: {best_params}")

        ssl_pct = SslPCT(
            **best_params,
            pruning_method=pruning_method,
            percentage_labeled=percentage_labeled,
            is_multi_target=True,
        )

        evaluate_ssl_stability(
            ssl_pct, X_train, Y_train, train_dataset.time_grid,
            cv_for_grid_search, seed, variance_warning_threshold
        )

        print("Fitting the final model on the entire training dataset...")
        ssl_pct.fit(X_train, Y_train)

    # Final Predictions and Evaluations
    y_true_rul = test_dataset.to_rul()

    y_hat = ssl_pct.predict(X_test)
    y_pred_rul = PyclusDataset.survival_targets_to_rul(
        y_hat["true values"], test_dataset.time_grid, method='threshold', enforce_monotonic=True
    )

    rmse = float(np.sqrt(np.mean((y_true_rul - y_pred_rul) ** 2)))
    score = float(utils_cmapss.cmapss_score(y_pred_rul, y_true_rul))

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score}")

    # 1. Save Best Model Checkpoint
    final_checkpoints_path, final_results_path = utils_cmapss.create_and_get_checkpoints_results_path(
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
        model_version=model_version,
        sub_dataset=sub_dataset,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    model_metadata_path = os.path.join(final_checkpoints_path, "best_ssl_pct_model.joblib")
    joblib.dump(ssl_pct, model_metadata_path)
    print(f"Best model successfully saved to: {model_metadata_path}")

    # 2. Save Metrics and Parameters Document
    if results_path:
        run_summary = {
            "dataset": sub_dataset,
            "timestamp": datetime_for_folders,
            "metrics": {
                "test_rmse": rmse,
                "cmapss_score": score
            },
            "parameters": best_params,
            "global_parameters": {
                "w_parameter": "Global hyperparameter inherited from class defaults"
            }
        }

        summary_file_path = os.path.join(final_results_path, "results.json")
        with open(summary_file_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=4)
        print(f"Training summary saved to: {summary_file_path}")

    return rmse, score


def custom_ssl_grid_search(
        X_train, Y_train, time_grid,
        cv_folds: int,
        min_leaf_size_grid: list[int],
        n_trees_grid: list[int],
        max_depth_grid: list[int],
        pruning_method: str,
        percentage_labeled: int,
        random_state: int,
        variance_warning_threshold: float
):
    param_grid = {
        "min_leaf_size": min_leaf_size_grid,
        "n_trees": n_trees_grid,
        "max_depth": max_depth_grid,
    }
    grid = list(ParameterGrid(param_grid))

    status = Y_train["Status"]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_train, status))

    best_params = None
    best_rmse = float('inf')
    best_std = 0.0

    for params in grid:
        fold_rmses = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]

            model = SslPCT(
                min_leaf_size=params["min_leaf_size"],
                n_trees=params["n_trees"],
                max_depth=params["max_depth"],
                pruning_method=pruning_method,
                percentage_labeled=percentage_labeled,
                is_multi_target=True,
            )
            model.fit(X_tr, Y_tr)

            y_hat_val = model.predict(X_val)
            y_pred_rul_val = PyclusDataset.survival_targets_to_rul(
                y_hat_val["true values"], time_grid, method='threshold', enforce_monotonic=True
            )

            # Use Y_val['Time'] as the true validation RUL target
            true_rul_val = Y_val['Time']
            rmse_val = float(np.sqrt(np.mean((true_rul_val - y_pred_rul_val) ** 2)))
            fold_rmses.append(rmse_val)

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_std = std_rmse
            best_params = params

    print(f"Best CV RMSE: {best_rmse:.4f} (+/- {best_std:.4f})")

    if best_std > variance_warning_threshold:
        warnings.warn(
            f"\n[ROBUSTNESS WARNING] High variance detected for the best parameter set (std: {best_std:.4f} > {variance_warning_threshold}). "
            "Performance fluctuates significantly across folds. This indicates potential instability.\n"
        )

    print("Fitting the final model on the entire training dataset with best parameters...")
    final_model = SslPCT(
        **best_params,
        pruning_method=pruning_method,
        percentage_labeled=percentage_labeled,
        is_multi_target=True,
    )
    final_model.fit(X_train, Y_train)

    return final_model, best_params


def evaluate_ssl_stability(
        model: SslPCT,
        X_train, Y_train, time_grid,
        cv_folds: int,
        random_state: int,
        variance_warning_threshold: float
):
    status = Y_train["Status"]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_train, status))

    fold_rmses = []
    for train_idx, val_idx in splits:
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]

        # Instantiate a fresh model clone to prevent state retention
        fold_model = SslPCT(
            **model.get_params() if hasattr(model, 'get_params') else {
                "min_leaf_size": model.min_leaf_size,
                "n_trees": model.n_trees,
                "max_depth": model.max_depth,
                "pruning_method": model.pruning_method,
                "percentage_labeled": model.percentage_labeled,
                "is_multi_target": True
            }
        )

        fold_model.fit(X_tr, Y_tr)
        y_hat_val = fold_model.predict(X_val)
        y_pred_rul_val = PyclusDataset.survival_targets_to_rul(
            y_hat_val["true values"], time_grid, method='threshold', enforce_monotonic=True
        )

        true_rul_val = Y_val['Time']
        rmse_val = float(np.sqrt(np.mean((true_rul_val - y_pred_rul_val) ** 2)))
        fold_rmses.append(rmse_val)

    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)

    print(f"CV RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")

    if std_rmse > variance_warning_threshold:
        warnings.warn(
            f"\n[ROBUSTNESS WARNING] High variance detected in cross-validation (std: {std_rmse:.4f} > {variance_warning_threshold}). "
            "The model's performance fluctuates significantly across folds.\n"
        )