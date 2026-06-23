import numpy as np
from numpy import ndarray
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest

from dataset.ScikitDataset import ScikitDataset
from utils.utils import cmapss_score


def train_model(
        n_estimators: list[int] | None,
        max_depth: list[int] | None,
        min_samples_split: list[int] | None,
        min_samples_leaf: list[int] | None,
        cv_for_grid_search: int,
        dataset_root: str,
        sub_dataset: str,
        max_rul: int,
        norm_type: str,
        cluster_operations: bool,
        norm_by_operations: bool,
        use_max_rul_on_test: bool,
        use_max_rul_on_valid: bool,
        percent_of_censored_data: float,
        percent_of_broken_data: float | None,
        summarize_features: bool,
        random_state: int=42,
) -> tuple[float, float]:
    rsf = RandomSurvivalForest(
        n_jobs=-1,
        random_state=random_state,
    )

    print("Loading dataset...")

    cluster_operations = False if summarize_features else cluster_operations
    norm_by_operations = False if summarize_features else norm_by_operations

    train_dataset, test_dataset, _ = ScikitDataset.from_cmapss(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        seed=random_state,
        max_rul=max_rul,
        norm_type=norm_type,
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

    if ((n_estimators and len(n_estimators) > 1)
            or (max_depth and len(max_depth) > 1)
            or (min_samples_leaf and len(min_samples_leaf) > 1)
            or (min_samples_split and len(min_samples_split) > 1)):
        print("Multiple params the programe will perform a grid search...")

        best_rsf = select_best_params(
            rsf,
            train_X,
            train_Y,
            cv_for_grid_search,
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            random_state,
        )

    else:
        print("No grid search needed the model will be create with the given parameters...")

        best_params = {}

        if n_estimators is not None and len(n_estimators) > 0:
            best_params["n_estimators"] = n_estimators[0]

        if max_depth is not None and len(max_depth) > 0:
            best_params["max_depth"] = max_depth[0]

        if min_samples_split is not None and len(min_samples_split) > 0:
            best_params["min_samples_split"] = min_samples_split[0]

        if min_samples_leaf is not None and len(min_samples_leaf) > 0:
            best_params["min_samples_leaf"] = min_samples_leaf[0]

        print("The parameters are : ", best_params)

        best_rsf = RandomSurvivalForest(
            **best_params,
            n_jobs=-1,
            random_state=random_state,
        )

        best_rsf.fit(train_X, train_Y)

    survival_funcs = best_rsf.predict_survival_function(test_X)

    predicted_total_times = []
    for fn in survival_funcs:
        # The area under the survival curve provides the estimated total time-to-failure (T_total)
        # RUL = T_total - current operating time
        predicted_total_times.append(np.trapezoid(fn.y, fn.x))

    predicted_ruls = np.array(predicted_total_times) - test_Y['Time']

    rmse = np.sqrt(mean_squared_error(test_dataset.rul, predicted_ruls))
    score = cmapss_score(predicted_ruls, test_dataset.rul)

    print(f'Test RMSE for {sub_dataset}: {rmse}')
    print(f'Score for {sub_dataset}: {score}')

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
) -> RandomSurvivalForest:
    param_grid = {}

    if n_estimators is not None:
        param_grid["n_estimators"] = n_estimators

    if max_depth is not None:
        param_grid["max_depth"] = max_depth

    if min_samples_split is not None:
        param_grid["min_samples_split"] = min_samples_split

    if min_samples_leaf is not None:
        param_grid["min_samples_leaf"] = min_samples_leaf

    status = train_Y["Status"]

    stratified_k_fold = StratifiedKFold(n_splits=cv_for_grid_search, shuffle=True, random_state=random_state)
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

    print(f"The best params for RSF is : {grid_search.best_params_}")

    return grid_search.best_estimator_
