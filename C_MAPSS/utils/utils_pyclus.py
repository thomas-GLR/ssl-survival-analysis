import numpy as np

from dataset.PyclusDataset import PyclusDataset
from models import SslPCT
from utils.utils import cmapss_score


def train_model(
        # Dataset params
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
        random_state: int | None,
        summarize_features: bool,
        # Model
        min_leaf_size: int,
        n_trees: int,
        max_death: int,
        pruning_method: str,
) -> tuple[float, float]:
    """

    :param dataset_root:
    :param sub_dataset:
    :param max_rul:
    :param norm_type:
    :param cluster_operations:
    :param norm_by_operations:
    :param use_max_rul_on_test:
    :param use_max_rul_on_valid:
    :param percent_of_censored_data:
    :param percent_of_broken_data:
    :param random_state:
    :param summarize_features:
    :param min_leaf_size:
    :param n_trees:
    :param max_death:
    :param pruning_method:
        - The pruning method for regression trees is M5
        - For multi-target regression trees, the pruning method is M5MultiTarget
        - CartVSB work better than M5 on multi-target regression
    :return:
    """
    print("Loading dataset...")

    train_dataset, test_dataset, _ = PyclusDataset.from_cmapss(
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
        summarize_features=summarize_features
    )

    X_train, Y_train = train_dataset.X, train_dataset.Y
    X_test, Y_test = test_dataset.X, test_dataset.Y

    percentage_labeled = int((1 - percent_of_censored_data) * 100)

    ssl_pct = SslPCT(
        min_leaf_size=min_leaf_size,
        n_trees=n_trees,
        max_death=max_death,
        pruning_method=pruning_method,
        percentage_labeled=percentage_labeled,
        is_multi_target=True,
    )

    ssl_pct.fit(X_train, Y_train)

    y_true_rul = test_dataset.to_rul()

    y_hat = ssl_pct.predict(X_test)
    y_pred_rul = PyclusDataset.survival_targets_to_rul(
        y_hat["true values"], test_dataset.time_grid, method='threshold', enforce_monotonic=True
    )

    rmse = np.sqrt(np.mean((y_true_rul - y_pred_rul) ** 2))
    score = cmapss_score(y_pred_rul, y_true_rul)

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score}")

    return rmse, score
