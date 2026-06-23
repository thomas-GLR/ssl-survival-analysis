from dataset.PyclusDataset import PyclusDataset
from models.SslPCT import SslPCT
import numpy as np

from utils.utils import cmapss_score

C_MAPSS_DIR = "data/C_MAPSS"


if __name__ == "__main__":
    sub_dataset = "FD001"
    max_rul = None
    validation_rate = 0.0
    norm_type = "z-score"
    summarize_features = True
    cluster_operations = False
    norm_by_operations = False
    use_max_rul_on_test = False
    use_max_rul_on_valid = True
    percent_of_censored_data = 0.9
    percent_of_broken_data = None
    seed = 42

    train_dataset, test_dataset, _ = PyclusDataset.from_cmapss(
        dataset_root=C_MAPSS_DIR,
        sub_dataset=sub_dataset,
        max_rul=max_rul,
        seed=seed,
        validation_rate=validation_rate,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
        summarize_features=summarize_features
    )

    X_train, y_train = train_dataset.X, train_dataset.Y
    X_test, y_test = test_dataset.X, test_dataset.Y

    percentage_labeled = int((1 - percent_of_censored_data) * 100)

    sslPCT = SslPCT(
        min_leaf_size=1,
        n_trees=100,
        max_death=5,
        pruning_method="M5MultiTarget",
        percentage_labeled=percentage_labeled,
        is_multi_target=True,
    )

    sslPCT.fit(X_train, y_train)

    y_true_rul = test_dataset.to_rul()

    y_hat = sslPCT.predict(test_dataset.X)
    y_pred_rul = PyclusDataset.survival_targets_to_rul(
        y_hat["true values"], test_dataset.time_grid, method='threshold', enforce_monotonic=True
    )

    rmse = np.sqrt(np.mean((y_true_rul - y_pred_rul) ** 2))

    print(f"Test RMSE: {rmse}")
    print(f"Score: {cmapss_score(y_pred_rul, y_true_rul)}")
