from dataset.PyclusDataset import PyclusDataset
from models.SslPCT import SslPCT
import numpy as np

from utils.utils import cmapss_score

C_MAPSS_DIR = "data/C_MAPSS"


if __name__ == "__main__":
    sub_dataset = "FD001"
    max_rul = None
    validation_rate = 0.0
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
    )



    X_train, y_train = train_dataset.X, train_dataset.Y
    X_test, y_test = test_dataset.X, test_dataset.Y

    # print(f"Train : {y_train}")
    # print(f"Valid : {y_valid}")
    # print(f"Test : {y_test}")

    sslPCT = SslPCT()

    sslPCT.fit(X_train, y_train)

    y_true_rul = test_dataset.to_rul()  # exact, vient directement de true_rul

    y_hat = sslPCT.predict(test_dataset.X)
    y_pred_rul = PyclusDataset.survival_targets_to_rul(
        y_hat["true values"], test_dataset.time_grid, method='threshold', enforce_monotonic=True
    )

    rmse = np.sqrt(np.mean((y_true_rul - y_pred_rul) ** 2))

    print(f"Test RMSE: {rmse}")
    print(f"Score: {cmapss_score(y_pred_rul, y_true_rul)}")
