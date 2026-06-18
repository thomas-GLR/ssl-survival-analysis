from dataset.PyclusDataset import PyclusDataset
from models.SslPCT import SslPCT
import numpy as np

C_MAPSS_DIR = "data/C_MAPSS"


if __name__ == "__main__":
    sub_dataset = "FD001"
    max_rul = None
    validation_rate = 0.0
    use_max_rul_on_test = False
    use_max_rul_on_valid = True
    percent_of_censored_data = 0.9
    percent_of_broken_data = None

    train_dataset, test_dataset, _ = PyclusDataset.from_cmapss(
        dataset_root=C_MAPSS_DIR,
        sub_dataset=sub_dataset,
        max_rul=max_rul,
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


    def score(y_pred, y_true):
        d = y_pred - y_true
        return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


    print(f"Test RMSE: {rmse}")
    print(f"Score: {score(y_pred_rul, y_true_rul)}")
