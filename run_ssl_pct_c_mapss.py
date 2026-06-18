from dataset.PyclusDataset import PyclusDataset
from models.SslPCT import SslPCT

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

    y_hat_valid = sslPCT.predict(X_test)

    print(y_test)
    print(y_hat_valid["true values"])

    y_hat = sslPCT.predict(X_test)

    print(y_test)
    print(y_hat["true values"])
