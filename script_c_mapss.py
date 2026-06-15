import os
import pandas as pd
import numpy as np
from pathlib import Path

from C_MAPSS.dataset import CMAPSSDataset
from models.SslPCT import SslPCT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCANIA_COMPONENT_X_DIR = DATA_DIR / "scania_components_x"
C_MAPSS_DIR = DATA_DIR / "C_MAPSS"


if __name__ == "__main__":
    sub_dataset = "FD001"
    max_rul = None
    validation_rate = 0.2
    use_max_rul_on_test = False
    use_max_rul_on_valid = True

    train_dataset, test_dataset, valid_dataset = CMAPSSDataset.get_data_for_ssl_pct(
        dataset_root="data\\C_MAPSS",
        sub_dataset=sub_dataset,
        max_rul=max_rul,
        validation_rate=validation_rate,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid
    )

    X_train, y_train = train_dataset.X, train_dataset.Y
    X_valid, y_valid = valid_dataset.X, valid_dataset.Y
    X_test, y_test = test_dataset.X, test_dataset.Y

    # print(f"Train : {y_train}")
    # print(f"Valid : {y_valid}")
    # print(f"Test : {y_test}")

    sslPCT = SslPCT()

    sslPCT.fit(X_train, y_train)

    y_hat_valid = sslPCT.predict(X_valid)

    print(y_valid)
    print(y_hat_valid["true values"])

    y_hat = sslPCT.predict(X_test)

    print(y_test)
    print(y_hat["true values"])
