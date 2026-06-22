import numpy as np
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.models import Simple_LSTM, CNN1D
from models import Coprog
import os
import torch

from utils.utils import score

C_MAPSS_DIR = "data/C_MAPSS"


def get_simple_lstm_model(
        feature_num: int,
        sequence_len: int,
        hidden_dim: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        fc_layer_dim: int,
        fc_dropout: float,
        model_path: str = "",
        force_creation: bool = False
) -> Simple_LSTM:
    if not force_creation and os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        return torch.load(model_path, weights_only=False)

    return Simple_LSTM(
        feature_num=feature_num,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout
    )


def get_cnn_1d_model(
        feature_num: int,
        model_path: str = "",
        force_creation: bool = False
):
    if not force_creation and os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        return torch.load(model_path, weights_only=False)

    return CNN1D(num_features=feature_num)


if __name__ == "__main__":
    first_model_path = "coprog_first_model.pth"
    second_model_path = "coprog_second_model.pth"

    train_dataset, test_dataset, _ = CMAPSSLoader.get_datasets(
        dataset_root=C_MAPSS_DIR,
        sub_dataset="FD001",
        sequence_len=30,
        max_rul=125,
        percent_of_broken_data=None,
        percent_of_censored_data=0.9,
        norm_type="z-score",
        cluster_operations=True,
        norm_by_operations=True,
        validation_rate=0
    )

    first_model = get_simple_lstm_model(
        model_path=first_model_path,
        feature_num=24,
        sequence_len=30,
        hidden_dim=32,
        lstm_num_layers=3,
        lstm_dropout=0.2,
        fc_layer_dim=32,
        fc_dropout=0.2,
    )

    second_model = get_cnn_1d_model(
        model_path=second_model_path,
        feature_num=24,
    )

    coprog_already_trained = os.path.isfile(first_model_path) and os.path.isfile(second_model_path)

    coprog = Coprog(
        first_model=first_model,
        second_model=second_model,
        batch_size=128,
        epochs=100,
        verbose=1,
        first_and_second_model_already_trained=coprog_already_trained
    )

    features_uncensored, targets_uncensored, features_censored = train_dataset.get_censored_split_tensors()
    features_tensor, targets_tensor = test_dataset.get_features_targets()

    if not coprog_already_trained:
        print("Training coprog...")

        coprog.train(
            failure_data=features_uncensored,
            failure_label=targets_uncensored,
            suspension_data=features_censored,
            iterations=5,
            suspension_pool_size=5  # int(len(features_censored) * 0.5)
        )

        print("Saving first and second trained models...")

        torch.save(coprog.first_model, first_model_path)
        torch.save(coprog.second_model, second_model_path)

    # y_hat = coprog.predict(features_tensor)
    #
    # print(y_hat)
    # print(targets_tensor)


    y_hat = coprog.predict(features_tensor)

    print(y_hat)
    print(targets_tensor.flatten())

    rmse = torch.sqrt(torch.mean((targets_tensor - y_hat) ** 2))

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score(y_hat, targets_tensor)}")
