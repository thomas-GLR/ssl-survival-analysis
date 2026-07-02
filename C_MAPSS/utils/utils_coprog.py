import os
from datetime import datetime

import torch

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.models import CNN1D, Simple_LSTM
from C_MAPSS.utils import utils_cmapss
from models import Coprog
from C_MAPSS.utils import utils_cmapss


def train_model(
    checkpoints_path: str,
    results_path: str,
    model_version: str,
    device: str | None,
    # Model params
    lstm_num_layers: int,
    hidden_dim: int,
    lstm_dropout: float,
    fc_layer_dim: int,
    fc_dropout: float,
    coprog_iterations: int,
    coprog_suspension_pool_size: int,
    # Training params
    lr_first_model: float,
    lr_second_model: float,
    epochs_first_model: int,
    epochs_second_model: int,
    batch_size_first_model: int,
    batch_size_second_model: int,
    # Dataset params
    dataset_root: str,
    seed: int | None,
    sub_dataset: str,
    sequence_len: int,
    max_rul: int=125,
    return_sequence_label: bool=False,
    norm_type: str='z-score',
    cluster_operations: bool=True,
    norm_by_operations: bool=True,
    include_cols: list[str] | None=None,
    exclude_cols: list[str] | None=None,
    return_id: bool= False,
    validation_rate=0.2,
    use_only_final_on_test: bool=True,
    use_max_rul_on_test: bool=False,
    use_max_rul_on_valid: bool=True,
    percent_of_broken_data: float | None=None,
    percent_of_censored_data: float=0.9,
    datetime_for_folders=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    utils_cmapss.assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )

    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

    final_checkpoints_path, final_results_path = utils_cmapss.create_and_get_checkpoints_results_path(
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
        model_version=model_version,
        sub_dataset=sub_dataset,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

    print("Loading datasets...")

    train_dataset, test_dataset, _ = CMAPSSLoader.get_datasets(
        dataset_root=dataset_root,
        seed=seed,
        sub_dataset=sub_dataset,
        sequence_len=sequence_len,
        max_rul=max_rul,
        return_sequence_label=return_sequence_label,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
        return_id=return_id,
        validation_rate=validation_rate,
        use_only_final_on_test=use_only_final_on_test,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
    )

    features_uncensored, targets_uncensored, features_censored, ids_censored = train_dataset.get_censored_split_tensors()
    features_tensor, targets_tensor = test_dataset.get_features_targets()

    print("Creating first model (CNN1D)...")

    feature_num = len(train_dataset.feature_cols)

    cnn = CNN1D(
        num_features=feature_num,
    )

    print("Creating second model (LSTM)...")

    lstm = Simple_LSTM(
        feature_num=feature_num,
        sequence_len=sequence_len,
        lstm_num_layers=lstm_num_layers,
        hidden_dim=hidden_dim,
        lstm_dropout=lstm_dropout,
        fc_layer_dim=fc_layer_dim,
        fc_dropout=fc_dropout,
    )

    coprog = Coprog(
        first_model=cnn,
        second_model=lstm,
        lr_first_model=lr_first_model,
        lr_second_model=lr_second_model,
        epochs_first_model=epochs_first_model,
        epochs_second_model=epochs_second_model,
        batch_size_first_model=batch_size_first_model,
        batch_size_second_model=batch_size_second_model,
        verbose=1,
        device=device,
    )

    print(f"Training Coprog model...")

    coprog.train(
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        iterations=coprog_iterations,
        suspension_pool_size=coprog_suspension_pool_size
    )

    coprog.calculate_weights(
        x_test=features_tensor,
        target=targets_tensor,
        criteria_callback=cmapss_score,
        mode="min",
    )

    print("Saving first and second trained models...")

    torch.save(coprog.first_model, f"{final_checkpoints_path}/coprog_cnn.pth")
    torch.save(coprog.second_model, f"{final_checkpoints_path}/coprog_lstm.pth")

    y_hat = coprog.predict(features_tensor)

    rmse = torch.sqrt(torch.mean((targets_tensor - y_hat) ** 2))
    score = utils_cmapss.cmapss_score(y_hat.cpu().detach().numpy().flatten(), targets_tensor.cpu().detach().numpy().flatten())

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score}")

    return rmse.item(), score

def cmapss_score(predict: torch.Tensor, label: torch.Tensor) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = torch.exp(-error[error < 0] / a1) - 1
    neg_e = torch.exp(error[error >= 0] / a2) - 1
    return torch.sum(pos_e).item() + torch.sum(neg_e).item()
