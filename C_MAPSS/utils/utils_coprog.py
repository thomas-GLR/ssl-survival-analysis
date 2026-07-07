import shutil
import tempfile
from datetime import datetime
from typing import Callable

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch import callbacks

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning_module import TransformerLstmModule
from models import CNN1D, Simple_LSTM
from C_MAPSS.utils import utils_cmapss
from models import Coprog


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
    patience: int,
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

    train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(
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

    if valid_dataset is None:
        raise ValueError(
            "Coprog needs a validation set for early stopping / best-model selection and for the "
            "ensemble weights. Set validation_rate > 0 in the config."
        )

    features_uncensored, targets_uncensored, features_censored, ids_censored = train_dataset.get_censored_split_tensors()
    features_tensor, targets_tensor = test_dataset.get_features_targets()

    # Labelled (uncensored) validation data: used both for early stopping / best-checkpoint
    # selection during training and to compute the ensemble weights (instead of the test set).
    val_features, val_targets, _, _ = valid_dataset.get_censored_split_tensors()

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
        verbose=1,
    )

    # Wrap each model in a Lightning module (TransformerLstmModule works for CNN too).
    cnn_module = TransformerLstmModule(lr=lr_first_model, model=cnn)
    lstm_module = TransformerLstmModule(lr=lr_second_model, model=lstm)

    # Each _train_fun call builds a fresh Trainer from these factories. The ModelCheckpoint
    # lets Coprog reload the best (val_loss) weights instead of the last-epoch ones, and
    # EarlyStopping avoids over/under-training. Checkpoints go to throwaway temp dirs that
    # we remove at the end (there is one fit per candidate, so this can be a lot of files).
    created_ckpt_dirs: list[str] = []

    def make_trainer_factory(max_epochs: int) -> Callable[[], Trainer]:
        def factory() -> Trainer:
            ckpt_dir = tempfile.mkdtemp(prefix="coprog_ckpt_")
            created_ckpt_dirs.append(ckpt_dir)

            early_stop_callback = callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode='min',
            )
            checkpoint_callback = callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor='val_loss',
                filename='best-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min',
            )

            return Trainer(
                default_root_dir=ckpt_dir,
                accelerator="auto",
                max_epochs=max_epochs,
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

        return factory

    coprog.setup_training(
        lightning_modules=[cnn_module, lstm_module],
        trainer_factories=[
            make_trainer_factory(epochs_first_model),
            make_trainer_factory(epochs_second_model),
        ],
        batch_sizes=[batch_size_first_model, batch_size_second_model],
        shuffle_dataloaders=[True, True],
    )

    print(f"Training Coprog model...")

    try:
        coprog.train(
            failure_data=features_uncensored,
            failure_label=targets_uncensored,
            suspension_data=features_censored,
            suspension_ids=ids_censored,
            iterations=coprog_iterations,
            suspension_pool_size=coprog_suspension_pool_size,
            val_data=val_features,
            val_label=val_targets,
        )

        # Ensemble weights are computed on the validation set, not the test set,
        # to avoid leaking test information into the weighting.
        coprog.calculate_weights(
            x_test=val_features,
            target=val_targets,
            criteria_callback=cmapss_score,
            mode="min",
        )
    finally:
        for ckpt_dir in created_ckpt_dirs:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    print("Saving first and second trained models...")

    torch.save(coprog._h1, f"{final_checkpoints_path}/coprog_cnn.pth")
    torch.save(coprog._h2, f"{final_checkpoints_path}/coprog_lstm.pth")

    # Flatten both sides so we compute a real element-wise RMSE. targets_tensor is (N, 1)
    # and predict() returns (N,); subtracting them directly would broadcast to (N, N).
    y_hat = coprog.predict(features_tensor).detach().cpu().view(-1)
    targets_flat = targets_tensor.detach().cpu().view(-1)

    rmse = torch.sqrt(torch.mean((targets_flat - y_hat) ** 2))
    score = utils_cmapss.cmapss_score(y_hat.numpy(), targets_flat.numpy())

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score}")

    scores = pd.DataFrame(columns=['test_rmse', 'test_score', 'weight_h1', 'weight_h2'])

    scores.loc[0] = [rmse.item(), score, coprog.w1, coprog.w2]

    # Save the results
    scores.to_csv(f'{final_results_path}/{model_version}-turbofan-{sub_dataset}.csv', index=False)

    return rmse.item(), score

def cmapss_score(predict: torch.Tensor, label: torch.Tensor) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = torch.exp(-error[error < 0] / a1) - 1
    neg_e = torch.exp(error[error >= 0] / a2) - 1
    return torch.sum(pos_e).item() + torch.sum(neg_e).item()