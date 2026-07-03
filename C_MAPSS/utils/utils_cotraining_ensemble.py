import shutil
import tempfile
from typing import Callable

import torch
from lightning import Trainer, LightningModule
from lightning.pytorch import callbacks
from torch import nn

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning_module.TransformerLstmModule import TransformerLstmModule
from C_MAPSS.models import CNN1D
from C_MAPSS.utils import utils_cmapss
from C_MAPSS.models import Simple_LSTM
from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2, SelectionMode
from C_MAPSS.models.TransformerFeatures import TransformerFeatures
from C_MAPSS.models.TransformerTimeSequence import TransformerTimeSequence


def train_model(
    coprog_iterations: int,
    coprog_suspension_pool_size: int,
    # Model params
    max_epochs: int,
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
) -> tuple[float, float]:

    print("Loading datasets...")

    dataset_params = {
        "dataset_root": dataset_root,
        "seed": seed,
        "sub_dataset": sub_dataset,
        "sequence_len": sequence_len,
        "max_rul": max_rul,
        "return_sequence_label": return_sequence_label,
        "norm_type": norm_type,
        "cluster_operations": cluster_operations,
        "norm_by_operations": norm_by_operations,
        "include_cols": include_cols,
        "exclude_cols": exclude_cols,
        "return_id": return_id,
        "validation_rate": validation_rate,
        "use_only_final_on_test": use_only_final_on_test,
        "use_max_rul_on_test": use_max_rul_on_test,
        "use_max_rul_on_valid": use_max_rul_on_valid,
        "percent_of_broken_data": percent_of_broken_data,
        "percent_of_censored_data": percent_of_censored_data,
    }

    print(f"Dataset params are : {dataset_params}")

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
            "The co-training ensemble needs a validation set for early stopping / best-model "
            "selection and for the ensemble weights. Set validation_rate > 0."
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

    lstm = Simple_LSTM(
        feature_num=feature_num,
        sequence_len=sequence_len,
        hidden_dim=128,
        lstm_num_layers=1,
        lstm_dropout=0.3,
        fc_layer_dim=32,
        fc_dropout=0.4,
    )

    transformer_features = TransformerFeatures(
        feature_num=feature_num,
        d_model=sequence_len,
        transformer_encoder_head_num=8,
        fc_layer_dim=32,
        fc_dropout=0.4,
    )

    transformer_time_sequence = TransformerTimeSequence(
        feature_num=feature_num,
        d_model=sequence_len,
        transformer_encoder_head_num=8,
        fc_layer_dim=32,
        fc_dropout=0.4,
    )

    models = [cnn, lstm, transformer_features, transformer_time_sequence]

    cotraining_ensemble = CoTrainingEnsemble_v2(
        models=models,
        verbose=2
    )

    models_number = len(models)

    max_epochs = max_epochs
    patience = 50

    batchs_size = [128 for _ in range(models_number)]
    shuffle_dataloaders = [True for _ in range(models_number)]

    lightning_modules = [TransformerLstmModule(lr=0.0002, model=model) for model in models]

    # Each _train_fun call builds a fresh Trainer from these factories. The ModelCheckpoint
    # lets the ensemble reload the best (val_loss) weights instead of the last-epoch ones, and
    # EarlyStopping avoids over/under-training. Checkpoints go to throwaway temp dirs that we
    # remove at the end (there is one fit per candidate, so this can be a lot of files).
    created_ckpt_dirs: list[str] = []

    def make_trainer() -> Trainer:
        ckpt_dir = tempfile.mkdtemp(prefix="cotraining_ckpt_")
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
            max_epochs=max_epochs,
            accelerator="auto",
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    trainer_factories: list[Callable[[], Trainer]] = [make_trainer] * models_number

    fine_tune_trainable_params: list[Callable[[LightningModule], list[nn.Parameter]]] = [
        lambda lm: list(lm.net.regressor.parameters()),
        lambda lm: list(lm.net.linear.parameters()),
        lambda lm: list(lm.net.linear.parameters()),
        lambda lm: list(lm.net.linear.parameters()),
    ]

    cotraining_ensemble.setup_training(
        lightning_modules=lightning_modules,
        trainer_factories=trainer_factories,
        batchs_size=batchs_size,
        shuffle_dataloaders=shuffle_dataloaders,
        fine_tune_trainable_params=fine_tune_trainable_params
    )

    print(f"Training Coprog model...")

    try:
        cotraining_ensemble.train(
            is_fine_tuning_during_finding_best_suspension_data=True,
            is_fine_tuning_for_last_step=False,
            selection_mode=SelectionMode.VOTING,
            train_with_censored_data=False,
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
        cotraining_ensemble.calculate_weights(
            x_test=val_features,
            target=val_targets,
            criteria_callback=cmapss_score,
            mode="min",
        )
    finally:
        for ckpt_dir in created_ckpt_dirs:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Flatten both sides so we compute a real element-wise RMSE. targets_tensor is (N, 1)
    # and predict() returns (N, 1); flattening keeps the comparison aligned as (N,).
    y_hat = cotraining_ensemble.predict(features_tensor).detach().cpu().view(-1)
    targets_flat = targets_tensor.detach().cpu().view(-1)

    rmse = torch.sqrt(torch.mean((targets_flat - y_hat) ** 2))
    score = utils_cmapss.cmapss_score(y_hat.numpy(), targets_flat.numpy())

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


if __name__ == "__main__":

    coprog_iterations = 10
    coprog_suspension_pool_size = 25
    dataset_root = "data/C_MAPSS"
    seed = 42
    sub_dataset = "FD002"
    sequence_len = 32
    max_rul = 125
    return_sequence_label = False
    norm_type = "z-score"
    cluster_operations = True
    norm_by_operations = True
    include_cols = []
    exclude_cols = []
    return_id = False
    validation_rate = 0.2
    use_only_final_on_test = True
    use_max_rul_on_test = True
    use_max_rul_on_valid = True
    percent_of_broken_data = 0.9
    percent_of_censored_data = 0.7

    train_model(
        coprog_iterations=coprog_iterations,
        coprog_suspension_pool_size=coprog_suspension_pool_size,
        max_epochs=1,
        dataset_root="../../data/C_MAPSS",
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
