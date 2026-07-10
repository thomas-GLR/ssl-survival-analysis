import os
import shutil
import tempfile
from datetime import datetime
from typing import Callable

import pandas as pd
import torch
from lightning import Trainer, LightningModule
from lightning.pytorch import callbacks
from torch import nn

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning_module.TransformerLstmModule import TransformerLstmModule
from C_MAPSS.models import CNN1D
from C_MAPSS.utils import utils_cmapss
from C_MAPSS.models import Simple_LSTM
from models.CoTrainingEnsemble import CoTrainingEnsemble, SelectionMode
from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2 as CoTrainingEnsembleV2
from C_MAPSS.models.TransformerFeatures import TransformerFeatures
from C_MAPSS.models.TransformerTimeSequence import TransformerTimeSequence


def train_model(
        checkpoints_path: str,
        results_path: str,
        model_version: str,
        # Training
        coprog_iterations: int,
        coprog_suspension_pool_size: int,
        is_fine_tuning_during_finding_best_suspension_data: bool,
        is_fine_tuning_for_last_step: bool,
        selection_mode_str: str,
        max_epochs: list[int],
        patiences: list[int],
        batchs_size: list[int],
        shuffle_dataloaders: list[bool],
        lr: list[float],
        fine_tune_lr_factor: float,
        forgetting_warning_tolerance: float,
        # Model params
        hidden_dim_lstm: int,
        lstm_num_layers_lstm: int,
        lstm_dropout_lstm: float,
        fc_layer_dim_lstm: int,
        fc_dropout_lstm: float,
        transformer_encoder_head_num_transformer_features: int,
        fc_layer_dim_transformer_features: int,
        fc_dropout_transformer_features: float,
        transformer_encoder_head_num_transformer_time_series: int,
        fc_layer_dim_transformer_time_series: int,
        fc_dropout_transformer_time_series: float,
        num_layers_transformer_features: int,
        num_layers_transformer_time_series: int,
        # Dataset params
        dataset_root: str,
        seed: int | None,
        sub_dataset: str,
        sequence_len: int,
        max_rul: int = 125,
        return_sequence_label: bool = False,
        norm_type: str = 'z-score',
        cluster_operations: bool = True,
        norm_by_operations: bool = True,
        include_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        return_id: bool = False,
        validation_rate=0.2,
        use_only_final_on_test: bool = True,
        use_max_rul_on_test: bool = False,
        use_max_rul_on_valid: bool = True,
        percent_of_broken_data: float | None = None,
        percent_of_censored_data: float = 0.9,
        # Others
        device: str | None = None,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    match selection_mode_str:
        case SelectionMode.VOTING.value:
            selection_mode = SelectionMode.VOTING
        case SelectionMode.EVIDENCE.value:
            selection_mode = SelectionMode.EVIDENCE
        case _:
            raise ValueError(f"Unknown selection_mode: {selection_mode_str}")

    if (len(batchs_size) != 4
            or len(patiences) != 4
            or len(shuffle_dataloaders) != 4
            or len(max_epochs) != 4
            or len(lr) != 4):
        raise ValueError(
            f"bachs_size {len(batchs_size)}, patiences {len(patiences)},"
            f"shuffle_dataloaders {len(shuffle_dataloaders)}, lr {len(lr)} and"
            f"max_epochs {len(max_epochs)} must be lists of length 4, one for each model in the ensemble."
        )

    utils_cmapss.assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )

    checkpoints_path, results_path = utils_cmapss.create_and_get_checkpoints_results_path(
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
        model_version=model_version,
        sub_dataset=sub_dataset,
        datetime_for_folders=datetime_for_folders,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
    )

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
        hidden_dim=hidden_dim_lstm,
        lstm_num_layers=lstm_num_layers_lstm,
        lstm_dropout=lstm_dropout_lstm,
        fc_layer_dim=fc_layer_dim_lstm,
        fc_dropout=fc_dropout_lstm,
    )

    transformer_features = TransformerFeatures(
        feature_num=feature_num,
        sequence_len=sequence_len,
        transformer_encoder_head_num=transformer_encoder_head_num_transformer_features,
        fc_layer_dim=fc_layer_dim_transformer_features,
        fc_dropout=fc_dropout_transformer_features,
        num_layers=num_layers_transformer_features,
    )

    transformer_time_sequence = TransformerTimeSequence(
        feature_num=feature_num,
        sequence_len=sequence_len,
        d_model=sequence_len,
        transformer_encoder_head_num=transformer_encoder_head_num_transformer_time_series,
        fc_layer_dim=fc_layer_dim_transformer_time_series,
        fc_dropout=fc_dropout_transformer_time_series,
        num_layers=num_layers_transformer_time_series,
    )

    models = [cnn, lstm, transformer_features, transformer_time_sequence]

    cotraining_ensemble = CoTrainingEnsemble(
        models=models,
        verbose=2,
        fine_tune_lr_factor=fine_tune_lr_factor,
        forgetting_warning_tolerance=forgetting_warning_tolerance,
    )

    models_number = len(models)

    lightning_modules = [TransformerLstmModule(lr=lr[j], model=model) for j, model in enumerate(models)]

    # Each _train_fun call builds a fresh Trainer from these factories. The ModelCheckpoint
    # lets the ensemble reload the best (val_loss) weights instead of the last-epoch ones, and
    # EarlyStopping avoids over/under-training. Checkpoints go to throwaway temp dirs that we
    # remove at the end (there is one fit per candidate, so this can be a lot of files).
    created_ckpt_dirs: list[str] = []

    def make_trainer_factory(patience: int, max_epoch: int) -> Callable[[], Trainer]:
        def factory() -> Trainer:
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
                max_epochs=max_epoch,
                accelerator="auto",
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

        return factory

    # One factory per model, each carrying that model's own epochs / patience.
    trainer_factories: list[Callable[[], Trainer]] = [
        make_trainer_factory(patiences[j], max_epochs[j]) for j in range(models_number)
    ]

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

    # Persistent run log next to the results. Created (truncated) here with a metadata
    # header, then handed to the ensemble which appends all its messages under it.
    log_file_path = os.path.join(results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Co-Training Ensemble run ===\n")
        f.write(f"Datetime: {datetime_for_folders}\n")
        f.write(f"Sub-dataset: {sub_dataset}\n")
        f.write(f"Percent censored: {percent_of_censored_data}\n")
        f.write(f"Percent broken: {percent_of_broken_data}\n")
        f.write("================================\n")

    try:
        cotraining_ensemble.train(
            is_fine_tuning_during_finding_best_suspension_data=is_fine_tuning_during_finding_best_suspension_data,
            is_fine_tuning_for_last_step=is_fine_tuning_for_last_step,
            selection_mode=selection_mode,
            train_with_censored_data=False,
            failure_data=features_uncensored,
            failure_label=targets_uncensored,
            suspension_data=features_censored,
            suspension_ids=ids_censored,
            iterations=coprog_iterations,
            suspension_pool_size=coprog_suspension_pool_size,
            val_data=val_features,
            val_label=val_targets,
            test_data=features_tensor,
            test_label=targets_tensor,
            criteria_callback=cmapss_score,
            weight_mode="min",
            metrics_file=os.path.join(results_path, "metrics_per_stage.csv"),
            log_file=log_file_path,
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

    # Persist the final trained models. Only the per-model best weights that survive
    # training (now held in cotraining_ensemble.lightning_modules) are saved to
    # checkpoints_path — the many throwaway per-candidate checkpoints above are gone.
    # We use the same {"state_dict": ...} layout that _fit_and_reload_best reloads,
    # so these files can be loaded back with torch.load(path)["state_dict"].
    for j, module in enumerate(cotraining_ensemble.lightning_modules):
        final_ckpt_path = os.path.join(
            checkpoints_path, f"{model_version}-turbofan-{sub_dataset}-model_{j}.ckpt"
        )
        torch.save({"state_dict": module.state_dict()}, final_ckpt_path)
        print(f"Saved final checkpoint for model {j} to {final_ckpt_path}")

    # Flatten both sides so we compute a real element-wise RMSE. targets_tensor is (N, 1)
    # and predict() returns (N, 1); flattening keeps the comparison aligned as (N,).
    y_hat = cotraining_ensemble.predict(features_tensor).detach().cpu().view(-1)
    targets_flat = targets_tensor.detach().cpu().view(-1)

    rmse = torch.sqrt(torch.mean((targets_flat - y_hat) ** 2))
    score = utils_cmapss.cmapss_score(y_hat.numpy(), targets_flat.numpy())

    columns = ['test_rmse', 'test_score']

    for j in range(models_number):
        columns.append(f"weight_{j}")

    scores = pd.DataFrame(columns=columns)

    row = [rmse, score]

    for j in range(models_number):
        if cotraining_ensemble is not None:
            row.append(cotraining_ensemble.weights[j])
        else:
            row.append(None)

    # Add the results to the dataframe
    scores.loc[0] = row

    scores.to_csv(f'{results_path}/{model_version}-turbofan-{sub_dataset}.csv', index=False)

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score}")

    return rmse.item(), score


def train_model_v2(
        checkpoints_path: str,
        results_path: str,
        model_version: str,
        # Training
        coprog_iterations: int,
        max_epochs: list[int],
        patiences: list[int],
        batchs_size: list[int],
        shuffle_dataloaders: list[bool],
        lr: list[float],
        confidence: float,
        # Model params
        hidden_dim_lstm: int,
        lstm_num_layers_lstm: int,
        lstm_dropout_lstm: float,
        fc_layer_dim_lstm: int,
        fc_dropout_lstm: float,
        transformer_encoder_head_num_transformer_features: int,
        fc_layer_dim_transformer_features: int,
        fc_dropout_transformer_features: float,
        transformer_encoder_head_num_transformer_time_series: int,
        fc_layer_dim_transformer_time_series: int,
        fc_dropout_transformer_time_series: float,
        num_layers_transformer_features: int,
        num_layers_transformer_time_series: int,
        # Dataset params
        dataset_root: str,
        seed: int | None = 42,
        sub_dataset: str = "FD001",
        sequence_len: int = 32,
        max_rul: int = 125,
        return_sequence_label: bool = False,
        norm_type: str = 'z-score',
        cluster_operations: bool = True,
        norm_by_operations: bool = True,
        include_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        return_id: bool = False,
        validation_rate=0.2,
        use_only_final_on_test: bool = True,
        use_max_rul_on_test: bool = False,
        use_max_rul_on_valid: bool = True,
        percent_of_broken_data: float | None = None,
        percent_of_censored_data: float = 0.9,
        # Others
        device: str | None = None,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[float, float]:
    """Confidence-based co-training ensemble (``CoTrainingEnsemble_v2``).

    Mirrors :func:`train_model` but drives the v2 ensemble: no suspension pool size and no
    fine-tuning (models are always retrained from scratch); censored units are ranked by the
    width of a conformal prediction interval (``crepes``) at the ``confidence`` level instead
    of by the retraining "delta".
    """
    if (len(batchs_size) != 4
            or len(patiences) != 4
            or len(shuffle_dataloaders) != 4
            or len(max_epochs) != 4
            or len(lr) != 4):
        raise ValueError(
            f"bachs_size {len(batchs_size)}, patiences {len(patiences)},"
            f"shuffle_dataloaders {len(shuffle_dataloaders)}, lr {len(lr)} and"
            f"max_epochs {len(max_epochs)} must be lists of length 4, one for each model in the ensemble."
        )

    utils_cmapss.assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )

    checkpoints_path, results_path = utils_cmapss.create_and_get_checkpoints_results_path(
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
            "The co-training ensemble needs a validation set for early stopping / best-model "
            "selection, for the conformal calibration and for the ensemble weights. "
            "Set validation_rate > 0."
        )

    features_uncensored, targets_uncensored, features_censored, ids_censored = train_dataset.get_censored_split_tensors()
    features_tensor, targets_tensor = test_dataset.get_features_targets()

    # Labelled (uncensored) validation data: used for early stopping / best-checkpoint
    # selection, as the calibration set for the conformal regressors, and to compute the
    # ensemble weights (instead of the test set).
    val_features, val_targets, _, _ = valid_dataset.get_censored_split_tensors()

    print("Creating first model (CNN1D)...")

    feature_num = len(train_dataset.feature_cols)

    cnn = CNN1D(
        num_features=feature_num,
    )

    lstm = Simple_LSTM(
        feature_num=feature_num,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim_lstm,
        lstm_num_layers=lstm_num_layers_lstm,
        lstm_dropout=lstm_dropout_lstm,
        fc_layer_dim=fc_layer_dim_lstm,
        fc_dropout=fc_dropout_lstm,
    )

    transformer_features = TransformerFeatures(
        feature_num=feature_num,
        sequence_len=sequence_len,
        transformer_encoder_head_num=transformer_encoder_head_num_transformer_features,
        fc_layer_dim=fc_layer_dim_transformer_features,
        fc_dropout=fc_dropout_transformer_features,
        num_layers=num_layers_transformer_features,
    )

    transformer_time_sequence = TransformerTimeSequence(
        feature_num=feature_num,
        sequence_len=sequence_len,
        d_model=sequence_len,
        transformer_encoder_head_num=transformer_encoder_head_num_transformer_time_series,
        fc_layer_dim=fc_layer_dim_transformer_time_series,
        fc_dropout=fc_dropout_transformer_time_series,
        num_layers=num_layers_transformer_time_series,
    )

    models = [cnn, lstm, transformer_features, transformer_time_sequence]

    cotraining_ensemble = CoTrainingEnsembleV2(
        models=models,
        verbose=2,
        confidence=confidence,
    )

    models_number = len(models)

    lightning_modules = [TransformerLstmModule(lr=lr[j], model=model) for j, model in enumerate(models)]

    # Each _train_fun call builds a fresh Trainer from these factories. The ModelCheckpoint
    # lets the ensemble reload the best (val_loss) weights instead of the last-epoch ones, and
    # EarlyStopping avoids over/under-training. Checkpoints go to throwaway temp dirs that we
    # remove at the end.
    created_ckpt_dirs: list[str] = []

    def make_trainer_factory(patience: int, max_epoch: int) -> Callable[[], Trainer]:
        def factory() -> Trainer:
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
                max_epochs=max_epoch,
                accelerator="auto",
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

        return factory

    # One factory per model, each carrying that model's own epochs / patience.
    trainer_factories: list[Callable[[], Trainer]] = [
        make_trainer_factory(patiences[j], max_epochs[j]) for j in range(models_number)
    ]

    cotraining_ensemble.setup_training(
        lightning_modules=lightning_modules,
        trainer_factories=trainer_factories,
        batchs_size=batchs_size,
        shuffle_dataloaders=shuffle_dataloaders,
    )

    print(f"Training Coprog v2 model...")

    # Persistent run log next to the results. Created (truncated) here with a metadata
    # header, then handed to the ensemble which appends all its messages under it.
    log_file_path = os.path.join(results_path, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Co-Training Ensemble v2 run ===\n")
        f.write(f"Datetime: {datetime_for_folders}\n")
        f.write(f"Sub-dataset: {sub_dataset}\n")
        f.write(f"Confidence: {confidence}\n")
        f.write(f"Percent censored: {percent_of_censored_data}\n")
        f.write(f"Percent broken: {percent_of_broken_data}\n")
        f.write("===================================\n")

    try:
        cotraining_ensemble.train(
            train_with_censored_data=False,
            failure_data=features_uncensored,
            failure_label=targets_uncensored,
            suspension_data=features_censored,
            suspension_ids=ids_censored,
            iterations=coprog_iterations,
            val_data=val_features,
            val_label=val_targets,
            test_data=features_tensor,
            test_label=targets_tensor,
            criteria_callback=cmapss_score,
            weight_mode="min",
            metrics_file=os.path.join(results_path, "metrics_per_stage.csv"),
            log_file=log_file_path,
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

    # Persist the final trained models using the same {"state_dict": ...} layout that
    # _fit_and_reload_best reloads, so these files can be loaded back with
    # torch.load(path)["state_dict"].
    for j, module in enumerate(cotraining_ensemble.lightning_modules):
        final_ckpt_path = os.path.join(
            checkpoints_path, f"{model_version}-turbofan-{sub_dataset}-model_{j}.ckpt"
        )
        torch.save({"state_dict": module.state_dict()}, final_ckpt_path)
        print(f"Saved final checkpoint for model {j} to {final_ckpt_path}")

    # Flatten both sides so we compute a real element-wise RMSE. targets_tensor is (N, 1)
    # and predict() returns (N, 1); flattening keeps the comparison aligned as (N,).
    y_hat = cotraining_ensemble.predict(features_tensor).detach().cpu().view(-1)
    targets_flat = targets_tensor.detach().cpu().view(-1)

    rmse = torch.sqrt(torch.mean((targets_flat - y_hat) ** 2))
    score = utils_cmapss.cmapss_score(y_hat.numpy(), targets_flat.numpy())

    columns = ['test_rmse', 'test_score']

    for j in range(models_number):
        columns.append(f"weight_{j}")

    scores = pd.DataFrame(columns=columns)

    row = [rmse, score]

    for j in range(models_number):
        row.append(cotraining_ensemble.weights[j])

    scores.loc[0] = row

    scores.to_csv(f'{results_path}/{model_version}-turbofan-{sub_dataset}.csv', index=False)

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
    # Paths (relative to C_MAPSS/utils/). Created here so assert_data_is_valid passes.
    checkpoints_path = "../checkpoints"
    results_path = "../results"
    dataset_root = "../../data/C_MAPSS"
    model_version = "cotraining_ensemble_v2"

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # Dataset params
    seed = 42
    sub_dataset = "FD001"
    sequence_len = 32
    max_rul = 125
    return_sequence_label = False
    norm_type = "z-score"
    cluster_operations = True
    norm_by_operations = True
    include_cols = None
    exclude_cols = None
    return_id = False
    validation_rate = 0.2
    use_only_final_on_test = True
    use_max_rul_on_test = True
    use_max_rul_on_valid = True
    percent_of_broken_data = None
    percent_of_censored_data = 0.9

    # Training params. The four list entries map, in order, to:
    # [CNN1D, Simple_LSTM, TransformerFeatures, TransformerTimeSequence].
    # Small values here are for a quick end-to-end smoke test.
    coprog_iterations = 2
    confidence = 0.95
    selection_mode_str = "voting"
    max_epochs = [2, 2, 2, 2]
    patiences = [2, 2, 2, 2]
    batchs_size = [256, 256, 256, 256]
    shuffle_dataloaders = [True, True, True, True]
    lr = [0.0002, 0.0002, 0.0002, 0.0002]

    # Model params. TransformerFeatures head count must divide sequence_len (= 32);
    # TransformerTimeSequence head count must divide d_model (= sequence_len = 32).
    hidden_dim_lstm = 32
    lstm_num_layers_lstm = 3
    lstm_dropout_lstm = 0.2
    fc_layer_dim_lstm = 32
    fc_dropout_lstm = 0.2
    transformer_encoder_head_num_transformer_features = 4
    fc_layer_dim_transformer_features = 32
    fc_dropout_transformer_features = 0.2
    num_layers_transformer_features = 2
    transformer_encoder_head_num_transformer_time_series = 4
    fc_layer_dim_transformer_time_series = 32
    fc_dropout_transformer_time_series = 0.2
    num_layers_transformer_time_series = 2

    train_model_v2(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        model_version=model_version,
        # Training
        coprog_iterations=coprog_iterations,
        confidence=confidence,
        selection_mode_str=selection_mode_str,
        max_epochs=max_epochs,
        patiences=patiences,
        batchs_size=batchs_size,
        shuffle_dataloaders=shuffle_dataloaders,
        lr=lr,
        # Model params
        hidden_dim_lstm=hidden_dim_lstm,
        lstm_num_layers_lstm=lstm_num_layers_lstm,
        lstm_dropout_lstm=lstm_dropout_lstm,
        fc_layer_dim_lstm=fc_layer_dim_lstm,
        fc_dropout_lstm=fc_dropout_lstm,
        transformer_encoder_head_num_transformer_features=transformer_encoder_head_num_transformer_features,
        fc_layer_dim_transformer_features=fc_layer_dim_transformer_features,
        fc_dropout_transformer_features=fc_dropout_transformer_features,
        transformer_encoder_head_num_transformer_time_series=transformer_encoder_head_num_transformer_time_series,
        fc_layer_dim_transformer_time_series=fc_layer_dim_transformer_time_series,
        fc_dropout_transformer_time_series=fc_dropout_transformer_time_series,
        num_layers_transformer_features=num_layers_transformer_features,
        num_layers_transformer_time_series=num_layers_transformer_time_series,
        # Dataset params
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
