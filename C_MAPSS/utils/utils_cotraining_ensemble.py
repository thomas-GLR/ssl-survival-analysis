import torch
from lightning import Trainer

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning_module.TransformerLstmModule import TransformerLstmModule
from C_MAPSS.models import CNN1D
from C_MAPSS.utils import utils_cmapss
from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2, SelectionMode


def train_model(
    coprog_iterations: int,
    coprog_suspension_pool_size: int,
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

    cnn2 = CNN1D(
        num_features=feature_num,
    )

    cnn3 = CNN1D(
        num_features=feature_num,
    )

    models = [cnn, cnn2, cnn3]

    cotraining_ensemble = CoTrainingEnsemble_v2(
        models=models,
        verbose=2
    )

    models_number = len(models)

    batchs_size = [32 for _ in range(models_number)]
    shuffle_dataloaders = [False for _ in range(models_number)]

    lightning_modules = [TransformerLstmModule(lr=0.001, model=model) for model in models]

    def make_trainer() -> Trainer:
        return Trainer(max_epochs=3, accelerator="auto", logger=False, enable_progress_bar=False,
                       enable_model_summary=False)

    trainer_factories = [make_trainer] * models_number

    cotraining_ensemble.setup_training(
        lightning_modules=lightning_modules,
        trainer_factories=trainer_factories,
        batchs_size=batchs_size,
        shuffle_dataloaders=shuffle_dataloaders,
    )

    print(f"Training Coprog model...")

    cotraining_ensemble.train(
        is_fine_tuning_during_finding_best_suspension_data=False,
        is_fine_tuning_for_last_step=False,
        selection_mode=SelectionMode.VOTING,
        train_with_censored_data=False,
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        iterations=coprog_iterations,
        suspension_pool_size=coprog_suspension_pool_size
    )

    cotraining_ensemble.calculate_weights(
        x_test=features_tensor,
        target=targets_tensor,
        criteria_callback=cmapss_score,
        mode="min",
    )

    y_hat = cotraining_ensemble.predict(features_tensor)

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


if __name__ == "__main__":
    train_model(
        coprog_iterations=2,
        coprog_suspension_pool_size=5,
        dataset_root="../../data/C_MAPSS",
        seed=42,
        sub_dataset="FD001",
        sequence_len=30,
    )
