import json
import os
from typing import Callable

import torch
from lightning import LightningModule

from models.coprog_gpu_pool import TrainingSpec, run_training_job

INITIAL_MODELS_MANIFEST = "manifest.json"


def get_or_train_initial_models(
        initial_models_dir: str,
        module_builders: list[Callable[[], LightningModule]],
        batch_size: list[int],
        number_of_models: int,
        max_epochs: list[int],
        patiences: list[int],
        shuffle_dataloaders: list[bool],
        version_strs: list[str],
        failure_data: torch.Tensor,
        failure_label: torch.Tensor,
        val_data: torch.Tensor | None,
        val_label: torch.Tensor | None,
        seed: int | None,
        run_log_path: str | None = None,
) -> list[LightningModule]:
    """Retrieve pretrained initial models from cache directory or train them if missing.

    :param initial_models_dir: Directory where pretrained models and manifest are stored.
    :param module_builders: Callables that return a fresh LightningModule.
    :param batch_size: Batch size for each model.
    :param number_of_models: Number of initial models.
    :param max_epochs: Maximum epochs for each model.
    :param patiences: Early stopping patience for each model.
    :param shuffle_dataloaders: Whether to shuffle dataloaders for each model.
    :param version_strs: String identifiers for the models.
    :param failure_data: Training features (uncensored / failure rows).
    :param failure_label: Training targets (uncensored / failure rows).
    :param val_data: Validation features.
    :param val_label: Validation targets.
    :param seed: Random seed.
    :param run_log_path: Optional path to log file.
    :return: List of trained/loaded LightningModule instances.
    """
    if run_log_path is not None:
        assert os.path.exists(run_log_path), f"{run_log_path} does not exist"

    def log(message: str) -> None:
        """Print a run-level milestone and append it to the run log."""
        print(message)
        if run_log_path is not None:
            with open(run_log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    initial_models: list[LightningModule | None] = [None] * number_of_models

    index_model_per_model_missing: dict[str, int] = {}

    manifest_path = os.path.join(initial_models_dir, INITIAL_MODELS_MANIFEST)
    if os.path.exists(manifest_path):
        log(f"Loading the shared initial models from {initial_models_dir}...")

        for model_name in version_strs:
            model_file_name = os.path.join(initial_models_dir, f"initial_model_{model_name}.pth")
            if os.path.exists(model_file_name):
                log(f"Loading initial model {model_name} from {model_file_name} at index {version_strs.index(model_name)}")
                initial_models[version_strs.index(model_name)] = torch.load(model_file_name, weights_only=False)

        for model_name in version_strs:
            model_file_name = os.path.join(initial_models_dir, f"initial_model_{model_name}.pth")
            if not os.path.exists(model_file_name):
                log(f"\t{model_name} at index {version_strs.index(model_name)} is missing")
                index_model_per_model_missing[model_name] = version_strs.index(model_name)

        if len(index_model_per_model_missing) == 0:
            return initial_models

        log(f"Missing {len(index_model_per_model_missing)} models need to train them...")

    else:
        for model_name in version_strs:
            index_model_per_model_missing[model_name] = version_strs.index(model_name)

    for model_index in index_model_per_model_missing.values():
        x_i, y_i = failure_data, failure_label

        log(f"Initial training of model {model_index} named {version_strs[model_index]} on {len(x_i)} failure samples...")

        initial_state_dicts = []
        for builder in module_builders:
            template = builder()
            initial_state_dicts.append(
                {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
            )

        spec = TrainingSpec(
            module_builder=module_builders[model_index],
            initial_state_dict=initial_state_dicts[model_index],
            max_epochs=max_epochs[model_index],
            patience=patiences[model_index],
            batch_size=batch_size[model_index],
            shuffle=shuffle_dataloaders[model_index],
            train_x=x_i.detach().cpu(),
            train_y=y_i.detach().cpu(),
            val_x=val_data.detach().cpu() if val_data is not None else None,
            val_y=val_label.detach().cpu() if val_label is not None else None,
            return_state=True,
            accelerator="auto",
            devices=None,
            is_censored=None,
            lower_bound=None,
            use_cotraining_survival_loss=False,
        )

        result = run_training_job(spec)

        module = module_builders[model_index]()
        module.load_state_dict(result["state_dict"])
        initial_models[model_index] = module

    os.makedirs(initial_models_dir, exist_ok=True)
    model_files = []
    for i, (module, version_str) in enumerate(zip(initial_models, version_strs)):
        name = f"initial_model_{version_str}.pth"
        torch.save(module, os.path.join(initial_models_dir, name))
        model_files.append(name)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "number_of_models": number_of_models,
                "version_strs": version_strs,
                "model_files": model_files,
                "seed": seed,
                "max_epochs": max_epochs,
                "patiences": patiences,
                "shuffle_dataloaders": shuffle_dataloaders,
                "batch_size": batch_size,
            },
            f,
            indent=2,
        )
    log(f"Shared initial models saved to {initial_models_dir}")

    return initial_models
