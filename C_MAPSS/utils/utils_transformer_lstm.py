import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning.TransformerLstmModule import TransformerLstmModule
from C_MAPSS.models.Simple_LSTM import Simple_LSTM
from C_MAPSS.models.TransformerEncoder_LSTM_1 import TransformerEncoder_LSTM_1
from C_MAPSS.utils import utils_cmapss

# For PyTorch 2.6+
# We indicate to PyTorch that these classes are "safe" when loading checkpoints
add_safe_globals([Simple_LSTM, TransformerEncoder_LSTM_1])


def train_model(
        checkpoints_path: str,
        results_path: str,
        model_version: str,
        device: str | None,
        # Dataset params
        dataset_root: str,
        seed: int | None,
        sub_dataset: str,
        sequence_len: int,
        max_rul: int,
        return_sequence_label: bool,
        norm_type: str,
        cluster_operations: bool,
        norm_by_operations: bool,
        include_cols: list[str] | None,
        exclude_cols: list[str] | None,
        return_id: bool,
        validation_rate,
        use_only_final_on_test: bool,
        use_max_rul_on_test: bool,
        use_max_rul_on_valid: bool,
        percent_of_broken_data: float | None,
        percent_of_censored_data: float,
        # Model params
        lstm_num_layers: int,
        hidden_dim: int,
        lstm_dropout: float,
        fc_layer_dim: int,
        fc_dropout: float,
        # Training
        batch_size: int,
        lr: float,
        patience: int,
        max_epochs: int,
        transformer_encoder_head_num: int | None = None,
        datetime_for_folders: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
) -> tuple[float, float]:
    utils_cmapss.assert_data_is_valid(
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
    )

    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

    scores = pd.DataFrame(columns=['train_rmse', 'val_rmse', 'test_rmse', 'test_score'])

    broken_percentage = 0. if percent_of_broken_data is None else percent_of_broken_data

    folder_for_current_training = f"/model-{model_version}-turbofan-{sub_dataset}-{datetime_for_folders}/censored-{percent_of_censored_data:.2f}-broken-{broken_percentage:.2f}"

    checkpoints_path = f'{checkpoints_path}/{folder_for_current_training}'
    results_path = f'{results_path}/{folder_for_current_training}'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    dataset_kwargs = {
        'dataset_root': dataset_root,
        'sub_dataset': sub_dataset,
        'sequence_len': sequence_len,
        'max_rul': max_rul,
        'return_sequence_label': return_sequence_label,
        'norm_type': norm_type,
        'cluster_operations': cluster_operations,
        'norm_by_operations': norm_by_operations,
        'include_cols': include_cols,
        'exclude_cols': exclude_cols,
        'return_id': return_id,
        'validation_rate': validation_rate,
        'use_only_final_on_test': use_only_final_on_test,
        'use_max_rul_on_test': use_max_rul_on_test,
        'use_max_rul_on_valid': use_max_rul_on_valid,
        'percent_of_broken_data': percent_of_broken_data,
        'percent_of_censored_data': percent_of_censored_data,
    }

    print("Creating data loader with the following parameters :")
    print(dataset_kwargs)

    train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        sequence_len=sequence_len,
        seed=seed,
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

    feature_num = len(train_dataset.feature_cols)

    # As the models can only handle supervised learning we need to filtered censored data
    train_loader = train_dataset.get_data_loader_without_censored_data(batch_size=batch_size)
    test_loader = test_dataset.get_data_loader_without_censored_data(batch_size=batch_size)
    valid_loader = valid_dataset.get_data_loader_without_censored_data(
        batch_size=batch_size) if valid_dataset is not None else None

    model_kwargs = {
        'sequence_len': sequence_len,
        'feature_num': feature_num,
        'hidden_dim': hidden_dim,
        'fc_layer_dim': fc_layer_dim,
        'lstm_num_layers': lstm_num_layers,
        'transformer_encoder_head_num': transformer_encoder_head_num,
        'fc_dropout': fc_dropout,
        'lstm_dropout': lstm_dropout,
    }
    print('Training model with the following parameters:')
    print(f"Sequence length : {sequence_len}")
    print(f"Patience : {patience}")
    print(f"Models parameters : {model_kwargs}")

    if model_version == 'transformer':
        model = TransformerEncoder_LSTM_1(**model_kwargs)
    elif model_version == 'lstm':
        model = Simple_LSTM(**model_kwargs)
    else:
        raise ValueError(f"Model version {model_version} is not supported")

    transformer_lstm_module = TransformerLstmModule(
        lr=lr,
        model=model
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        default_root_dir=checkpoints_path,
        accelerator=device,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(transformer_lstm_module, train_loader, val_dataloaders=valid_loader or test_loader)

    callbacks_metrics = trainer.callback_metrics
    train_rmse = callbacks_metrics['train_rmse']
    val_rmse = callbacks_metrics['val_rmse']

    trainer.test(dataloaders=test_loader, ckpt_path='best')

    callbacks_metrics = trainer.callback_metrics

    test_rmse = callbacks_metrics['test_rmse']
    test_score = callbacks_metrics['test_score']

    # Add the results to the dataframe
    scores.loc[0] = [train_rmse, val_rmse, test_rmse, test_score]

    # Save the results
    scores.to_csv(f'{results_path}/{model_version}-turbofan-{sub_dataset}.csv', index=False)

    print(f"Scores from train and test :\n{scores}")

    # Save model predictions
    if model_version == 'transformer':
        model_for_reload = TransformerEncoder_LSTM_1(**model_kwargs)
    elif model_version == 'lstm':
        model_for_reload = Simple_LSTM(**model_kwargs)
    else:
        raise ValueError(f"Model reload version {model_version} is not supported")

    transformer_lstm_module_with_trained_model = TransformerLstmModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model=model_for_reload,
    )

    transformer_lstm_module_with_trained_model = transformer_lstm_module_with_trained_model.to(device)
    transformer_lstm_module_with_trained_model.eval()

    return _generate_and_save_model_prediction(
        loader=test_loader,
        device=device,
        module=transformer_lstm_module_with_trained_model,
        model_version=model_version,
        prediction_type='test',
        results_path=results_path,
        sub_dataset=sub_dataset,
    )


def _generate_and_save_model_prediction(
        loader: DataLoader,
        device: str,
        module: pl.LightningModule,
        model_version: str,
        prediction_type: str,
        results_path: str,
        sub_dataset: str,
) -> tuple[float, float]:
    """
    Generate results for predictions from the data_loader and save them in csv file

    :param loader: the loader to use for predictions.
    :param device: the device to use for predictions.
    :param module: the module containing the model.
    :param model_version: the model version.
    :param prediction_type: the prediction type.
    :param results_path: the path from which to save the results.
    :param sub_dataset: the sub dataset name to use in the results file name.
    """
    predictions = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y_hat = module(x)
        predictions.extend(y_hat.cpu().detach().numpy().flatten())
        targets.extend(y.cpu().detach().numpy().flatten())

    df = pd.DataFrame({
        'targets': targets,
        'predictions': predictions
    })

    csv_path = f'{results_path}/{model_version}_{prediction_type}_results_{sub_dataset}.csv'

    df.to_csv(csv_path, index=False)

    print(f"Results for {prediction_type} are saved under : {csv_path}")

    predictions_tensor = torch.Tensor(predictions)
    targets_tensor = torch.Tensor(targets)

    rmse = torch.sqrt(torch.mean((targets_tensor - predictions_tensor) ** 2))
    score = utils_cmapss.cmapss_score(np.array(predictions), np.array(targets))

    return rmse.item(), score
