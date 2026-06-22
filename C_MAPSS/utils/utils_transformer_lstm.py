import numpy as np
import pandas as pd
import os
import pytorch_lightning as pl
from torch import accelerator
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader

from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning.TransformerLstmModule import TransformerLstmModule
from C_MAPSS.models.TransformerEncoder_LSTM_1 import TransformerEncoder_LSTM_1
from C_MAPSS.models.Simple_LSTM import Simple_LSTM

# For PyTorch 2.6+
# We indicate to PyTorch that these classes are "safe" when loading checkpoints
add_safe_globals([Simple_LSTM, TransformerEncoder_LSTM_1])


def train_model(
        # Dataset params
        data_dir=None,
        sub_dataset='FD001',
        model_version='transformer',
        sequence_len=30,
        feature_num=14,
        norm_type='z-score',
        cluster_operations=True,
        norm_by_operations=True,
        use_max_rul_on_test=False,
        piecewise_rul=125,
        validation_rate=0,
        percent_of_broken_data=None,
        percent_of_censored_data=0.9,
        # Model params
        transformer_encoder_head_num=2,
        lstm_num_layers=3,
        hidden_dim=32,
        lstm_dropout=0.2,
        fc_layer_dim=32,
        fc_dropout=0.2,
        # Training
        device='cpu',
        batch_size=256,
        lr=0.001,
        patience=10,
        max_epochs: int=500,
):
    scores = pd.DataFrame(columns=['train_rmse', 'val_rmse', 'test_rmse'])
    model_kwargs = {
        'sequence_len': sequence_len,
        'feature_num': feature_num,
        'hidden_dim': hidden_dim,
        'fc_layer_dim': fc_layer_dim,
        'lstm_num_layers': lstm_num_layers,
        'transformer_encoder_head_num': transformer_encoder_head_num,
        'fc_dropout': fc_dropout,
        'lstm_dropout': lstm_dropout,
        # 'device': device,
    }
    print('Training model with the following parameters:')
    print(sequence_len)
    print(patience)
    print(model_kwargs)
    train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(
        dataset_root=data_dir,
        sub_dataset=sub_dataset,
        sequence_len=sequence_len,
        max_rul=piecewise_rul,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        use_max_rul_on_test=use_max_rul_on_test,
        validation_rate=validation_rate,
        return_id=False,
        use_only_final_on_test=True,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
    )

    train_loader = train_dataset.get_data_loader_without_censored_data(batch_size=batch_size)
    test_loader = test_dataset.get_data_loader_without_censored_data(batch_size=batch_size)
    valid_loader = valid_dataset.get_data_loader_without_censored_data(batch_size=batch_size)

    if model_version == 'transformer':
        model = TransformerEncoder_LSTM_1(**model_kwargs)
    else:
        model = Simple_LSTM(**model_kwargs)

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
        dirpath=f'./checkpoints/model-{model_version}-turbofan',
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        default_root_dir='./checkpoints',
        accelerator=device,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        # checkpoint_callback=False,
        # logger=False,
        # progress_bar_refresh_rate=0
    )
    trainer.fit(transformer_lstm_module, train_loader, val_dataloaders=valid_loader or test_loader)
    t = trainer.callback_metrics
    train_rmse = t['train_rmse']
    val_rmse = t['val_rmse']
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    t = trainer.callback_metrics
    test_rmse = t['test_rmse']
    # Add the results to the dataframe
    scores.loc[0] = [train_rmse, val_rmse, test_rmse]
    # Save the results

    if not os.path.exists("./results"):
        os.makedirs("./results")

    scores.to_csv(f'./results/{model_version}-turbofan.csv', index=False)

    # Save model predictions
    transformer_lstm_module = TransformerLstmModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    mode = transformer_lstm_module.to(device)
    transformer_lstm_module.eval()

    generate_and_save_model_prediction(
        loader=test_loader,
        device=device,
        module=transformer_lstm_module,
        model_version=model_version,
        prediction_type='test',
    )

    generate_and_save_model_prediction(
        loader=train_loader,
        device=device,
        module=transformer_lstm_module,
        model_version=model_version,
        prediction_type='train',
    )


def generate_and_save_model_prediction(
        loader: DataLoader,
        device: str,
        module: pl.LightningModule,
        model_version: str,
        prediction_type: str # test or train
):
    predictions = []
    targets = []
    for x, y in loader:
        x = x.to(device)
        y_hat = module(x)
        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions.tofile(f'./results/{model_version}_{prediction_type}_predictions.csv', sep=',')
    targets.tofile(f'./results/{model_version}_{prediction_type}_targets.csv', sep=',')
