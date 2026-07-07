from typing import Callable

from lightning import Trainer
from lightning.pytorch import callbacks

from models import Simple_LSTM, CNN1D
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from C_MAPSS.lightning_module.TransformerLstmModule import TransformerLstmModule
from models import Coprog
import torch

from C_MAPSS.utils import utils_cmapss
from C_MAPSS.utils.utils_coprog import cmapss_score as cmapss_score_torch

C_MAPSS_DIR = "../../data/C_MAPSS"
CHECKPOINTS_DIR = "../checkpoints"

if __name__ == "__main__":
    dataset_root = C_MAPSS_DIR
    sub_dataset = "FD001"
    sequence_len = 30
    max_rul = 125
    broken_data = None
    censored_data = 0.9
    norm_type = "z-score"
    cluster_operations = True
    norm_by_operations = True
    validation_rate = 0.2

    feature_num = 24
    hidden_dim = 3
    lstm_num_layers = 3
    lstm_dropout = 0.2
    fc_layer_dim = 32
    fc_dropout = 0.2

    batch_size = 256
    epochs = 100
    verbose = 1
    lr = 1e-3

    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Turbofan Example')

    # Dataset parameters
    parser.add_argument('--dataset-root', type=str, default=C_MAPSS_DIR, help='The dir of CMAPSS dataset')
    parser.add_argument('--sub-dataset', type=str, default="FD001", help='FD001/2/3/4')
    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--broken-data', type=float, default=None, help='The percentage of broken data')
    parser.add_argument('--censored-data', type=float, default=0.9, help='The percentage of censored data')
    parser.add_argument('--norm-type', type=str, default="z-score", help='z-score, -1-1 or 0-1')
    parser.add_argument('--cluster-operations', action='store_true', default=True)
    parser.add_argument('--norm-by-operations', action='store_true', default=True)
    parser.add_argument('--validation-rate', type=float, default=0.2, help='validation set ratio of train set')
    parser.add_argument('--use-max-rul-on-test', action='store_true', default=True)

    # Models parameters
    parser.add_argument('--feature-num', type=int, default=24)
    parser.add_argument('--hidden-dim', type=int, default=32, help='LSTM hidden dims')
    parser.add_argument('--lstm-num-layers', type=int, default=3)
    parser.add_argument('--lstm-dropout', type=float, default=0.2)
    parser.add_argument('--fc-layer-dim', type=int, default=32)
    parser.add_argument('--fc-dropout', type=float, default=0.2)

    # Coprog parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--suspension-pool-size', type=int, default=1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_kwargs = {
        'dataset_root': args.dataset_root,
        'sub_dataset': args.sub_dataset,
        'sequence_len': args.sequence_len,
        'max_rul': args.max_rul,
        'percent_of_broken_data': args.broken_data,
        'percent_of_censored_data': args.censored_data,
        'norm_type': args.norm_type,
        'cluster_operations': args.cluster_operations,
        'norm_by_operations': args.norm_by_operations,
        'validation_rate': args.validation_rate,
        'seed': 42,
    }

    print("Creating dataset with parameters :")
    print(dataset_kwargs)

    train_dataset, test_dataset, valid_dataset = CMAPSSLoader.get_datasets(**dataset_kwargs)

    if valid_dataset is None:
        raise ValueError("A validation set is required (set --validation-rate > 0).")

    lstm_kwargs = {
        'feature_num': args.feature_num,
        'sequence_len': args.sequence_len,
        'hidden_dim': args.hidden_dim,
        'lstm_num_layers': args.lstm_num_layers,
        'lstm_dropout': args.lstm_dropout,
        'fc_layer_dim': args.fc_layer_dim,
        'fc_dropout': args.fc_dropout,
    }

    print("Creating LSTM model with parameters :")
    print(lstm_kwargs)

    lstm = Simple_LSTM(**lstm_kwargs)

    print(f"Creating CNN model with parameters :\nnum_features: {args.feature_num}")

    cnn = CNN1D(
        num_features=args.feature_num,
    )

    coprog = Coprog(
        first_model=cnn,
        second_model=lstm,
        verbose=args.verbose,
    )

    cnn_module = TransformerLstmModule(lr=args.lr, model=cnn)
    lstm_module = TransformerLstmModule(lr=args.lr, model=lstm)

    def make_trainer_factory(max_epochs: int) -> Callable[[], Trainer]:
        def factory() -> Trainer:
            early_stop_callback = callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=False, mode='min',
            )
            checkpoint_callback = callbacks.ModelCheckpoint(
                monitor='val_loss', filename='best-{epoch:02d}-{val_loss:.4f}', save_top_k=1, mode='min',
            )
            return Trainer(
                accelerator=device,
                max_epochs=max_epochs,
                callbacks=[early_stop_callback, checkpoint_callback],
                enable_progress_bar=False,
                enable_model_summary=False,
            )

        return factory

    coprog.setup_training(
        lightning_modules=[cnn_module, lstm_module],
        trainer_factories=[make_trainer_factory(args.epochs), make_trainer_factory(args.epochs)],
        batch_sizes=[args.batch_size, args.batch_size],
        shuffle_dataloaders=[True, True],
    )

    features_uncensored, targets_uncensored, features_censored, ids_censored = train_dataset.get_censored_split_tensors()
    features_tensor, targets_tensor = test_dataset.get_features_targets()
    val_features, val_targets, _, _ = valid_dataset.get_censored_split_tensors()

    print(f"Training Coprog model...")

    coprog.train(
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        suspension_ids=ids_censored,
        iterations=args.iterations,
        suspension_pool_size=args.suspension_pool_size,
        val_data=val_features,
        val_label=val_targets,
    )

    coprog.calculate_weights(
        x_test=val_features,
        target=val_targets,
        criteria_callback=cmapss_score_torch,
        mode="min",
    )

    print("Saving first and second trained models...")

    torch.save(coprog._h1, f"{CHECKPOINTS_DIR}/coprog_{args.sub_dataset}_cnn.pth")
    torch.save(coprog._h2, f"{CHECKPOINTS_DIR}/coprog_{args.sub_dataset}_lstm.pth")

    y_hat = coprog.predict(features_tensor).detach().cpu().view(-1)
    targets_flat = targets_tensor.detach().cpu().view(-1)

    print(y_hat)
    print(targets_flat)

    rmse = torch.sqrt(torch.mean((targets_flat - y_hat) ** 2))

    print(f"Test RMSE: {rmse}")
    print(f"Score: {utils_cmapss.cmapss_score(y_hat.numpy(), targets_flat.numpy())}")