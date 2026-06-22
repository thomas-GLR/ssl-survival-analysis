from C_MAPSS.models import Simple_LSTM, CNN1D
from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader
from models import Coprog
import torch

from utils.utils import score

C_MAPSS_DIR = "../../data/C_MAPSS"
CHECKPOINTS_DIR = "../checkpoints"

if __name__ == "__main__":
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
    parser.add_argument('--validation-rate', type=float, default=0., help='validation set ratio of train set')
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--suspension-pool-size', type=int, default=5)

    args = parser.parse_args()

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
    }

    print("Creating dataset with parameters :")
    print(dataset_kwargs)

    train_dataset, test_dataset, _ = CMAPSSLoader.get_datasets(**dataset_kwargs)

    lstm_kwargs = {
        'feature_num': args.feature_num,
        'sequence_len': args.sequence_len,
        'hidden_dim': args.hidden_dim,
        'lstm_num_layers': args.lstm_num_layers,
        'lstm_dropout': args.lstm_dropout,
        'fc_layer_dim': args.fc_layer_dim,
        'fc_dropout': args.fc_dropout
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
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        lr=args.lr
    )

    features_uncensored, targets_uncensored, features_censored = train_dataset.get_censored_split_tensors()
    features_tensor, targets_tensor = test_dataset.get_features_targets()

    print(f"Training Coprog model...")

    coprog.train(
        failure_data=features_uncensored,
        failure_label=targets_uncensored,
        suspension_data=features_censored,
        iterations=args.iterations,
        suspension_pool_size=args.suspension_pool_size
    )

    print("Saving first and second trained models...")

    torch.save(coprog.first_model, f"{CHECKPOINTS_DIR}/coprog_{args.sub_dataset}_cnn.pth")
    torch.save(coprog.second_model, f"{CHECKPOINTS_DIR}/coprog_{args.sub_dataset}_lstm.pth")

    y_hat = coprog.predict(features_tensor)

    print(y_hat)
    print(targets_tensor.flatten())

    rmse = torch.sqrt(torch.mean((targets_tensor - y_hat) ** 2))

    print(f"Test RMSE: {rmse}")
    print(f"Score: {score(y_hat, targets_tensor)}")
