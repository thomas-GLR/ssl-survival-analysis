from utils.utils_transformer_lstm import train_model

C_MAPSS_DIR = "../../data/C_MAPSS"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Turbofan Example')

    parser.add_argument('--sequence-len', type=int, default=60)
    parser.add_argument('--feature-num', type=int, default=24)
    parser.add_argument('--hidden-dim', type=int, default=100, help='LSTM hidden dims')
    parser.add_argument('--fc-layer-dim', type=int, default=100)
    parser.add_argument('--rnn-num-layers', type=int, default=5)
    parser.add_argument('--lstm-dropout', type=float, default=0.2)
    parser.add_argument('--feature-head-num', type=int, default=6)
    parser.add_argument('--fc-dropout', type=float, default=0.2)
    parser.add_argument('--dataset-root', type=str, default=C_MAPSS_DIR, help='The dir of CMAPSS dataset')
    parser.add_argument('--sub-dataset', type=str, default='FD001', help='FD001/2/3/4')
    parser.add_argument('--norm-type', type=str, default='z-score', help='z-score, -1-1 or 0-1')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--cluster-operations', action='store_true', default=True)
    parser.add_argument('--norm-by-operations', action='store_true', default=True)
    parser.add_argument('--use-max-rul-on-test', action='store_true', default=False)
    parser.add_argument('--validation-rate', type=float, default=0.2, help='validation set ratio of train set')
    parser.add_argument('--broken-rate', type=float, default=None)
    parser.add_argument('--censored-rate', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model-version', type=str, default='lstm',
                        help='model version to train values: transformer, lstm')

    args = parser.parse_args()

    train_model(
        # Dataset params
        data_dir=args.dataset_root,
        sub_dataset=args.sub_dataset,
        model_version=args.model_version,
        sequence_len=args.sequence_len,
        feature_num=args.feature_num,
        norm_type=args.norm_type,
        cluster_operations=args.cluster_operations,
        norm_by_operations=args.norm_by_operations,
        use_max_rul_on_test=args.use_max_rul_on_test,
        piecewise_rul=args.max_rul,
        validation_rate=args.validation_rate,
        percent_of_broken_data=args.broken_rate,
        percent_of_censored_data=args.censored_rate,
        # Model params
        transformer_encoder_head_num=args.feature_head_num,
        lstm_num_layers=args.rnn_num_layers,
        hidden_dim=args.hidden_dim,
        lstm_dropout=args.lstm_dropout,
        fc_layer_dim=args.fc_layer_dim,
        fc_dropout=args.fc_dropout,
        # Training
        device='cpu',
        batch_size=256,#args.batch_size,
        lr=args.lr,
        patience=args.patience,
        max_epochs=args.max_epochs,
    )