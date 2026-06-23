from utils.utils_self_supervised import train_self_supervised

C_MAPSS_DIR = "../../data/C_MAPSS"

if __name__ == '__main__':
    # Dataset parameters
    dataset_root = C_MAPSS_DIR
    sub_dataset = "FD001"
    seq_len = 30
    max_rul = 125
    percent_of_broken_data = None
    percent_of_censored_data = 0.9
    cluster_operations = True
    norm_by_operations = True
    validation_rate = 0.2
    seed = 42

    # Pretrain model parameters
    mode = "autoencoder"  # metric or autoencoder
    in_channels = 24
    lr = 0.0001
    dropout = 0.1

    train_self_supervised(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        seq_len=seq_len,
        seed=seed,
        max_rul=max_rul,
        percent_of_broken_data=percent_of_broken_data,
        percent_of_censored_data=percent_of_censored_data,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        validation_rate=validation_rate,
        mode=mode,
        in_channels=in_channels,
        lr=lr,
        dropout=dropout,
        max_epochs=1,
        max_epochs_baseline=1,
        batch_size_pretraining=256,
        batch_size_baseline=256,
    )
