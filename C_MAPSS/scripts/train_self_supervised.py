from C_MAPSS.utils.utils_self_supervised import train_self_supervised

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
    norm_type = "z-score"
    include_cols = []
    exclude_cols = []
    return_id = False
    use_only_final_on_test = True
    use_max_rul_on_test = False
    use_max_rul_on_valid = True

    # Pretrain model parameters
    model_name = "autoencoder"  # metric or autoencoder
    in_channels = 24
    lr = 0.0001
    dropout = 0.1
    batch_size = 256
    max_epochs = 1

    checkpoints_path = "../checkpoints"

    # train_self_supervised(
    #     dataset_root=dataset_root,
    #     sub_dataset=sub_dataset,
    #     sequence_len=seq_len,
    #     seed=seed,
    #     max_rul=max_rul,
    #     percent_of_broken_data=percent_of_broken_data,
    #     percent_of_censored_data=percent_of_censored_data,
    #     cluster_operations=cluster_operations,
    #     norm_by_operations=norm_by_operations,
    #     validation_rate=validation_rate,
    #     model_name=model_name,
    #     in_channels=in_channels,
    #     pretraining_lr=lr,
    #     dropout=dropout,
    #     max_epochs=1,
    #     max_epochs_baseline=1,
    #     batch_size_pretraining=256,
    #     batch_size_baseline=256,
    # )

    rmse, score = train_self_supervised(
        checkpoints_path=checkpoints_path,
        model_version=model_name,
        #=#,
        dataset_root=dataset_root,
        seed=seed,
        sub_dataset=sub_dataset,
        sequence_len=seq_len,
        max_rul=max_rul,
        return_sequence_label=False,
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
        #=#,
        in_channels=in_channels,
        pretraining_lr=lr,
        dropout=dropout,
        max_epochs=max_epochs,
        patience=50,
        batch_size_pretraining=batch_size,
        #=#,
        lr_baseline=lr,
        max_epochs_baseline=max_epochs,
        batch_size_baseline=batch_size,
    )

    print(f"RMSE : {rmse}")
    print(f"Score : {score}")
