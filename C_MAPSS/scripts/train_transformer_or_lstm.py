from utils.utils_transformer_lstm import train_model, benchmark_for_transformer_or_lstm

C_MAPSS_DIR = "../../data/C_MAPSS"


if __name__ == '__main__':
    model_version = "transformer"

    dataset_root = C_MAPSS_DIR
    checkpoints_path = "../checkpoints"
    results_path = "../results"
    config_path = "../config"
    sub_dataset = "FD001"
    seed = 42
    sequence_len = 30
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
    use_max_rul_on_test = False
    use_max_rul_on_valid = True
    percent_of_broken_data = None
    percent_of_censored_data = 0.

    transformer_encoder_head_num = 10
    lstm_num_layers = 1
    hidden_dim = 128
    lstm_dropout = 0.3
    fc_layer_dim = 32
    fc_dropout = 0.4

    device = None,
    batch_size = 128,
    lr = 0.0002,
    patience = 50,
    max_epochs = 1,

    benchmark_for_transformer_or_lstm(
        config_path=config_path,
        checkpoints_path=checkpoints_path,
        results_path=results_path,
        dataset_root=dataset_root,
        model_version=model_version,
        device="cpu",
    )

    # train_model(
    #     checkpoints_path=checkpoints_path,
    #     results_path=results_path,
    #     model_version=model_version,
    #     # Dataset params
    #     dataset_root = dataset_root,
    #     sub_dataset = sub_dataset,
    #     sequence_len = sequence_len,
    #     seed=seed,
    #     max_rul = max_rul,
    #     return_sequence_label = return_sequence_label,
    #     norm_type = norm_type,
    #     cluster_operations = cluster_operations,
    #     norm_by_operations = norm_by_operations,
    #     include_cols = include_cols,
    #     exclude_cols = exclude_cols,
    #     return_id = return_id,
    #     validation_rate = validation_rate,
    #     use_only_final_on_test = use_only_final_on_test,
    #     use_max_rul_on_test = use_max_rul_on_test,
    #     use_max_rul_on_valid = use_max_rul_on_valid,
    #     percent_of_broken_data = percent_of_broken_data,
    #     percent_of_censored_data = percent_of_censored_data,
    #     # Model params
    #     transformer_encoder_head_num = transformer_encoder_head_num,
    #     lstm_num_layers = lstm_num_layers,
    #     hidden_dim = hidden_dim,
    #     lstm_dropout = lstm_dropout,
    #     fc_layer_dim = fc_layer_dim,
    #     fc_dropout = fc_dropout,
    #     # Training
    #     device = device,
    #     batch_size=batch_size,
    #     lr=lr,
    #     patience=patience,
    #     max_epochs=max_epochs,
    # )