from C_MAPSS.dataset.CMAPSSLoader import CMAPSSLoader

C_MAPSS_DIR = "data/C_MAPSS"

if __name__ == "__main__":
    train_dataset, test_dataset, _ = CMAPSSLoader.get_datasets(
        dataset_root=C_MAPSS_DIR,
        sub_dataset="FD001",
        sequence_len=30,
        max_rul=125,
        percent_of_broken_data=None,
        percent_of_censored_data=0.9,
        norm_type="z-score",
        cluster_operations=True,
        norm_by_operations=True,
        validation_rate=0
    )

    features_tensor, targets_tensor = test_dataset.get_features_targets()
    print(targets_tensor)

    # first_model = TransformerEncoder_LSTM_1(
    #     feature_num=24,
    #     sequence_len=30,
    #     transformer_encoder_head_num=2,
    #     hidden_dim=32,
    #     lstm_num_layers=3,
    #     lstm_dropout=0.2,
    #     fc_layer_dim=32,
    #     fc_dropout=0.2
    # )
    # second_model = TransformerEncoder_LSTM_1(
    #     feature_num=24,
    #     sequence_len=30,
    #     transformer_encoder_head_num=2,
    #     hidden_dim=32,
    #     lstm_num_layers=3,
    #     lstm_dropout=0.2,
    #     fc_layer_dim=32,
    #     fc_dropout=0.2
    # )

    # first_model = Simple_LSTM(
    #     feature_num=24,
    #     sequence_len=30,
    #     hidden_dim=32,
    #     lstm_num_layers=3,
    #     lstm_dropout=0.2,
    #     fc_layer_dim=32,
    #     fc_dropout=0.2
    # )
    #
    # second_model = Simple_LSTM(
    #     feature_num=24,
    #     sequence_len=30,
    #     hidden_dim=32,
    #     lstm_num_layers=3,
    #     lstm_dropout=0.2,
    #     fc_layer_dim=32,
    #     fc_dropout=0.2
    # )
    #
    # coprog = Coprog(
    #     first_model=first_model,
    #     second_model=second_model,
    #     batch_size=128,
    #     epochs=10,
    #     verbose=2
    # )
    #
    # features_uncensored, targets_uncensored, features_censored = train_dataset.get_censored_split_tensors()
    # features_tensor, targets_tensor = test_dataset.get_features_targets()
    #
    # coprog.train(
    #     failure_data=features_uncensored,
    #     failure_label=targets_uncensored,
    #     suspension_data=features_censored,
    #     iterations=10,
    #     suspension_pool_size=5#int(len(features_censored) * 0.5)
    # )
    #
    # y_hat = coprog.predict(features_tensor)
    #
    # print(y_hat)
    # print(targets_tensor)
    #
    # torch.save(coprog.first_model, "coprog_first_model.pth")
    # torch.save(coprog.second_model, "coprog_second_model.pth")
