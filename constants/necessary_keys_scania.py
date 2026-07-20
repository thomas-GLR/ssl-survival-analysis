# =======================================================
# DATASET
# =======================================================

NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS = [
    "sequence_len",
    "seed",
    "val_rate",
    "test_rate",
    "stratify",
    "norm_type",
    "num_workers",
    "pin_memory",
    "return_sequence_label",
    "batch_size",
    "shuffle_loader",
    "counter_mode",
    "include_histograms",
]

NECESSARY_DATASET_SELF_SUPERVISED_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

NECESSARY_DATASET_RSF_KEYS = [
    "seed",
    "val_rate",
    "test_rate",
    "stratify",
    "norm_type",
    "return_sequence_label",
    "counter_mode",
    "include_histograms",
]

NECESSARY_DATASET_COPROG_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

NECESSARY_DATASET_PYCLUS_KEYS = NECESSARY_DATASET_RSF_KEYS

NECESSARY_DATASET_TRANSFORMER_FEATURES_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
NECESSARY_DATASET_TRANSFORMER_TIME_SEQUENCE_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

# =======================================================
# MODEL
# =======================================================

NECESSARY_TRANSFORMER_KEYS = [
    "transformer_encoder_head_num",
    "hidden_dim",
    "lstm_num_layers",
    "lstm_dropout",
    "fc_layer_dim",
    "fc_dropout",
]

NECESSARY_LSTM_KEYS = [
    "hidden_dim",
    "lstm_num_layers",
    "lstm_dropout",
    "fc_layer_dim",
    "fc_dropout",
]

NECESSARY_SELF_SUPERVISED_KEYS = [
    "sequence_len",
    "pretraining_lr",
    "dropout",
    "num_layers",
    "kernel_size",
    "base_filters",
    "latent_dim",
    "weight_decay",
    "max_epochs",
    "patience",
    "batch_size_pretraining",
    "latent_dim_baseline",
    "lr_baseline",
    "max_epochs_baseline",
    "batch_size_baseline",
    "min_distance",
    "shuffle_loader",
]

NECESSARY_RSF_KEYS = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "cv_for_grid_search",
    "variance_warning_threshold",
]

NECESSARY_COPROG_KEYS = [
    "first_model",
    "second_model"
]

NECESSARY_PYCLUS_KEYS = [
    "n_trees",
    "max_depth",
    "min_leaf_size",
    "cv_for_grid_search",
    "pruning_method",
    "variance_warning_threshold",
    "shuffle_loader",
]

NECESSARY_CNN_KEYS = []

NECESSARY_TRANSFORMER_FEATURES_KEYS = [
    "transformer_encoder_head_num",
    "transformer_num_layer",
    "fc_layer_dim",
    "fc_dropout",
]

NECESSARY_TRANSFORMER_TIME_SEQUENCE_KEYS = [
    "transformer_encoder_head_num",
    "transformer_num_layer",
    "fc_layer_dim",
    "fc_dropout",
]

# =======================================================
# TRAINING
# =======================================================

NECESSARY_TRAINING_CNN_KEYS = [
    "lr",
    "patience",
    "max_epochs",
    "rul_target_standardization",
]

NECESSARY_TRAINING_LSTM_KEYS = NECESSARY_TRAINING_CNN_KEYS
NECESSARY_TRAINING_TRANSFORMER_KEYS = NECESSARY_TRAINING_CNN_KEYS
NECESSARY_TRAINING_TRANSFORMER_FEATURES_KEYS = NECESSARY_TRAINING_CNN_KEYS
NECESSARY_TRAINING_TRANSFORMER_TIME_SEQUENCE_KEYS = NECESSARY_TRAINING_CNN_KEYS

NECESSARY_TRAINING_RSF_KEYS = []

NECESSARY_TRAINING_COPROG_KEYS = [
    "lr",
    "patiences",
    "max_epochs",
    "coprog_iterations",
    "coprog_suspension_pool_size",
    "rul_target_standardization",
]
