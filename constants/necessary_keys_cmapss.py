# =======================================================
# DATASET
# =======================================================
from constants.necessary_shared_keys import SEED, VALIDATION_RATE, SEQUENCE_LEN, BATCH_SIZE

NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS = [
    SEED,
    "max_rul",
    "return_sequence_label",
    "norm_type",
    "cluster_operations",
    "norm_by_operations",
    "include_cols",
    "exclude_cols",
    "return_id",
    "validation_rate",
    "use_only_final_on_test",
    "use_max_rul_on_test",
    "use_max_rul_on_valid",
]

NECESSARY_DATASET_SELF_SUPERVISED_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

NECESSARY_DATASET_RSF_KEYS = [
    SEED,
    "max_rul",
    "norm_type",
    "cluster_operations",
    "norm_by_operations",
    "include_cols",
    "exclude_cols",
    "use_max_rul_on_test",
    "use_max_rul_on_valid",
    "summarize_features",
]

NECESSARY_DATASET_COPROG_KEYS = [
    SEQUENCE_LEN,
    BATCH_SIZE,
    SEED,
    "max_rul",
    "return_sequence_label",
    "norm_type",
    "cluster_operations",
    "norm_by_operations",
    "include_cols",
    "exclude_cols",
    "return_id",
    VALIDATION_RATE,
    "use_only_final_on_test",
    "use_max_rul_on_test",
    "use_max_rul_on_valid",
]

NECESSARY_DATASET_PYCLUS_KEYS = NECESSARY_DATASET_RSF_KEYS

NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_V2_KEYS = NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_KEYS

# =======================================================
# MODEL
# =======================================================

NECESSARY_TRANSFORMER_KEYS = [
    SEQUENCE_LEN,
    "transformer_encoder_head_num",
    "hidden_dim",
    "lstm_num_layers",
    "lstm_dropout",
    "fc_layer_dim",
    "fc_dropout",
    BATCH_SIZE,
    "lr",
    "patience",
    "max_epochs",
]

# NECESSARY_LSTM_KEYS = [
#     "SEQUENCE_LEN,
#     "hidden_dim",
#     "lstm_num_layers",
#     "lstm_dropout",
#     "fc_layer_dim",
#     "fc_dropout",
#     BATCH_SIZE,
#     "lr",
#     "patience",
#     "max_epochs",
# ]

NECESSARY_SELF_SUPERVISED_KEYS = [
    SEQUENCE_LEN,
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
    "second_model",
]

NECESSARY_CNN_KEYS = []

NECESSARY_LSTM_KEYS = [
    "hidden_dim",
    "lstm_num_layers",
    "lstm_dropout",
    "fc_layer_dim",
    "fc_dropout",
]

NECESSARY_TRANSFORMER_FEATURES_KEYS = [
    "transformer_encoder_head_num",
    "transformer_num_layer",
    "fc_layer_dim",
    "fc_dropout",
]

NECESSARY_COPROG_TRANSFORMER_TIME_SEQUENCE_KEYS = [
    "transformer_encoder_head_num",
    "transformer_num_layer",
    "fc_layer_dim",
    "fc_dropout",
]

NECESSARY_PYCLUS_KEYS = [
    "n_trees",
    "max_depth",
    "min_leaf_size",
    "cv_for_grid_search",
    "pruning_method",
    "variance_warning_threshold",
]

# NECESSARY_CNN_KEYS = [
#     SEQUENCE_LEN,
#     BATCH_SIZE,
#     "lr",
#     "patience",
#     "max_epochs",
# ]

NECESSARY_CO_TRAINING_ENSEMBLE_KEYS = [
    SEQUENCE_LEN,
    "coprog_iterations",
    "coprog_suspension_pool_size",
    "coprog_add_ratio",
    "coprog_fine_tune_max_epochs",
    "is_fine_tuning_during_finding_best_suspension_data",
    "is_fine_tuning_for_last_step",
    "selection_mode_str",
    "max_epochs",
    "patiences",
    "batchs_size",
    "lr",
    "shuffle_dataloaders",
    "fine_tune_lr_factor",
    "forgetting_warning_tolerance",
    "hidden_dim_lstm",
    "lstm_num_layers_lstm",
    "lstm_dropout_lstm",
    "fc_layer_dim_lstm",
    "fc_dropout_lstm",
    "transformer_encoder_head_num_transformer_features",
    "fc_layer_dim_transformer_features",
    "fc_dropout_transformer_features",
    "transformer_encoder_head_num_transformer_time_series",
    "fc_layer_dim_transformer_time_series",
    "fc_dropout_transformer_time_series",
    "num_layers_transformer_features",
    "num_layers_transformer_time_series",
]

NECESSARY_CO_TRAINING_ENSEMBLE_V2_KEYS = [
    SEQUENCE_LEN,
    "coprog_iterations",
    "coprog_suspension_pool_size",
    "coprog_add_ratio",
    "max_epochs",
    "patiences",
    "batchs_size",
    "lr",
    "confidence",
    "shuffle_dataloaders",
    "hidden_dim_lstm",
    "lstm_num_layers_lstm",
    "lstm_dropout_lstm",
    "fc_layer_dim_lstm",
    "fc_dropout_lstm",
    "transformer_encoder_head_num_transformer_features",
    "fc_layer_dim_transformer_features",
    "fc_dropout_transformer_features",
    "transformer_encoder_head_num_transformer_time_series",
    "fc_layer_dim_transformer_time_series",
    "fc_dropout_transformer_time_series",
    "num_layers_transformer_features",
    "num_layers_transformer_time_series",
]

# =======================================================
# TRAINING
# =======================================================

NECESSARY_TRAINING_COPROG_KEYS = [
    "lr",
    "patiences",
    "max_epochs",
    "coprog_iterations",
    "coprog_suspension_pool_size",
]
