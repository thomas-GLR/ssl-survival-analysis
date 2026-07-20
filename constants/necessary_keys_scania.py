# =======================================================
# DATASET
# =======================================================

NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS = [
    "sequence_len",
    "seed",
    "data_fraction",
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
    "histogram_mode",
]

NECESSARY_DATASET_SELF_SUPERVISED_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

NECESSARY_DATASET_RSF_KEYS = [
    "seed",
    "data_fraction",
    "val_rate",
    "test_rate",
    "stratify",
    "norm_type",
    "return_sequence_label",
    "counter_mode",
    "include_histograms",
    "histogram_mode",
]

NECESSARY_DATASET_COPROG_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

# The co-training ensembles use the same windowed dataset as the supervised deep models.
NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_V2_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_V3_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

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

# The co-training ensembles take a variable-length list of models; per-model architecture
# params + per-model lr/max_epochs/patience/rul_target_standardization live inside each list
# entry (validated in scania.utils.utils_cotraining_common.parse_models_config), so only the
# list itself is required at the config-block level.
NECESSARY_CO_TRAINING_ENSEMBLE_KEYS = [
    "models",
]
NECESSARY_CO_TRAINING_ENSEMBLE_V2_KEYS = [
    "models",
]
NECESSARY_CO_TRAINING_ENSEMBLE_V3_KEYS = [
    "models",
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

# Ensemble-level training params only. suspension_pool_size / add_ratio are fractions in
# (0, 1] (CoTrainingEnsemble semantics), unlike COPROG's integer pool count.
NECESSARY_TRAINING_CO_TRAINING_ENSEMBLE_KEYS = [
    "iterations",
    "suspension_pool_size",
    "add_ratio",
    "is_fine_tuning_during_finding_best_suspension_data",
    "is_fine_tuning_for_last_step",
    "fine_tune_lr_factor",
    "fine_tune_max_epochs",
    # Chunk size for inference forward passes; caps peak memory during scoring/metrics.
    "inference_batch_size",
]

NECESSARY_TRAINING_CO_TRAINING_ENSEMBLE_V2_KEYS = [
    "iterations",
    "suspension_pool_size",
    "add_ratio",
    "confidence",
    # Chunk size for inference forward passes; caps peak memory during conformal scoring/metrics.
    "inference_batch_size",
]

# v3 adds the owner-based selection cap (best_ratio), the latent-kNN estimator (n_neighbors) and
# the fine-tuning budget. confidence_tol / model_pred_blend / use_monotone_projection /
# monotone_residual_weight default in the util signature, so they are optional in the config.
NECESSARY_TRAINING_CO_TRAINING_ENSEMBLE_V3_KEYS = [
    "iterations",
    "suspension_pool_size",
    "add_ratio",
    "best_ratio",
    "confidence",
    "n_neighbors",
    "fine_tune_lr_factor",
    "fine_tune_max_epochs",
    "fine_tune_patience",
    # Chunk size for inference forward passes; caps peak memory during CPS scoring / embedding.
    "inference_batch_size",
]
