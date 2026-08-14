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

NECESSARY_DATASET_PYCLUS_KEYS = NECESSARY_DATASET_RSF_KEYS

NECESSARY_DATASET_TRANSFORMER_FEATURES_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
NECESSARY_DATASET_TRANSFORMER_TIME_SEQUENCE_KEYS = NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS

# Dynamic DeepHit (TF rewrite) trains on one full vehicle history per sample, not a fixed sliding
# window: "sequence_len" is computed at run time (the longest vehicle's own readout count) rather
# than configured, and pad_mode is hard-coded to "nan" internally -- neither is a config knob for
# this model, unlike every windowed model. There is therefore no DataLoader either (no
# batch_size/num_workers/pin_memory/shuffle_loader/return_sequence_label). This also means Dynamic
# DeepHit gets its own dataset cache (a different sequence_len) and cannot usefully share
# --force-load-from-cache with the windowed models.
NECESSARY_DATASET_DYNAMIC_DEEPHIT_KEYS = [
    "seed",
    "data_fraction",
    "val_rate",
    "test_rate",
    "stratify",
    "norm_type",
    "counter_mode",
    "include_histograms",
    "histogram_mode",
]

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

# Dynamic DeepHit (TF rewrite) architecture -- these mirror dynamic_deephit/main.py's own
# "network_settings" dict field names directly, so the paper's own hyperparameters drop straight
# into this config. FC_active_fn/RNN_active_fn are "relu"/"tanh" strings here (mapped to
# tf.nn.relu/tf.nn.tanh inside the trainer, since JSON cannot hold a function reference).
NECESSARY_DYNAMIC_DEEPHIT_KEYS = [
    "h_dim_RNN",
    "h_dim_FC",
    "num_layers_RNN",
    "num_layers_ATT",
    "num_layers_CS",
    "RNN_type",
    "FC_active_fn",
    "RNN_active_fn",
    "reg_W",
    "reg_W_out",
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

# Dynamic DeepHit (TF rewrite) training -- mirrors dynamic_deephit/main.py's own "new_parser"
# field names (mb_size/iteration_burn_in/iteration/keep_prob/lr_train/alpha/beta/gamma) directly.
# There is no early stopping (matching main.py): "iteration" always runs to completion, and the
# checkpoint whose validation C-index was best (computed every "eval_every" iterations, over
# "c_index_time_quantiles" absolute-time horizons, capped to "c_index_max_vehicles" validation
# vehicles since the metric is O(n^2)) is restored afterwards. "num_category_bins" is the target
# number of absolute-time discretization bins (see utils_dynamic_deephit._build_time_bin_edges).
NECESSARY_TRAINING_DYNAMIC_DEEPHIT_KEYS = [
    "mb_size",
    "burn_in_mode",
    "iteration_burn_in",
    "iteration",
    "keep_prob",
    "lr_train",
    "alpha",
    "beta",
    "gamma",
    "eval_every",
    "num_category_bins",
    "use_gpu",
    "c_index_time_quantiles",
    "c_index_max_vehicles",
    "inference_batch_size",
]

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

# =======================================================
# HYPERPARAMETER OPTIMIZATION (CoTrainingEnsemble_v2, Scania)
# =======================================================

# Every lever swept by the benchmark. Each key maps to a *list* of candidate values in the
# hyper_parameter_optimization_co_training_ensemble_v2.json "hyper_parameters" block; the
# benchmark takes their (pruned, deduped) cartesian product. suspension_pool_size and add_ratio
# are swept too, so unlike NECESSARY_TRAINING_CO_TRAINING_ENSEMBLE_V2_KEYS they live here and
# not in the fixed training block.
NECESSARY_HPO_CO_TRAINING_ENSEMBLE_V2_HYPER_PARAMETER_KEYS = [
    "use_average_window_confidence",
    # None disables the filter; a float value is the width cutoff.
    "confidence_width_threshold",
    "use_monotone_projection",
    "monotone_residual_weight",
    # Only read when use_monotone_projection is True.
    "disable_isotonic_regression",
    "use_fine_tuning",
    "fine_tune_lr_factor",
    "fine_tune_max_epochs",
    # Paired element-wise with fine_tune_max_epochs (same length), never crossed with it.
    "fine_tune_patience",
    "fine_tune_from_initial_model",
    "peer_weighted_pseudo_label",
    "keep_best_model_mode",
    "isotonic_time_weighting",
    "computing_weight_mode",
    "use_cotraining_ensemble_survival_loss_function",
    "cotraining_survival_loss_lambda",
    "suspension_pool_size",
    "add_ratio",
    "use_mondrian_categorizer",
    # Only read when use_mondrian_categorizer is True.
    "mondrian_no_bins",
    "use_cps",
    "difficulty_estimator_k",
]

# Held constant across every configuration of the sweep. train_with_censored_data is optional
# (defaults to False) and only affects the one-off shared initial training.
NECESSARY_HPO_CO_TRAINING_ENSEMBLE_V2_TRAINING_KEYS = [
    "iterations",
    "confidence",
    "inference_batch_size",
    "bagging_failure_data",
]
