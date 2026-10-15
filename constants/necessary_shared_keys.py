SEED="seed"
VALIDATION_RATE="val_rate"
CALIB_RATE="calib_rate"
SEQUENCE_LEN="sequence_len"
BATCH_SIZE="batch_size"

COPROG_FIRST_MODEL="first_model"
COPROG_SECOND_MODEL="second_model"

MODELS="models"

NECESSARY_CONFORMAL_CO_TRAINING_ENSEMBLE_KEYS = [
    MODELS,
]


# =======================================================
# TRAINING PARAMS FOR CONFORMAL CO-TRAINING ENSEMBLE
# =======================================================
LEARNING_RATE = "lr"
PATIENCES = "patiences"
MAX_EPOCHS = "max_epochs"
COPROG_ITERATIONS = "coprog_iterations"
COPROG_SUSPENSION_POOL_SIZE = "coprog_suspension_pool_size"

# =======================================================
# TRAINING PARAMS FOR CONFORMAL CO-TRAINING ENSEMBLE
# =======================================================
ITERATIONS = "iterations"
SUSPENSION_POOL_SIZE = "suspension_pool_size"
ADD_RATIO = "add_ratio"
CONFIDENCE = "confidence"
INFERENCE_BATCH_SIZE = "inference_batch_size"
TRAIN_WITH_CENSORED_DATA = "train_with_censored_data"
ISOTONIC_TIME_WEIGHTING = "isotonic_time_weighting"

NECESSARY_TRAINING_CONFORMAL_CO_TRAINING_ENSEMBLE_KEYS = [
    ITERATIONS,
    SUSPENSION_POOL_SIZE,
    ADD_RATIO,
    CONFIDENCE,
    # Chunk size for inference forward passes; caps peak memory during conformal scoring/metrics.
    INFERENCE_BATCH_SIZE,
    ISOTONIC_TIME_WEIGHTING,
]