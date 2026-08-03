from enum import Enum


class ModelVersion(Enum):
    TRANSFORMER_LSTM = "transformer_lstm"
    LSTM = "lstm"
    AUTOENCODER = "autoencoder"
    METRIC = "metric"
    RSF = "rsf"
    PYCLUS = "pyclus"
    COPROG = "coprog"
    CNN = "cnn"
    CO_TRAINING_ENSEMBLE = "co_training_ensemble"
    CO_TRAINING_ENSEMBLE_V2 = "co_training_ensemble_v2"
    CO_TRAINING_ENSEMBLE_ORDINAL = "co_training_ensemble_ordinal"
    TRANSFORMER_FEATURES = "transformer_features"
    TRANSFORMER_TIME_SEQUENCE = "transformer_time_sequence"
