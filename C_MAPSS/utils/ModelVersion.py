from enum import Enum


class ModelVersion(Enum):
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    AUTOENCODER = "autoencoder"
    METRIC = "metric"
    RSF = "rsf"
    PYCLUS = "pyclus"
    COPROG = "coprog"
    CNN = "cnn"
    CO_TRAINING_ENSEMBLE = "co_training_ensemble"
