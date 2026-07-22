from types import ModuleType

from shared.utils.ModelVersion import ModelVersion


def get_necessary_dataset_keys(model_version: ModelVersion, necessary_keys_module: ModuleType) -> list[str]:
    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
            return necessary_keys_module.NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
        case ModelVersion.LSTM:
            return necessary_keys_module.NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
        case ModelVersion.AUTOENCODER:
            return necessary_keys_module.NECESSARY_DATASET_SELF_SUPERVISED_KEYS
        case ModelVersion.METRIC:
            return necessary_keys_module.NECESSARY_DATASET_SELF_SUPERVISED_KEYS
        case ModelVersion.RSF:
            return necessary_keys_module.NECESSARY_DATASET_RSF_KEYS
        case ModelVersion.PYCLUS:
            return necessary_keys_module.NECESSARY_DATASET_PYCLUS_KEYS
        case ModelVersion.COPROG:
            return necessary_keys_module.NECESSARY_DATASET_COPROG_KEYS
        case ModelVersion.COBCREG:
            return necessary_keys_module.NECESSARY_DATASET_COBCREG_KEYS
        case ModelVersion.CNN:
            return necessary_keys_module.NECESSARY_DATASET_TRANSFORMER_LSTM_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return necessary_keys_module.NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return necessary_keys_module.NECESSARY_DATASET_CO_TRAINING_ENSEMBLE_V2_KEYS
        case ModelVersion.TRANSFORMER_FEATURES:
            return necessary_keys_module.NECESSARY_DATASET_TRANSFORMER_FEATURES_KEYS
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            return necessary_keys_module.NECESSARY_DATASET_TRANSFORMER_TIME_SEQUENCE_KEYS
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def get_necessary_model_keys(model_version: ModelVersion, necessary_keys_module: ModuleType) -> list[str]:
    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
            return necessary_keys_module.NECESSARY_TRANSFORMER_KEYS
        case ModelVersion.LSTM:
            return necessary_keys_module.NECESSARY_LSTM_KEYS
        case ModelVersion.AUTOENCODER:
            return necessary_keys_module.NECESSARY_SELF_SUPERVISED_KEYS
        case ModelVersion.METRIC:
            return necessary_keys_module.NECESSARY_SELF_SUPERVISED_KEYS
        case ModelVersion.RSF:
            return necessary_keys_module.NECESSARY_RSF_KEYS
        case ModelVersion.PYCLUS:
            return necessary_keys_module.NECESSARY_PYCLUS_KEYS
        case ModelVersion.COPROG:
            return necessary_keys_module.NECESSARY_COPROG_KEYS
        case ModelVersion.COBCREG:
            return necessary_keys_module.NECESSARY_COBCREG_KEYS
        case ModelVersion.CNN:
            return necessary_keys_module.NECESSARY_CNN_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return necessary_keys_module.NECESSARY_CO_TRAINING_ENSEMBLE_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return necessary_keys_module.NECESSARY_CO_TRAINING_ENSEMBLE_V2_KEYS
        case ModelVersion.TRANSFORMER_FEATURES:
            return necessary_keys_module.NECESSARY_TRANSFORMER_FEATURES_KEYS
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            return necessary_keys_module.NECESSARY_TRANSFORMER_TIME_SEQUENCE_KEYS
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")


def get_necessary_training_keys(model_version: ModelVersion, necessary_keys_module: ModuleType) -> list[str]:
    match model_version:
        case ModelVersion.TRANSFORMER_LSTM:
            return necessary_keys_module.NECESSARY_TRAINING_TRANSFORMER_KEYS
        case ModelVersion.LSTM:
            return necessary_keys_module.NECESSARY_TRAINING_LSTM_KEYS
        case ModelVersion.AUTOENCODER:
            raise NotImplementedError(f"Model {model_version.value} is not yet implemented")
        case ModelVersion.METRIC:
            raise NotImplementedError(f"Model {model_version.value} is not yet implemented")
        case ModelVersion.RSF:
            return necessary_keys_module.NECESSARY_TRAINING_RSF_KEYS
        case ModelVersion.PYCLUS:
            raise NotImplementedError(f"Model {model_version.value} is not yet implemented")
        case ModelVersion.COPROG:
            return necessary_keys_module.NECESSARY_TRAINING_COPROG_KEYS
        case ModelVersion.COBCREG:
            return necessary_keys_module.NECESSARY_TRAINING_COBCREG_KEYS
        case ModelVersion.CNN:
            return necessary_keys_module.NECESSARY_TRAINING_CNN_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE:
            return necessary_keys_module.NECESSARY_TRAINING_CO_TRAINING_ENSEMBLE_KEYS
        case ModelVersion.CO_TRAINING_ENSEMBLE_V2:
            return necessary_keys_module.NECESSARY_TRAINING_CO_TRAINING_ENSEMBLE_V2_KEYS
        case ModelVersion.TRANSFORMER_FEATURES:
            return necessary_keys_module.NECESSARY_TRAINING_TRANSFORMER_FEATURES_KEYS
        case ModelVersion.TRANSFORMER_TIME_SEQUENCE:
            return necessary_keys_module.NECESSARY_TRAINING_TRANSFORMER_TIME_SEQUENCE_KEYS
        case _:
            raise ValueError(f"Model version {model_version.value} not supported")