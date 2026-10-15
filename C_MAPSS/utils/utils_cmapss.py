from constants import necessary_keys_cmapss
from shared.enums import ModelVersion
from shared.utils import necessary_keys as shared_necessary_keys
from shared.utils.config import extract_data_from_config


def extract_benchmark_information_from_config(config_benchmark_file_path: str) -> tuple[
    list[float], list[float], list[str]]:
    """
    Extract benchmark information from config file

    :param config_benchmark_file_path: th path to the config benchmark gile
    :return:
    """
    config = extract_data_from_config(config_benchmark_file_path)

    broken_percentage: list[float] = config["broken_percentage"]
    censored_percentage: list[float] = config["censored_percentage"]
    cmapss_files: list[str] = config["cmapss_files"]

    return broken_percentage, censored_percentage, cmapss_files

def get_necessary_dataset_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_dataset_keys(model_version, necessary_keys_cmapss)


def get_necessary_model_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_model_keys(model_version, necessary_keys_cmapss)


def get_necessary_training_keys(model_version: ModelVersion) -> list[str]:
    return shared_necessary_keys.get_necessary_training_keys(model_version, necessary_keys_cmapss)
