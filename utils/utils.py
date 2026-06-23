import pandas as pd
import numpy as np
from pathlib import Path
import os
import datetime

from lifelines.utils import add_covariate_to_timeline, to_long_format

import constants.scania_component_x_columns as scania_cols
import constants.c_mapss_columns as c_mapss_cols

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCANIA_COMPONENT_X = DATA_DIR / "scania_components_x"
C_MAPSS = DATA_DIR / "C_MAPSS"

TRAIN_OPERATIONAL_READOUTS_COLUMNS_NAMES = ["vehicle_id", "time_step", "171_0", "666_0", "427_0", "837_0", "167_0",
                                            "167_1", "167_2", "167_3", "167_4", "167_5", "167_6", "167_7", "167_8",
                                            "167_9", "309_0", "272_0", "272_1", "272_2", "272_3", "272_4", "272_5",
                                            "272_6", "272_7", "272_8", "272_9", "835_0", "370_0", "291_0", "291_1",
                                            "291_2", "291_3", "291_4", "291_5", "291_6", "291_7", "291_8", "291_9",
                                            "291_10", "158_0", "158_1", "158_2", "158_3", "158_4", "158_5", "158_6",
                                            "158_7", "158_8", "158_9", "100_0", "459_0", "459_1", "459_2", "459_3",
                                            "459_4", "459_5", "459_6", "459_7", "459_8", "459_9", "459_10", "459_11",
                                            "459_12", "459_13", "459_14", "459_15", "459_16", "459_17", "459_18",
                                            "459_19", "397_0", "397_1", "397_2", "397_3", "397_4", "397_5", "397_6",
                                            "397_7", "397_8", "397_9", "397_10", "397_11", "397_12", "397_13", "397_14",
                                            "397_15", "397_16", "397_17", "397_18", "397_19", "397_20", "397_21",
                                            "397_22", "397_23", "397_24", "397_25", "397_26", "397_27", "397_28",
                                            "397_29", "397_30", "397_31", "397_32", "397_33", "397_34", "397_35"]

TRAIN_TTE_COLUMNS_NAMES = [scania_cols.VEHICLE_ID, scania_cols.LENGTH_OF_STUDY_TIME_STEP, scania_cols.IN_STUDY_REPAIR]

TRAIN_SPECIFICATIONS_COLUMNS_NAMES = [scania_cols.VEHICLE_ID, scania_cols.SPEC_0, scania_cols.SPEC_1,
                                      scania_cols.SPEC_2, scania_cols.SPEC_3, scania_cols.SPEC_4,
                                      scania_cols.SPEC_5, scania_cols.SPEC_6, scania_cols.SPEC_7]

SPECIFICATIONS_CATEGORICAL_COLUMNS = [scania_cols.SPEC_0, scania_cols.SPEC_1, scania_cols.SPEC_2, scania_cols.SPEC_3,
                                      scania_cols.SPEC_4, scania_cols.SPEC_5,
                                      scania_cols.SPEC_6, scania_cols.SPEC_7]


def load_train_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train_operational_readouts = pd.read_csv(SCANIA_COMPONENT_X / "train_operational_readouts.csv")
    df_train_tte = pd.read_csv(SCANIA_COMPONENT_X / "train_tte.csv")
    df_train_specifications = pd.read_csv(SCANIA_COMPONENT_X / "train_specifications.csv")

    return df_train_operational_readouts, df_train_tte, df_train_specifications


def load_test_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test_operational_readouts = pd.read_csv(SCANIA_COMPONENT_X / "test_operational_readouts.csv")
    df_test_labels = pd.read_csv(SCANIA_COMPONENT_X / "test_labels.csv")
    df_test_specifications = pd.read_csv(SCANIA_COMPONENT_X / "test_specifications.csv")

    return df_test_operational_readouts, df_test_labels, df_test_specifications


def load_validation_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_validation_operational_readouts = pd.read_csv(SCANIA_COMPONENT_X / "validation_operational_readouts.csv")
    df_validation_labels = pd.read_csv(SCANIA_COMPONENT_X / "validation_labels.csv")
    df_validation_specifications = pd.read_csv(SCANIA_COMPONENT_X / "validation_specifications.csv")

    return df_validation_operational_readouts, df_validation_labels, df_validation_specifications


def create_x_train_y_train() -> tuple[np.ndarray, np.ndarray]:
    df_train_operational_readouts, df_train_tte, df_train_specifications = load_train_dataset()

    pd.get_dummies(df_train_specifications, columns=SPECIFICATIONS_CATEGORICAL_COLUMNS, drop_first=True, dtype=int)

    X_train = pd.merge(df_train_operational_readouts, df_train_specifications, on=scania_cols.VEHICLE_ID)

    X_train = pd.merge(X_train, df_train_tte, on=scania_cols.VEHICLE_ID)

    print(f"Shape before removing : {X_train.shape}")

    # The ground truth = 0 at time t include  in [0, 48] might be wrong because of censoring. Then we removed them.
    X_train = X_train.loc[
        (X_train[scania_cols.IN_STUDY_REPAIR] != 0) |
        ((X_train[scania_cols.LENGTH_OF_STUDY_TIME_STEP] - X_train[scania_cols.TIME_STEP]) >= 48)
        ]

    print(f"Shape after removing : {X_train.shape}")

    conditions = [
        X_train[scania_cols.IN_STUDY_REPAIR] == 0,
        (X_train[scania_cols.IN_STUDY_REPAIR] == 1) & (
                    (X_train[scania_cols.LENGTH_OF_STUDY_TIME_STEP] - X_train[scania_cols.TIME_STEP]) <= 6),
        (X_train[scania_cols.IN_STUDY_REPAIR] == 1) & (
                    (X_train[scania_cols.LENGTH_OF_STUDY_TIME_STEP] - X_train[scania_cols.TIME_STEP]) <= 12),
        (X_train[scania_cols.IN_STUDY_REPAIR] == 1) & (
                    (X_train[scania_cols.LENGTH_OF_STUDY_TIME_STEP] - X_train[scania_cols.TIME_STEP]) <= 24),
        (X_train[scania_cols.IN_STUDY_REPAIR] == 1) & (
                    (X_train[scania_cols.LENGTH_OF_STUDY_TIME_STEP] - X_train[scania_cols.TIME_STEP]) <= 48)
    ]

    values = [
        0,
        4,
        3,
        2,
        1
    ]

    X_train[scania_cols.CLASS_LABEL] = np.select(conditions, values, default=0)

    # We keep time_step and in_study_repair in case we wan't to do survival analysis machine learning.
    Y_train = X_train[scania_cols.TIME_STEP, scania_cols.CLASS_LABEL, scania_cols.IN_STUDY_REPAIR]

    X_train = X_train.drop(
        columns=[scania_cols.CLASS_LABEL, scania_cols.LENGTH_OF_STUDY_TIME_STEP, scania_cols.IN_STUDY_REPAIR,
                 scania_cols.VEHICLE_ID], axis=1)

    return X_train.to_numpy(), Y_train.to_numpy()


def get_train_survival_dataset_time_varying(
        file_name: str = "survival_dataset_time_varying.csv",
        print_information: bool = False,
        force_generation: bool = False) -> pd.DataFrame:
    if os.path.isfile(SCANIA_COMPONENT_X / file_name) and not force_generation:
        print(f"Loading survival dataset from {file_name}...")

        return pd.read_csv(SCANIA_COMPONENT_X / file_name)

    print(f"Generating survival dataset...")

    start = datetime.datetime.now()

    survival_dataset = generate_train_survival_dataset_time_varying(print_information=print_information)

    end = datetime.datetime.now()

    print(f"Survival dataset generated in {end - start}.")

    print(f"Saving survival dataset to {file_name}...")
    survival_dataset.to_csv(SCANIA_COMPONENT_X / file_name, index=False)

    return survival_dataset


def generate_train_survival_dataset_time_varying(print_information: bool = False) -> pd.DataFrame:
    df_train_operational_readouts, df_train_tte, df_train_specifications = load_train_dataset()

    df_train_specifications = pd.get_dummies(df_train_specifications, columns=SPECIFICATIONS_CATEGORICAL_COLUMNS,
                                             drop_first=True, dtype=int)

    if print_information:
        print("#################### df_train_specifications after get_dummies ####################")
        print(df_train_specifications.head())

    survival_dataset = pd.merge(df_train_operational_readouts, df_train_specifications, on=scania_cols.VEHICLE_ID)

    df_train_tte[scania_cols.IN_STUDY_REPAIR] = df_train_tte[scania_cols.IN_STUDY_REPAIR].astype(bool)
    df_train_tte = to_long_format(df_train_tte, duration_col=scania_cols.LENGTH_OF_STUDY_TIME_STEP)

    if print_information:
        print("#################### Train tte after to_long_format ####################")
        print(df_train_tte.head())

    survival_dataset = add_covariate_to_timeline(
        df_train_tte,
        survival_dataset,
        duration_col=scania_cols.TIME_STEP,
        id_col=scania_cols.VEHICLE_ID,
        event_col=scania_cols.IN_STUDY_REPAIR
    )

    if print_information:
        print("#################### Survival dataset after add_covariate_to_timeline ####################")
        print(survival_dataset.head())

    return survival_dataset


def get_train_survival_dataset(print_information: bool = False) -> pd.DataFrame:
    df_train_operational_readouts, df_train_tte, df_train_specifications = load_train_dataset()

    pd.get_dummies(df_train_specifications, columns=SPECIFICATIONS_CATEGORICAL_COLUMNS, drop_first=True, dtype=int)

    survival_dataset = pd.merge(df_train_operational_readouts, df_train_specifications, on=scania_cols.VEHICLE_ID)

    survival_dataset = pd.merge(survival_dataset, df_train_tte, on=scania_cols.VEHICLE_ID)

    idx_max = survival_dataset.groupby(scania_cols.VEHICLE_ID)[scania_cols.TIME_STEP].idxmax()

    survival_dataset = survival_dataset.loc[idx_max]

    if print_information:
        print(survival_dataset.head())

    return survival_dataset.drop(columns=[scania_cols.LENGTH_OF_STUDY_TIME_STEP, scania_cols.VEHICLE_ID], axis=1)

def cmapss_score(predict, label):
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)