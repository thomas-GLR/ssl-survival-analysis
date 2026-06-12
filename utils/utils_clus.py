import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

import torch

from sequitur.models import LSTM_AE
from sequitur import quick_train
from sequitur.models.lstm_ae import Encoder

import constants.c_mapss_columns as c_mapss_cols

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCANIA_COMPONENT_X_DIR = DATA_DIR / "scania_components_x"
C_MAPSS_DIR = DATA_DIR / "C_MAPSS"

C_MAPSS_COLUMNS = [
    c_mapss_cols.UNIT_NUMBER,
    c_mapss_cols.TIME,
    c_mapss_cols.OPERATIONAL_SETTINGS_1,
    c_mapss_cols.OPERATIONAL_SETTINGS_2,
    c_mapss_cols.OPERATIONAL_SETTINGS_3,
    c_mapss_cols.SENSOR_MEASUREMENT_1,
    c_mapss_cols.SENSOR_MEASUREMENT_2,
    c_mapss_cols.SENSOR_MEASUREMENT_3,
    c_mapss_cols.SENSOR_MEASUREMENT_4,
    c_mapss_cols.SENSOR_MEASUREMENT_5,
    c_mapss_cols.SENSOR_MEASUREMENT_6,
    c_mapss_cols.SENSOR_MEASUREMENT_7,
    c_mapss_cols.SENSOR_MEASUREMENT_8,
    c_mapss_cols.SENSOR_MEASUREMENT_9,
    c_mapss_cols.SENSOR_MEASUREMENT_10,
    c_mapss_cols.SENSOR_MEASUREMENT_11,
    c_mapss_cols.SENSOR_MEASUREMENT_12,
    c_mapss_cols.SENSOR_MEASUREMENT_13,
    c_mapss_cols.SENSOR_MEASUREMENT_14,
    c_mapss_cols.SENSOR_MEASUREMENT_15,
    c_mapss_cols.SENSOR_MEASUREMENT_16,
    c_mapss_cols.SENSOR_MEASUREMENT_17,
    c_mapss_cols.SENSOR_MEASUREMENT_18,
    c_mapss_cols.SENSOR_MEASUREMENT_19,
    c_mapss_cols.SENSOR_MEASUREMENT_20,
    c_mapss_cols.SENSOR_MEASUREMENT_21
]


def create_target_dataframe_for_c_mapss_train_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target dataframe from the C-MAPSS dataset.
    It must contain columns : ["unit_number", "time"] and optionally "event" column.
    If the "event" column does not exist it will be created with value 1 for all rows.

    :param df: the original dataframe of the C-MAPSS dataset.
    :return: a dataframe with columns ["id", "time_to_event", "event"]
    where "id" is the unit number, "time_to_event" is the time when the event has been observed and "event" is the event indicator.
    """
    if "event" not in df.columns:
        # By default, in C_MAPSS data, there is no unlabeled or censored data
        df.insert(len(df.columns), "event", 1)

    target_df = df.groupby(c_mapss_cols.UNIT_NUMBER).agg(
        time_to_event=pd.NamedAgg(c_mapss_cols.TIME, aggfunc="max"),
        event=pd.NamedAgg("event", aggfunc="max") # all values for one unit will have the same values
    ).reset_index()

    target_df = target_df.rename(columns={c_mapss_cols.UNIT_NUMBER: "id"})

    logger.info("Train target dataframe has been created :\n%s", target_df.head().to_string())

    return target_df


def create_target_dataframe_for_c_mapss_test_dataset(df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target dataframe from the C-MAPSS dataset.
    It must contain columns : ["unit_number", "time"] and optionally "event" column.
    If the "event" column does not exist it will be created with value 1 for all rows.

    :param rul_df: the list of RUL to predict for each unit.
    :param df: the original dataframe of the C-MAPSS dataset.
    :return: a dataframe with columns ["id", "time_to_event", "event"]
    where "id" is the unit number, "time_to_event" is the time when the event has been observed and "event" is the event indicator.
    """
    if "event" not in df.columns:
        df.insert(len(df.columns), "event", 1)

    target_df = df.groupby(c_mapss_cols.UNIT_NUMBER).agg(
        event=pd.NamedAgg("event", aggfunc="max") # all values for one unit will have the same values
    ).reset_index()

    last_cycles = df.groupby(c_mapss_cols.UNIT_NUMBER)[c_mapss_cols.TIME].max().reset_index()
    last_cycles.columns = [c_mapss_cols.UNIT_NUMBER, 'last_cycle']

    rul_df[c_mapss_cols.UNIT_NUMBER] = np.unique(df[c_mapss_cols.UNIT_NUMBER].values)
    rul_df = rul_df.rename(columns={c_mapss_cols.TIME: 'rul'})

    target_df = target_df.merge(last_cycles, on=c_mapss_cols.UNIT_NUMBER)
    target_df = target_df.merge(rul_df, on=c_mapss_cols.UNIT_NUMBER)

    # Temps absolu = dernier cycle observé + RUL restant
    target_df['time_to_event'] = target_df['last_cycle'] + target_df['rul']

    target_df = target_df.rename(columns={c_mapss_cols.UNIT_NUMBER: 'id'})

    target_df = target_df[['id', 'time_to_event', 'event']]

    logger.info("Test target dataframe has been created :\n%s", target_df.head().to_string())

    return target_df


def prepare_sequence(data, feature_cols: list[str], time_column: str):
    """
    Prepare a sequence for a group of data. The sequence is sorted by the time column and only the feature columns are kept.

    :param data: the original data.
    :param feature_cols: the feature columns to keep.
    :param time_column: the name of the time column.
    :return: the prepared sequence as a tensor of shape (seq_len, n_features).
    """
    group = data.sort_values(time_column)

    seq = group[feature_cols].values
    return torch.tensor(seq, dtype=torch.float32)


def aggregate_data_per_id(
        grp,
        id_column: str,
        time_column: str
):
    grp = grp.sort_values(time_column)
    row = {}

    feature_columns = [c for c in grp.columns if c not in [id_column, time_column, "event"]]

    for col in feature_columns:
        vals = grp[col].dropna()
        times = grp.loc[grp[col].notna(), time_column]

        if len(vals) == 0:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_last"] = np.nan
            row[f"{col}_slope"] = np.nan
            row[f"{col}_max"] = np.nan
        else:
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_last"] = vals.iloc[-1]
            row[f"{col}_max"] = vals.max()
            # Slope in linear regression if ≥ 2 values
            if len(vals) >= 2:
                slope = np.polyfit(times, vals, 1)[0]
            else:
                slope = 0.0
            row[f"{col}_slope"] = slope

    # Méta-features temporelles utiles
    row["obs_duration"] = grp[time_column].max() - grp[time_column].min()
    row["n_obs"] = len(grp)

    return pd.Series(row)


def transform_time_series_dataset_to_one_time_column_dataset(
        df: pd.DataFrame,
        id_column: str,
        time_column: str,
        target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform a time series dataset to one time event column for CLUS software.

    :param df: the original dataframe to transform.
    :param id_column: the id column
    :param time_column: the time column
    :param target_df: the target dataframe must contain columns : ["id", "time_to_event", "event"]
    :return: the new dataframe with new columns
    """

    if set(target_df.columns) != {"id", "time_to_event", "event"}:
        raise RuntimeError(f"The target dataframe must contain columns : ['id', 'time_to_event', 'event'] but he contain the following columns: [{target_df.columns.tolist()}]")

    transformed_df = df.groupby(id_column).apply(
        lambda grp: aggregate_data_per_id(grp, id_column, time_column),
        include_groups=False
    ).reset_index()

    transformed_df = transformed_df.rename(columns={c_mapss_cols.UNIT_NUMBER: "id"})

    logger.info("New dataset with statistics informations :\n%s", transformed_df.head().to_string())

    transformed_df = transformed_df.merge(target_df, on="id", how="inner")

    logger.info("Adding the target :\n%s", transformed_df.head().to_string())

    return transformed_df


def encode_time_series_dataset_to_one_time_column_dataset(
        df: pd.DataFrame,
        id_column: str,
        time_column: str,
        target_df: pd.DataFrame,
        encoder: Encoder | None = None
) -> tuple[pd.DataFrame, Encoder]:
    """
    Encode a time series dataset to on time column dataset thanks to LSTM encoder.

    :param df: the dataframe of the dataset to convert.
    :param id_column: the ID column name of the dataset.
    :param time_column: the time column name of the dataset.
    :param target_df: the target dataframe must contain columns : ["id", "time_to_event", "event"]
    :param encoder: the encoder if exist to embed the time series. If it does not exist it will be created.
    :return: the converted dataframe and the encoder.
    """

    if set(target_df.columns) != {"id", "time_to_event", "event"}:
        raise RuntimeError("The target dataframe must contain columns : ['id', 'time_to_event', 'event']")

    feature_cols = [c for c in df.columns if c not in [id_column, time_column]]

    train_seqs = [
        prepare_sequence(grp, feature_cols, time_column)
        for _, grp in df.groupby(id_column)
    ]

    encoding_dim = 16 # encoding_dim = 64 -> for lot of individual

    if encoder is None:
        logger.info("Training the encoder...")

        encoder, _, _, _ = quick_train(
            LSTM_AE,
            train_seqs,
            encoding_dim=encoding_dim,
            h_dims=[64], # h_dims=[256, 128, 64], -> for lot of individual
            epochs=100,
            lr=1e-3,
            verbose=True,
        )

    embeddings = []
    units_ids = df[id_column].unique()

    for unit_id, seq in zip(units_ids, train_seqs):
        z = encoder(seq)
        embeddings.append(z.detach().numpy())

    embedding_cols = [f"emb_{i}" for i in range(encoding_dim)]
    df_flat = pd.DataFrame(embeddings, columns=embedding_cols)
    df_flat.insert(0, "id", units_ids)

    logger.info("New embeded dataset thanks to encoder :\n%s", df_flat.head().to_string())

    df_flat = df_flat.merge(target_df, on="id", how="inner")

    logger.info("Adding the target to the dataframe :\n%s", df_flat.head().to_string())

    return df_flat, encoder


def transform_dataset_in_arff_format_and_create_settings(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        time_to_event_column: str,
        id_column: str,
        event_column: str,
        values_by_categorical_column: dict[str, list[str]],
        arff_file_name: str = "dataset",
        settings_file_name: str = "settings"
) -> None:
    """
    Transform a dataset into arff format file for CLUS and save it. Create a settings file and save it too.
    The code come from the paper named "Survival analysis with semi-supervised predictive clustering trees".
    The source code come from the repository that is given in the paper : http://source.ijs.si/tstepisnik/survival-pct-pipeline

    :param train_df: the train dataframe.
    :param test_df: the test dataframe.
    :param time_to_event_column: the time to event column name.
    :param id_column: the ID column name of the dataset.
    :param event_column: the event column name of the dataset.
    :param values_by_categorical_column: a dictionary of values by the name of categorical column.
    :param arff_file_name: the name of the arff file to save, the train and test dataset will be saved with suffix "_train" and "_test" respectively.
    :param settings_file_name: settings file name
    """
    train_file_name = arff_file_name + "_train"
    test_file_name = arff_file_name + "_test"

    # find all unique timestamps in the training data
    stamps = sorted(list(set(train_df[time_to_event_column].astype(int))))

    # Prepare data in arff format
    for file_name, dataset in [(train_file_name, train_df), (test_file_name, test_df)]:
        with open(f"{file_name}.arff", 'w') as f:
            print('@relation survival', file=f)

            columns = dataset.columns.tolist()

            # other columns
            for c in columns:
                if c in [time_to_event_column, event_column]:
                    continue
                elif c == id_column:
                    attr_type = 'key'
                elif c in values_by_categorical_column:
                    attr_type = '{' + ', '.join(values_by_categorical_column[c]) + '}'
                else:
                    attr_type = 'numeric'

                print(f'@attribute {c} {attr_type}', file=f)

            # status split by timestamps
            for s in stamps:
                print(f'@attribute time_{s} numeric', file=f)

            print('', file=f)
            print('@data', file=f)

            for _, row in dataset.iterrows():
                features = [str(row[id_column])]
                features += [str(row[c]) for c in columns if
                             c not in [event_column, time_to_event_column, id_column]]
                stamp = float(row[time_to_event_column])
                status = int(row[event_column])
                states = []
                for t in stamps:
                    if t < stamp:
                        # The event didn't occur
                        states.append('1')
                    elif status == 1:
                        # The event occur
                        states.append('0')
                    else:
                        # Censored data
                        states.append('?')
                print(','.join(features + states), file=f)

    # prepare the settings file for CLUS
    n_features = len([c for c in train_df.columns if
                      c not in [time_to_event_column, event_column]])
    n_stamps = len(stamps)
    n_all = n_features + n_stamps

    clustering_weights = [0] + [1] * (n_features - 1) + [stamps[0]]
    for i in range(1, len(stamps)):
        clustering_weights.append(stamps[i] - stamps[i - 1])

    settings_content = f"""
[General]
Verbose = 0

[Data]
File = {train_file_name}.arff
TestSet = {test_file_name}.arff

[Attributes]
Key = 1
Descriptive = 2-{n_features}
Target = {n_features + 1}-{n_all}
Clustering = 2-{n_all}
ClusteringWeights = {clustering_weights}

[Tree]
Heuristic = VarianceReduction
MissingClusteringAttrHandling = EstimateFromParentNode
MissingTargetAttrHandling = ParentNode
PruningMethod = M5

[Ensemble]
EnsembleMethod = RForest
Iterations = 100
SelectRandomSubspaces = SQRT
WriteEnsemblePredictions = Yes
NumberOfThreads = 8

[Output]
TrainErrors = Yes
TestErrors = Yes

[SemiSupervised]
SemiSupervisedMethod = PCT
PercentageLabeled = 100
PruningWhenTuning = No
InternalFolds = 3
%WeightScoresFile = weights.txt
PossibleWeights = [0.25,0.5,0.75]
    """

    with open(f'{settings_file_name}.s', 'w') as f:
        print(settings_content, file=f)


def generate_c_mapss_files_for_clus(
        information_in_file_name_to_generate: str="",
        encode_time_varying_covariate_with_lstm: bool=False
):
    files_types = ["FD001", "FD002", "FD003", "FD004"]
    files_names = os.listdir(C_MAPSS_DIR)

    for file_type in files_types:
        logger.info("################################################## Generating files for %s ##################################################", file_type)

        train_file_name = f"train_{file_type}.txt"
        test_file_name = f"test_{file_type}.txt"
        rul_file_name = f"RUL_{file_type}.txt"

        if train_file_name not in files_names or test_file_name not in files_names:
            raise RuntimeError(f"Files {train_file_name} and {test_file_name} must be in the directory {C_MAPSS_DIR}")

        train_df = pd.read_csv(C_MAPSS_DIR / train_file_name, sep=r'\s+', names=C_MAPSS_COLUMNS, header=None)
        test_df = pd.read_csv(C_MAPSS_DIR / test_file_name, sep=r'\s+', names=C_MAPSS_COLUMNS, header=None)
        rul_df = pd.read_csv(C_MAPSS_DIR / rul_file_name, sep=r'\s+', names=[c_mapss_cols.TIME], header=None)

        logger.info("Creating the target dataframe for the train and test dataset")
        target_train_df = create_target_dataframe_for_c_mapss_train_dataset(train_df)
        target_test_df = create_target_dataframe_for_c_mapss_test_dataset(test_df, rul_df)

        if encode_time_varying_covariate_with_lstm:
            train_df, encoder = encode_time_series_dataset_to_one_time_column_dataset(
                train_df,
                id_column=c_mapss_cols.UNIT_NUMBER,
                time_column=c_mapss_cols.TIME,
                target_df=target_train_df
            )

            test_df, _ = encode_time_series_dataset_to_one_time_column_dataset(
                test_df,
                id_column=c_mapss_cols.UNIT_NUMBER,
                time_column=c_mapss_cols.TIME,
                target_df=target_test_df,
                encoder=encoder
            )
        else:
            train_df = transform_time_series_dataset_to_one_time_column_dataset(
                train_df,
                id_column=c_mapss_cols.UNIT_NUMBER,
                time_column=c_mapss_cols.TIME,
                target_df=target_train_df
            )

            test_df = transform_time_series_dataset_to_one_time_column_dataset(
                test_df,
                id_column=c_mapss_cols.UNIT_NUMBER,
                time_column=c_mapss_cols.TIME,
                target_df=target_test_df
            )

        transform_dataset_in_arff_format_and_create_settings(
            train_df=train_df,
            test_df=test_df,
            time_to_event_column="time_to_event",
            id_column="id",
            event_column="event",
            values_by_categorical_column={},
            arff_file_name=f"c_mapss_{file_type}_{information_in_file_name_to_generate}",
            settings_file_name=f"settings_{file_type}"
        )


def parse_clus_preds(filepath):
    """Parse a CLUS ensemble .preds ARFF file into a DataFrame."""
    attrs = []
    data_lines = []
    in_data = False

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.upper() == '@DATA':
                in_data = True
                continue
            if not in_data and line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                attrs.append(parts[1])
            elif in_data and line and not line.startswith('%'):
                data_lines.append(line)

    rows = [list(map(float, l.split(','))) for l in data_lines]
    return pd.DataFrame(rows, columns=attrs)


def rul_from_survival(time_pts, s_pred):
    """Median survival time = first t where S(t) drops below 0.5."""
    t = np.array(time_pts)
    s = np.array(s_pred)
    below = np.where(s <= 0.5)[0]
    if len(below) == 0:
        return t[-1]  # Survived past all observed times
    return t[below[0]]


if __name__ == "__main__":
    # generate_c_mapss_files_for_clus(information_in_file_name_to_generate="statistics", encode_time_varying_covariate_with_lstm=False)

    train_df = parse_clus_preds('settings_FD001.ens.train.preds')
    test_df = parse_clus_preds('settings_FD001.ens.test.preds')

    # Split into the three groups
    time_cols = [c for c in train_df.columns if not c.endswith('-pred')
                 and not c.endswith('-stdev') and c != 'id']
    pred_cols = [c + '-pred' for c in time_cols]
    stdev_cols = [c + '-stdev' for c in time_cols]

    # Extract time points as integers
    time_points = [int(c.replace('time_', '')) for c in time_cols]

    # Actual vs predicted survival curves for engine unit 0
    unit = 0
    actual = test_df[time_cols].iloc[unit].values
    predicted = test_df[pred_cols].iloc[unit].values
    uncertainty = test_df[stdev_cols].iloc[unit].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_points, actual, label='Actual S(t)', color='black', lw=2)
    ax.plot(time_points, predicted, label='Predicted S(t)', color='steelblue', lw=2)
    ax.fill_between(time_points,
                    predicted - uncertainty,
                    predicted + uncertainty,
                    alpha=0.3, color='steelblue', label='±1 std (ensemble)')
    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Survival probability S(t)')
    ax.set_title('SSL-PCT Ensemble — Engine Unit 1, FD001')
    ax.legend()
    plt.tight_layout()
    plt.savefig('survival_curve_unit1.png', dpi=150)

    test_df['RUL_pred'] = [
        rul_from_survival(time_points, test_df[pred_cols].iloc[i].values)
        for i in range(len(test_df))
    ]
    print(test_df[['id', 'RUL_pred']].head(10))

    # Test avec le fichier de test
    test_raw = pd.read_csv(C_MAPSS_DIR / 'test_FD001.txt', sep=' ', header=None)
    test_raw.columns = ['unit', 'cycle', *[f's{i}' for i in range(1, test_raw.shape[1] - 1)]]

    last_cycle = test_raw.groupby('unit')['cycle'].max().reset_index()
    last_cycle.columns = ['id', 'last_cycle']

    # Fusionner avec vos prédictions
    result = test_df[['id', 'RUL_pred']].copy()
    result['id'] = result['id'].astype(int)
    result = result.merge(last_cycle, on='id')

    result['RUL_remaining'] = result['RUL_pred'] - result['last_cycle']

    # Comparer avec le ground truth
    rul_gt = pd.read_csv(C_MAPSS_DIR / 'RUL_FD001.txt', header=None, names=['RUL_true'])
    rul_gt['id'] = range(1, len(rul_gt) + 1)
    result = result.merge(rul_gt, on='id')

    result.to_csv("result.csv")

    print(result[['id', 'last_cycle', 'RUL_pred', 'RUL_remaining', 'RUL_true']])
