import logging
import os
import subprocess
import shutil
import re
import numpy as np
from sys import stdout, stderr
from typing import Union, Dict, Any, List, Tuple

from .base import PredictiveModel
from pyclus.helpers import Utilities


logger = logging.getLogger(__name__)

DenseData = List[List[Any]]
SparseData = Tuple[List[int], List[int], List[Any]]
RawFeatures = Union[np.ndarray, DenseData, SparseData]
RawTarget = Union[np.ndarray, List[List[Any]], List[Any]]
Data = Union[DenseData, SparseData]


class SettingSections:
    SECTION_GENERAL = "General"
    SECTION_DATA = "Data"
    SECTION_ATTRIBUTES = "Attributes"
    SECTION_MODEL = "Model"
    SECTION_TREE = "Tree"
    SECTION_RULES = "Rules"
    SECTION_ENSEMBLE = "Ensemble"
    SECTION_CONSTRAINTS = "Constraints"
    SECTION_OUTPUT = "Output"
    SECTION_BEAM = "Beam"
    SECTION_MLC = "Multilabel"
    SECTION_HMLC = "Hierarchical"
    SECTION_HMTR = "HMTR"
    SECTION_SEMI_SUPERVISED = "SemiSupervised"
    SECTION_RELIEF = "Relief"
    SECITON_OPTION_TREES = "OptionTree"
    SECTION_KNN = "kNN"

    @staticmethod
    def get_sections():
        return Utilities.get_list_of_allowed_values(
            vars(SettingSections),
            lambda name: name.startswith("SECTION_")
        )


class CommandLineSwitches:
    SWITCH_XVAL = "xval"
    SWITCH_FOLD = "fold"
    SWITCH_RULES = "rules"
    SWITCH_FOREST = "forest"
    SWITCH_BEAM = "beam"
    SWITCH_SIT = "sit"
    SWITCH_SILENT = "silent"
    SWITCH_INFO = "info"
    SWITCH_SSL = "ssl"
    SWITCH_RELIEF = "relief"
    SWITCH_KNN = "knn"

    @staticmethod
    def get_switches():
        return Utilities.get_list_of_allowed_values(
            vars(CommandLineSwitches),
            lambda name: name.startswith("SWITCH_")
        )


class ClusModel(PredictiveModel):
    # some parameters
    VERBOSITY = "verbose"
    RANDOM_STATE = "random_state"
    ENSEMBLE_METHOD = "ensemble_method"
    FEATURE_SUBSET = "feature_subset"
    MIN_LEAF_SIZE = "min_leaf_size"
    N_TREES = "n_trees"
    N_TREES_DEFAULT = 100
    SPLIT_HEURISTIC = "split_heuristic"
    FEATURE_RANKING = "feature_ranking"
    JAVA_PARAMETERS = "java_parameters"
    JAVA_OPEN = Utilities.determine_java_command()

    ALLOWED_SECTIONS = SettingSections.get_sections()
    ALLOWED_SWITCHES = CommandLineSwitches.get_switches()

    MTR = "is_mtr"
    CLUS_MISSING_VALUE = "?"
    HMLC_LABEL_SEPARATOR = "@"
    HMLC_SHAPE_TREE = "Tree"
    HMLC_SHAPE_DAG = "DAG"
    HMLC_SHAPES = [HMLC_SHAPE_DAG, HMLC_SHAPE_TREE]
    COLUMN_TYPE_NOM = "nominal column type"
    COLUMN_TYPE_NUM = "numeric column type"
    COLUMN_TYPE_HIERARCHICAL = "hierarchical"
    TRUE_VALUES_MODEL = "true values"

    # some dir/file names
    EXPERIMENT_NAME = "experiment"
    DATA_FILE = "data.arff"
    TEMP_DIR_PATTERN = os.path.join(PredictiveModel.TEMP_DIR, "temp_clus_run{}")

    def __init__(
            self,
            ensemble_method: str = "RForest",
            feature_subset: Union[float, int, str] = 0.75,
            min_leaf_size=1,
            n_trees: Union[None, int, List[int]] = None,
            verbose=1,
            random_state: Union[None, int] = None,
            feature_ranking="Genie3",
            is_multi_target=True,
            java_parameters=JAVA_OPEN,
            **kwargs):
        """
        Wrapper of ensembles of PCTs implemented in CLUS.
        The values of key-word arguments should be pairs
        of the form 'other<appendix>': (section name, option name, option value)
        :param ensemble_method: string, (default='RForest')
            Determines the ensemble method to use (e.g., 'RForest' or 'Bagging')
        :param feature_subset: int or float, (default=1.0)
            The size of the feature subset considered at each split.
            If given as an integer, that absolute number of features is used.
            If given as a float, it is treated as a percentage.
        :param min_leaf_size: int, (default=1)
            Minimum number of examples in leaves.
        :param n_trees: int, List[int] (default=100)
            Number of trees in the ensemble.
        :param feature_ranking: Genie3 or Symbolic or RForest
        :param is_multi_target: Do we induce multi-target or single-target model?
        :param java_parameters: parameter that are passed to the java virtual machine (JVM),
            e.g., "-Xmx10G" or "-Xmx10G -Xms128M". This argument is stripped and
            then passed verbatim to the JVM. Do not use -jar in it!
        :param kwargs: Key-value pairs, where key is of form

        a) <parameter section>_<parameter name>, and the value is
         the value of the parameter. Use kwargs for some more exotic (and non-default) CLUS options,
         as described in the manual. Example. The number of trees can be equivalently given as
          - n_trees = 100
          - Ensemble_Iterations = 100

          If the parameter is given in both ways, the kwargs value will be used.
        b) <clus command line switch> and the value is a list of arguments for this switch.
          For example, if you want to use ensembles, use forest=[].
          If you want to use semi-supervised learning, use ssl=[] etc.
        """
        super().__init__(None)
        self.random_seed = 1234 if random_state is None else random_state
        self.is_mtr = is_multi_target
        ClusModel._parameter_check(ensemble_method, feature_subset, min_leaf_size,
                                   n_trees, feature_ranking)
        # parameters in the main sections already specified
        # kwargs may add some sections and parameters and override named parameters
        self.method_parameters: Dict[str, Dict[str, Any]] = {
            SettingSections.SECTION_GENERAL: {
                "Verbose": verbose,
                "RandomSeed": self.random_seed
            },
            SettingSections.SECTION_DATA: {},  # later
            SettingSections.SECTION_ATTRIBUTES: {},  # later
            SettingSections.SECTION_ENSEMBLE: {
                "EnsembleMethod": ensemble_method,
                "SelectRandomSubspaces": feature_subset,
                "Iterations": n_trees,
                "Optimize": "Yes", "OOBestimate": "No",
                "EnsembleBootstrapping": "Yes",
                "FeatureRanking": feature_ranking,
                "FeatureRankingPerTarget": "Yes"
            },
            SettingSections.SECTION_MODEL: {
                "MinimalWeight": min_leaf_size,
            },
            SettingSections.SECTION_TREE: {
                "SplitPosition": "Middle"
            },
            SettingSections.SECTION_OUTPUT: {
                "TrainErrors": "No",
                "TestErrors": "No",
                "WritePerBagModelFile": "Yes"
            }
        }
        self.command_line_switches: Dict[str, str] = {}
        self.java_parameters = ""
        self._set_java_parameters(java_parameters)
        self.set_params(**kwargs)
        # special care for ensemble
        self.is_ensemble = False
        if CommandLineSwitches.SWITCH_FOREST in self.command_line_switches:
            self.is_ensemble = True
            if n_trees is None:
                self._set_n_trees(ClusModel.N_TREES_DEFAULT)
        elif n_trees is not None:
            case1 = isinstance(n_trees, int) and n_trees > 1
            case2 = isinstance(n_trees, List) and max(n_trees) > 1
            if case1 or case2:
                self.is_ensemble = True
                self.command_line_switches[CommandLineSwitches.SWITCH_FOREST] = ""
        if n_trees is None:
            self._set_n_trees(ClusModel.N_TREES_DEFAULT)
        # fields that are not that final
        self.n_targets_per_experiment = 0
        # [(model name,  {ranking name: scores, ...}), ...]
        self.feature_importances_: List[Tuple[str, Dict[str, np.ndarray]]] = []
        self.column_types_xs: List[Tuple[str, List[str]]] = []  # [(type, values), ...]
        self.column_types_ys: List[Tuple[str, List[str]]] = []  # [(type, values), ...]
        self.dummy_target = None
        self.model = None
        self.hierarchical_attribute = None  # used in HMLC

    @staticmethod
    def _parameter_check(ensemble_method, feature_subset, min_leaf_size, n_trees, feature_ranking):
        """
        Some simple parameter checks.
        :param ensemble_method:
        :param feature_subset:
        :param min_leaf_size:
        :param n_trees:
        :param feature_ranking:
        :return:
        """
        choice_message = "{} must be en element of but is {}"
        allowed_ensembles = ["RForest", "Bagging", "ExtraTrees"]
        if ensemble_method not in allowed_ensembles:
            raise ValueError(
                choice_message.format("ensemble_method", allowed_ensembles, ensemble_method)
            )
        allowed_rankings = ["RForest", "Genie3", "Symbolic"]
        if feature_ranking not in allowed_rankings:
            raise ValueError(
                choice_message.format("feature_ranking", allowed_rankings, feature_ranking)
            )
        if isinstance(feature_subset, int) and feature_subset <= 0:
            raise ValueError("Number of features should be non-negative")
        elif isinstance(feature_subset, float) and not (0.0 <= feature_subset <= 1.0):
            raise ValueError("Proportion of features should be in the interval [0, 1]")
        if not (isinstance(min_leaf_size, int) and min_leaf_size >= 1):
            raise ValueError("min_leaf_size should be a positive integer")
        if n_trees is None:
            pass
        elif isinstance(n_trees, int):
            if n_trees <= 1:
                raise ValueError("If integer, n_trees should be at least 2")
        elif isinstance(n_trees, List):
            if not (n_trees and all(isinstance(x, int) and x >= 1 for x in n_trees)):
                raise ValueError("If list, n_trees should be a list of positive integers.")
        else:
            raise ValueError("n_trees should be int >=2 or List[int] whose elements are >= 1.")

    def _find_temp_dir(self):
        i = 1
        while os.path.exists(ClusModel.TEMP_DIR_PATTERN.format(i)):
            i += 1
        self.temp_dir = ClusModel.TEMP_DIR_PATTERN.format(i)
        return self.temp_dir

    @staticmethod
    def _convert_to_slash(path):
        return re.sub("\\\\", "/", path)

    @staticmethod
    def _get_column_types(
            is_sparse, features: Data, targets: Data,
            hierarchical_attribute: Union[None, List[str]]
    ):
        default_column_type = (ClusModel.COLUMN_TYPE_NUM, [])
        # features
        if is_sparse:
            columns = {}
            for _, i_col, value in zip(*features):
                if i_col not in columns:
                    columns[i_col] = []
                column = columns[i_col]
                if ClusModel._is_numeric(value) and column and ClusModel._is_numeric(column[-1]):
                    continue  # space-efficient: do not need more than one numeric
                else:
                    column.append(value)
            n_features = max(columns) + 1  # some indices might be missing
            # by default numeric, since it does not demand any additional computation
            column_types_xs = [default_column_type for _ in range(n_features)]
            for c in range(n_features):
                values = columns[c]
                if values:
                    column_types_xs[c] = ClusModel._get_column_type(values)
        else:
            n_features = len(features[0])
            column_types_xs = [default_column_type for _ in range(n_features)]
            for c in range(n_features):
                values = [x[c] for x in features]
                column_types_xs[c] = ClusModel._get_column_type(values)
        # targets
        if hierarchical_attribute is None:
            n_targets = len(targets[0])
            column_types_ys = [default_column_type for _ in range(n_targets)]
            for c in range(n_targets):
                values = [y[c] for y in targets]
                column_types_ys[c] = ClusModel._get_column_type(values)
        else:
            column_types_ys = [(ClusModel.COLUMN_TYPE_HIERARCHICAL, hierarchical_attribute)]
        return column_types_xs, column_types_ys

    @staticmethod
    def _create_arff(
            file_name,
            features: Data, targets: Data, is_sparse: bool,
            column_types_xs, column_types_ys
    ):
        """
        Creates a table (with arff header) like this:

        x1, x2 ..., xN, y1, y2, ..., yN

        :param file_name:
        :param features:
        :param targets:
        :param column_types_xs:
        :param column_types_ys:
        :return:
        """
        with open(file_name, "w", newline='') as f:
            # header
            print("@relation strictlyProfessional", file=f)
            for i, column_type in enumerate(column_types_xs):
                print(ClusModel._arff_header_row(i, True, *column_type), file=f)
            for i, column_type in enumerate(column_types_ys):
                print(ClusModel._arff_header_row(i, False, *column_type), file=f)
            print("@data", file=f)
            # data
            if is_sparse:
                ClusModel._create_arff_sparse(features, targets, len(column_types_xs), f)
            else:
                ClusModel._create_arff_dense(features, targets, f)

    @staticmethod
    def _arff_header_row(i, is_feature, c_type, c_values):
        column_name = f"feature{i + 1}" if is_feature else f"target{i + 1}"
        if c_type == ClusModel.COLUMN_TYPE_NUM:
            values = "numeric"
        elif c_type == ClusModel.COLUMN_TYPE_NOM:
            values = f"{{{','.join(c_values)}}}"
        elif c_type == ClusModel.COLUMN_TYPE_HIERARCHICAL:
            values = f"{ClusModel.COLUMN_TYPE_HIERARCHICAL} {','.join(c_values)}"
        else:
            raise ValueError(f"Unknown column type {c_type}")
        return f"@attribute {column_name} {values}"

    @staticmethod
    def _create_arff_sparse(xs, ys, n_features, file_handle):
        examples = {}
        for row, col, val in zip(*xs):
            pair = (col, val)
            if row not in examples:
                examples[row] = [pair]
            else:
                examples[row].append(pair)
        n_examples = len(ys)
        n_examples_xs = ClusModel._get_n_examples(True, xs)
        if ClusModel._get_n_examples(True, xs) > n_examples:
            # <= is ok though
            raise ValueError(f"{n_examples} examples in ys, but more ({n_examples_xs}) in xs!")
        for i, y in enumerate(ys):
            if i in examples:
                example = examples[i]
            else:
                example = []
            parts = [f"{col + 1} {val}" for col, val in example]
            parts += [f"{n_features + col + 1} {val}" for col, val in enumerate(y)]
            print(f"{{{', '.join(parts)}}}", file=file_handle)

    @staticmethod
    def _create_arff_dense(xs, ys, file_handle):
        if len(xs) != len(ys):
            raise ValueError(f"len(features) = {len(xs)} != {len(ys)} = len(targets)")
        for x, y in zip(xs, ys):
            line = ",".join(str(c) for space in [x, y] for c in space)
            print(line, file=file_handle)

    @staticmethod
    def _get_column_type(values: List[Any]) -> Tuple[str, List[Any]]:
        """
        Computes column type and values
        :param values:
        :return:
        """
        possible_values = set()
        for value in values:
            if isinstance(value, str):
                if value != ClusModel.CLUS_MISSING_VALUE:
                    possible_values.add(value.strip())
                else:
                    continue
            elif ClusModel._is_numeric(value):
                return ClusModel.COLUMN_TYPE_NUM, []
            else:
                raise ValueError(f"Unknown column type for values {', '.join(values[:5])} ...")
        return ClusModel.COLUMN_TYPE_NOM, sorted(possible_values)

    @staticmethod
    def _is_numeric(value):
        return isinstance(value, float) or isinstance(value, int)

    @staticmethod
    def _get_n_examples(is_sparse: bool, xs: Data) -> int:
        if is_sparse:
            row_indices = xs[0]
            return 1 + max(row_indices)
        else:
            return len(xs)

    @staticmethod
    def _read_predictions(pred_file: str) -> Dict[str, List[List[Any]]]:
        """
        Loads the predictions from the predictions arff file, and gives predictions.
        """
        raise NotImplementedError("Implement in a subclass!")

    @staticmethod
    def _fit_arg_check_features(xs: RawFeatures) -> Tuple[bool, Data]:
        """
        We allow for three types of input:
        - np.2darray
        - list of examples, which are lists
        - triplet of row indices, column indices, values

        Indices are 0-based.

        :param xs: features
        :return: (is_sparse, xs as List[List[Any]] if dense and xs as the triplet if sparse)
        """
        is_sparse = False
        if isinstance(xs, Tuple):
            if len(xs) != 3:
                raise ValueError(
                    f"If xs is a tuple, it must be of form (row indices, column indices, values), "
                    f"but xs has length {len(xs)}."
                )
            if len(set(map(len, xs))) != 1:
                raise ValueError("All tuple components must have the same length.")
            if len(xs[0]) == 0:
                raise ValueError("There are no examples in the data.")
            is_sparse = True
            xs_final = xs
        elif isinstance(xs, List):
            if len(set(map(len, xs))) != 1:
                raise ValueError("All examples must have the same length.")
            if len(xs) == 0:
                raise ValueError("There are no examples in the data.")
            xs_final = xs
        elif isinstance(xs, np.ndarray):
            if len(xs.shape) != 2:
                raise ValueError(f"Expected 2D array, but your shape is {xs.shape}")
            if xs.shape[0] == 0:
                raise ValueError("There are no examples in the data.")
            xs_final = xs.tolist()
        else:
            raise TypeError(
                f"Unexpected type of the features: {type(xs)}. "
                f"Supported types: "
                f"np.ndarray, List[List[Any]], Tuple[List[int], List[int], List[Any]]"
            )
        return is_sparse, xs_final

    @staticmethod
    def _fit_arg_check_target(ys: RawTarget) -> Data:
        if isinstance(ys, np.ndarray):
            if not (1 <= len(ys.shape) <= 2):
                raise ValueError(f"Expected 1D or 2D target. Got {len(ys.shape)}D.")
            elif ys.shape[0] == 0:
                raise ValueError(f"No target values.")
            ys_final = ys.reshape((ys.shape[0], -1)).tolist()
        elif isinstance(ys, List):
            if len(ys) == 0:
                raise ValueError(f"No target values.")
            if not isinstance(ys[0], List):
                ys_final = [[y] for y in ys]
            else:
                ys_final = ys
        else:
            raise TypeError(
                f"Unexpected type of the targets: {type(ys)}. "
                f"Supported types: np.ndarray, List[List[Any]], List[Any]"
            )
        return ys_final

    def target_type_checker(self, ys: List[List[Any]], **kwargs) -> None:
        raise NotImplementedError("Implement in a subclass")

    def _parse_hmlc_attribute(self, hierarchy):
        pass

    @staticmethod
    def _get_dummy_target(target: Data):
        for t in target:
            if ClusModel.CLUS_MISSING_VALUE not in t:
                return t
        # must not raise an error: data such as
        # y1, y2
        # ? , a
        # ? , b
        # a , ?
        # b , ?
        # are completely valid
        logger.warning("There is no example with all target values present.")
        return target[0]

    def fit(
            self,
            features: RawFeatures, target_raw: RawTarget,
            **kwargs
    ):
        hierarchical = kwargs.get("hierarchy", None)
        hierarchy_shape = kwargs.get("hierarchy_shape", None)
        if hierarchy_shape is not None:
            self._set_param("Hierarchical_Type", hierarchy_shape)
        is_sparse, features = ClusModel._fit_arg_check_features(features)
        target = ClusModel._fit_arg_check_target(target_raw)
        self.target_type_checker(
            target,
            hierarchy=hierarchical, hierarchy_shape=hierarchy_shape
        )
        self.dummy_target = ClusModel._get_dummy_target(target)
        # determine column types
        column_types_xs, column_types_ys = ClusModel._get_column_types(
            is_sparse, features, target, hierarchical
        )
        self.column_types_xs = column_types_xs
        self.column_types_ys = column_types_ys

        temp_dir = self._find_temp_dir()
        models = []
        self.feature_importances_ = []

        n_targets = len(target[0])
        if self.is_mtr:
            self.n_targets_per_experiment = n_targets
            models, scores = self._fit_helper(
                is_sparse, features, target, self.column_types_ys, temp_dir
            )
            self.feature_importances_ = scores
        else:
            self.n_targets_per_experiment = 1
            for i in range(n_targets):
                target_i = [[ys[i]] for ys in target]
                model_i, scores_i = self._fit_helper(
                    is_sparse, features, target_i, [self.column_types_ys[i]], temp_dir
                )
                models.append(model_i)
                self.feature_importances_.append(scores_i)
        self.model = models
        return self

    def _fit_helper(self, is_sparse, features: Data, target: Data, column_types_target, temp_dir):
        """
        :param features: array-like, shape = [n_samples, n_features]
            The features of the learning data.
        :param target: array-like, shape = [n_samples, n_targets]
            The targets of the learning data.
        :param column_types_target: (sublist of) self.column_types_ys
            Self-explanatory
        :param temp_dir: str
            The main temporary directory
        :return:
        """

        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir)  # must not exist yet anyway

        self.perform_experiment(temp_dir, is_sparse, features, target, column_types_target, True)

        # load the model and ranking files
        models = ClusModel.load_model_files(temp_dir, self.get_number_trees(), self.is_ensemble)

        # let's simply find all the fimps (relief follows different nomenclature ...)
        feature_importance_values: List[Tuple[str, Dict[str, np.ndarray]]] = []
        importance_files = [file for file in os.listdir(temp_dir) if file.endswith(".fimp")]
        # file name should be of form <ClusModel.EXPERIMENT_NAME><something>.fimp
        # only for ensembles, we have
        # f"{ClusModel.EXPERIMENT_NAME}Trees{self.get_number_trees()}{score}.fimp"
        for importance_file in importance_files:
            pattern = f"({ClusModel.EXPERIMENT_NAME}(.*))\\.fimp"
            match = re.search(pattern, importance_file)
            if match is None:
                logger.warning(f"Unexpected feature importance file name ('{importance_file}')")
                model_name = importance_file[:importance_file.rfind(".")]
            else:
                model_name = match.group(2)
                if not model_name:
                    # empty appendix
                    model_name = match.group(1)
            path = os.path.join(temp_dir, importance_file)
            fimp = FeatureImportanceFile(path)
            feature_importance_values.append((model_name, fimp.get_importance_scores()))

        shutil.rmtree(temp_dir)
        return models, feature_importance_values

    def perform_experiment(
            self, temp_dir, is_sparse, features: Data, target: Data, column_types_target, is_fit
    ):
        logger.info(f"Performing clus experiment in {temp_dir}")
        ClusModel._create_arff(
            os.path.join(temp_dir, ClusModel.DATA_FILE), features, target, is_sparse,
            self.column_types_xs, column_types_target
        )
        n_features = len(self.column_types_xs)
        self._set_other("Data_File", os.path.join(temp_dir, ClusModel.DATA_FILE))
        if "Descriptive" not in self.method_parameters['Attributes']:
            self._set_other("Attributes_Descriptive", f"1-{n_features}")
        if "Target" not in self.method_parameters['Attributes']:
            self._set_other(
                "Attributes_Target",
                f"{n_features + 1}-{n_features + self.n_targets_per_experiment}")
        self._set_other("Ensemble_PrintAllModelFiles", "Yes" if is_fit else "No")
        self._set_other(
            "Output_WritePredictions",
            "[Test]" if is_fit else "[Train]"  # yes, this is correct
        )
        self._set_other("Output_TrainErrors", "No" if is_fit else "Yes")
        self._set_other("Model_LoadFromModelFile", "No" if is_fit else "Yes")
        self._set_other(
            "Output_WritePerBagModelFile",
            "Yes" if (self.is_ensemble and is_fit) else "No"
        )
        self._set_other(
            "Output_WriteModelFile",
            "Yes" if (not self.is_ensemble and is_fit) else "No"
        )
        current_bag_selection = self._get_bag_selection()
        if not is_fit:
            self._set_other("Ensemble_BagSelection", "0")
            # otherwise, we must leave it as it is ...
        s_file = os.path.join(temp_dir, ClusModel.EXPERIMENT_NAME + ".s")
        self._params_to_s_file(s_file)
        # reverse the necessary settings modification
        if current_bag_selection is not None:
            self._set_other("Ensemble_BagSelection", current_bag_selection)
        # run the experiment
        clus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clus.jar")
        java = " " + self.java_parameters if self.java_parameters else ""
        cmd_directives = " ".join(
            f"-{key} {value}" for key, value in self.command_line_switches.items()
        )
        if cmd_directives:
            cmd_directives += " "
        command = f"java{java} -jar {clus} {cmd_directives}{s_file}"
        command = re.sub(" +", " ", command).split(" ")
        logger.info(f"Clus start: calling {command}")
        p = subprocess.Popen(
            command, stdout=stdout, stderr=stderr
        )  # shell=True,
        p.communicate()
        exit_code = p.returncode
        if exit_code != 0:
            raise RuntimeError(
                f"The experiment with command {command} did not finish successfully and "
                f"exited with code {exit_code}. See the JAVA error message above."
            )
        logger.info("Clus end")

    def get_number_trees(self):
        candidates = self.method_parameters["Ensemble"]["Iterations"]
        if isinstance(candidates, int):
            return candidates
        else:
            return max(candidates)

    def _get_bag_selection(self):
        ensemble_section = "Ensemble"
        bag_selection = "BagSelection"
        if ensemble_section in self.method_parameters:
            if bag_selection in self.method_parameters[ensemble_section]:
                return self.method_parameters[ensemble_section][bag_selection]
        return None

    def predict(self, features: RawFeatures) -> Dict[str, List[List[Any]]]:
        cannot_predict = [
            CommandLineSwitches.SWITCH_KNN,
            CommandLineSwitches.SWITCH_RELIEF
        ]
        if self.model is None:
            raise RuntimeError("Cannot predict if there is no model. Fit it first.")
        elif any(switch in cannot_predict for switch in self.command_line_switches):
            raise RuntimeError(
                f"Predict is not supported if any of {cannot_predict} switches is present"
            )
        is_sparse, features = ClusModel._fit_arg_check_features(features)
        # create bag model files
        temp_dir = self._find_temp_dir()
        os.makedirs(temp_dir)
        if CommandLineSwitches.SWITCH_FOREST in self.command_line_switches:
            model_files = ClusModel.bag_model_file_names(temp_dir, self.get_number_trees())
        else:
            # a single model file
            model_files = [ClusModel.single_tree_model_file(temp_dir)]
        predictions = []
        if self.is_mtr:
            all_models = [self.model]
            column_types_ys = [self.column_types_ys]
            dummy_targets = [self.dummy_target]
        else:
            all_models = self.model
            column_types_ys = [[c_type] for c_type in self.column_types_ys]
            dummy_targets = [[d_target] for d_target in self.dummy_target]
        for a_model, c_types_ys, d_target in zip(all_models, column_types_ys, dummy_targets):
            ClusModel.save_model_files(model_files, a_model)
            # create an experiment ...
            n_examples = ClusModel._get_n_examples(is_sparse, features)
            dummy_target = [d_target for _ in range(n_examples)]
            self.perform_experiment(temp_dir, is_sparse, features, dummy_target, c_types_ys, False)
            # read predictions ...
            p_file = os.path.join(temp_dir, f"{ClusModel.EXPERIMENT_NAME}.train.1.pred.arff")
            predictions.append(self._read_predictions(p_file))
        shutil.rmtree(temp_dir)
        return ClusModel._concatenate_predictions(predictions)

    @staticmethod
    def _concatenate_predictions(predictions_per_target: List[Dict[str, List[List[Any]]]]):
        final_predictions = predictions_per_target[0]
        # concatenate rest
        for i in range(1, len(predictions_per_target)):
            for model, predictions_model in predictions_per_target[i].items():
                final_predictions_model = final_predictions[model]
                for j in range(len(final_predictions_model)):
                    final_predictions_model[j] += predictions_model[j]
        return final_predictions

    def get_params(self, deep=True):
        return self.method_parameters

    def _set_ensemble_method(self, v):
        self._set_other("Ensemble_EnsembleMethod", v)

    def _set_feature_subset(self, feature_subset):
        self._set_other("Ensemble_SelectRandomSubspaces", feature_subset)

    def _set_min_leaf_size(self, min_leaf_size):
        self._set_other("Model_MinimalWeight", min_leaf_size)

    def _set_n_trees(self, n_trees):
        self._set_other("Ensemble_Iterations", n_trees)

    def _set_split_heuristic(self, split_heuristic):
        self._set_other("Tree_Heuristic", split_heuristic)

    def _set_ros_subspace_size(self, size):
        self._set_other("Ensemble_ROSTargetSubspaceSize", size)

    def _set_ros_algorithm_type(self, t):
        self._set_other("Ensemble_ROSAlgorithmType", t)

    def _set_feature_ranking(self, r):
        self._set_other("Ensemble_FeatureRanking", r)

    def _set_is_mtr(self, v):
        self.is_mtr = v

    def _set_verbose(self, v):
        self._set_other("General_Verbose", v)

    def _set_random_state(self, v):
        self._set_other("General_RandomSeed", v)

    def _set_java_parameters(self, java_parameters):
        params = java_parameters.strip()
        add_opens = "--add-opens"
        if add_opens in params and ClusModel.JAVA_OPEN not in params:
            # wrong specification
            raise ValueError(
                f"If {add_opens} is specified, it should equal {ClusModel.JAVA_OPEN}"
            )
        elif add_opens not in params and add_opens in ClusModel.JAVA_OPEN:
            # forgotten specification --> let's add it
            params += f" {ClusModel.JAVA_OPEN}"
        self.java_parameters = re.sub(" +", " ", params).strip()

    def _set_other(self, key, value):
        underscore = "_"
        n_underscores = key.count(underscore)
        if n_underscores == 0:
            # command line switch
            if key in ClusModel.ALLOWED_SWITCHES and isinstance(value, List):
                self.command_line_switches[key] = " ".join(value)
            else:
                raise ValueError(
                    f"Command line switches should be passed as name=value, "
                    f"where value is the list of args for the switch (possibly empty). "
                    f"Allowed switches: {ClusModel.ALLOWED_SWITCHES}"
                )
        elif n_underscores == 1:
            # Section_Parameter
            section, parameter = key.split("_")
            if section not in ClusModel.ALLOWED_SECTIONS:
                raise ValueError(
                    f"Unknown section {section}. The allowed ones: {ClusModel.ALLOWED_SECTIONS}."
                )
            if section not in self.method_parameters:
                self.method_parameters[section] = {}
            self.method_parameters[section][parameter] = value
        else:
            raise ValueError(
                f"kwarg should contain no underscores (command line switch) or "
                f"one one underscore (Section_Parameter), "
                f"but yours ('{key}') contains {n_underscores}."
            )

    def _set_param(self, key, value):
        if key == ClusModel.ENSEMBLE_METHOD:
            self._set_ensemble_method(value)
        elif key == ClusModel.FEATURE_SUBSET:
            self._set_feature_subset(value)
        elif key == ClusModel.MIN_LEAF_SIZE:
            self._set_min_leaf_size(value)
        elif key == ClusModel.N_TREES:
            self._set_n_trees(value)
        elif key == ClusModel.FEATURE_RANKING:
            self._set_feature_ranking(value)
        elif key == ClusModel.MTR:
            self._set_is_mtr(value)
        elif key == ClusModel.VERBOSITY:
            self._set_verbose(value)
        elif key == ClusModel.RANDOM_STATE:
            self._set_random_state(value)
        elif key == ClusModel.JAVA_PARAMETERS:
            self._set_java_parameters(value)
        else:
            self._set_other(key, value)

    def set_params(self, **params):
        for key, value in params.items():
            self._set_param(key, value)
        return self

    def set_params_from_triplets(self, triplets):
        for section, setting, value in triplets:
            if section not in self.method_parameters:
                self.method_parameters[section] = {}
            self.method_parameters[section][setting] = value

    def _params_to_s_file(self, temp_s):
        with open(temp_s, "w") as f:
            for section in self.method_parameters:
                print("[{}]".format(section), file=f)
                for k, v in self.method_parameters[section].items():
                    print("{} = {}".format(k, v), file=f)
                print("", file=f)

    @staticmethod
    def bag_model_file_names(directory, trees):
        basic_name = os.path.join(directory, ClusModel.EXPERIMENT_NAME)
        model_files = [f"{basic_name}_bag{i}.model" for i in range(1, trees + 1)]
        return model_files

    @staticmethod
    def single_tree_model_file(directory):
        return os.path.join(directory, f"{ClusModel.EXPERIMENT_NAME}.model")

    @staticmethod
    def load_model_files(directory, trees, use_ensemble):
        if use_ensemble:
            model_files = ClusModel.bag_model_file_names(directory, trees)
        else:
            model_files = [ClusModel.single_tree_model_file(directory)]
        models = []
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    tree = f.read()
            else:
                logger.info(f"Tree in {model_file} will be skipped.")
                tree = b""
            models.append(tree)
        return models

    @staticmethod
    def save_model_files(model_files, models):
        if len(model_files) != len(models):
            raise ValueError(f"There should be the same number of model files ({len(model_files)}) "
                             f"and models ({len(models)}).")
        for model_file, tree in zip(model_files, models):
            if tree:
                with open(model_file, "wb") as f:
                    f.write(tree)
            else:
                logger.info(f"Model in {model_file} will be skipped.")


class FeatureImportanceFile:
    def __init__(self, f_name):
        self.f_name = f_name
        self.table = []  # [[dataset index, name, ranks, importance], ...]
        self.attrs = {}  # {name: [dataset index, ranks, importance], ...}
        self.ranking_names = None
        self._load_file()
        self._sort_by_attr_index()

    def _load_file(self):
        header = None
        with open(self.f_name) as f:
            for x in f:
                if x.startswith("---------"):
                    break
                else:
                    header = x.strip()
            for feat_ind, x in enumerate(f):
                ind, name, ranks, importance = x.strip().split("\t")
                ind = int(ind)
                ranks = eval(ranks)
                importance = eval(importance, {"Infinity": float("inf"), "NaN": float("nan"), "__builtins__": {}})
                self.attrs[name] = [ind, ranks, importance]
                self.table.append([ind, name, ranks, importance])
        _, _, _, ranking_names_str = header.split("\t")
        if not (ranking_names_str.startswith("[") and ranking_names_str.endswith("]")):
            raise ValueError(f"Unexpected ranking names: {ranking_names_str}")
        self.ranking_names = [name.strip() for name in ranking_names_str[1:-1].split(",")]

    def _sort_by_attr_index(self):
        self.table.sort(key=lambda row: row[0])

    def get_importance_scores(self, ranking_index=None):
        if ranking_index is None:
            # get all rankings
            ranking_indices = list(range(len(self.table[0][-1])))
        else:
            # get the chosen one (like Ginny W.)
            ranking_indices = [ranking_index]
        scores = {}
        for i in ranking_indices:
            ranking_name = self.ranking_names[i]
            scores_i = [row[-1][i] for row in self.table]
            scores[ranking_name] = np.array(scores_i)
        return scores
