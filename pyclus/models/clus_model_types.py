import re
from typing import Dict, List, Any, Tuple, Union
from .clus import ClusModel, RawFeatures, RawTarget, CommandLineSwitches

# CLASSES FOR EACH TASKS


class ClusModelClassification(ClusModel):
    @staticmethod
    def _process_classification_predictions_header(attributes: List[str]):
        targets: List[Tuple[str, List[str]]] = []  # (name, possible values)
        n_keys = 0
        model_names = []
        attribute_names_to_index = {}
        found_all_targets = False
        mega_space = 5 * " "  # we assume 5 spaces (there are many more)
        for line in attributes:
            if mega_space not in line:
                raise ValueError(f"Where is mega space in {line}?")
            attribute_name = line[:line.find(mega_space)]
            attribute_names_to_index[attribute_name] = len(attribute_names_to_index)
            if line.lower().endswith("key"):
                n_keys += 1
            elif line.lower().endswith("}") and not found_all_targets:
                possible_values = [
                    value.strip() for value in line[line.rfind("{") + 1: -1].split(",")
                ]
                targets.append((attribute_name, possible_values))
            elif line.lower().endswith("string"):
                model_names.append(re.search("(.+)-models", line).group(1).strip())
            elif not found_all_targets and line.lower().endswith("numeric"):
                # we have seen true values and predictions of the first model at this point
                found_all_targets = True
                if len(targets) % 2 != 0:
                    raise ValueError(
                        f"Expected even number of targets: {targets} before first numeric."
                    )
                targets = targets[:len(targets) // 2]
        # [indices for model1, ...] where first element looks like
        # [indices for model1 target1, ... ] where first element looks like
        # (index of prediction, {target value1: index of prob(value1), ...})
        n_targets = len(targets)
        ok_indices: List[List[Tuple[int, Dict[str, int]]]] = [
            [(n_keys + i, {}) for i in range(n_targets)]
        ]
        for model_name in model_names:
            model_indices = []
            for target_name, target_values in targets:
                attribute_name = f"{model_name}-p-{target_name}"
                i_model_target = attribute_names_to_index[attribute_name]
                i_model_target_values = {
                    val: attribute_names_to_index[f"{attribute_name}-{val}"]
                    for val in target_values
                }
                model_indices.append((i_model_target, i_model_target_values))
            ok_indices.append(model_indices)
        model_names = [ClusModel.TRUE_VALUES_MODEL] + model_names
        return model_names, ok_indices

    def _read_predictions(self, pred_file: str) -> Dict[str, List[List[Any]]]:
        """
        Reads classification predictions.

        :param pred_file:
        :return:
        """
        attribute = "@attribute"
        attributes = []
        with open(pred_file) as f:
            for _ in range(2):  # @relation, empty line
                f.readline()
            for line in f:
                line = line.strip()
                if line:
                    assert line.lower().startswith(attribute), line
                    attributes.append(line[len(attribute):].strip())
                else:
                    break
            model_names, ok_indices = ClusModelClassification.\
                _process_classification_predictions_header(attributes)
            assert len(model_names) == len(ok_indices)
            data_line = f.readline().lower()
            if not data_line.startswith("@data"):
                raise ValueError(f"The line {data_line} of {pred_file} should start by @data")
            results: Dict[str, List[List[Tuple[str, float]]]] = {
                model_name: [] for model_name in model_names
            }
            for line in f:
                line = line.strip().split(",")
                if not line:
                    # ignore empty lines
                    continue
                for model_name, model_indices in zip(model_names, ok_indices):
                    results_example = []
                    for i_target, i_target_values in model_indices:
                        value_target = line[i_target].strip()
                        if model_name == ClusModel.TRUE_VALUES_MODEL:
                            probability = 1.0
                        else:
                            i_value = i_target_values[value_target]
                            probability = float(line[i_value])
                        results_example.append((value_target, probability))
                    results[model_name].append(results_example)
        return results

    def target_type_checker(self, ys: List[List[Any]], **kwargs):
        if any(not isinstance(value, str) for y in ys for value in y):
            raise ValueError("All target values must be strings in classification!")


class ClusModelRegression(ClusModel):
    @staticmethod
    def _parse_float(f: str):
        if f == ClusModel.CLUS_MISSING_VALUE:
            return f
        else:
            return float(f)

    def _read_predictions(self, pred_file: str) -> Dict[str, List[List[Any]]]:
        """
        Loads the predictions from the predictions arff file for regression.
        See resources for the shape

        :param pred_file:
        :return: a dictionary of pairs model name: predictions, where predictions are 2D array,
        whose [i, j]-th value belongs to the j-th target of the i-th instance.
        """
        attributes = []  # all lines that start with attribute ...
        nb_targets = None
        n_keys = 0
        models = [ClusModel.TRUE_VALUES_MODEL]
        results = {ClusModel.TRUE_VALUES_MODEL: []}
        with open(pred_file) as f:
            for _ in range(2):  # @relation, empty line
                f.readline()
            for line in f:
                line = line.strip()
                if line.lower().endswith("key"):
                    n_keys += 1
                elif line.lower().endswith("string"):
                    if nb_targets is None:
                        nb_targets = (len(attributes) - n_keys) // 2
                    models.append(re.search("@ATTRIBUTE (.+)-models", line).group(1))
                    results[models[-1]] = []
                elif nb_targets is None:
                    attributes.append(re.search("@ATTRIBUTE ([^ ]+) ", line).group(1))
                elif not line:
                    break
            nb_models = len(models) - 1
            data_line = f.readline().lower()
            if not data_line.startswith("@data"):
                raise ValueError(f"The line {data_line} of {pred_file} should start by @data")
            ok_indices = [list(range(n_keys, n_keys + nb_targets))]
            ok_indices += [
                [m * (nb_targets + 1) + t + (n_keys + nb_targets) for t in range(nb_targets)]
                for m in range(nb_models)
            ]
            for line in f:
                line = line.strip().split(",")
                if not line:
                    # ignore empty lines
                    continue
                line = [
                    [ClusModelRegression._parse_float(line[i]) for i in package]
                    for package in ok_indices
                ]
                assert len(line) == len(models)
                for model, values in zip(models, line):
                    results[model].append(values)
        return results

    def target_type_checker(self, ys: List[List[Any]], **kwargs):
        for y in ys:
            for value in y:
                if value == ClusModel.CLUS_MISSING_VALUE:
                    continue
                try:
                    c = value > 0.0 or value <= 0
                except TypeError:
                    c = False
                if not c:
                    raise ValueError(
                        f"All target values must be numeric "
                        f"(or missing ('{ClusModel.CLUS_MISSING_VALUE}')"
                    )


class ClusModelMultiLabelClassification(ClusModelClassification):
    def target_type_checker(self, ys: List[List[Any]], **kwargs):
        allowed = ["0", "1", ClusModel.CLUS_MISSING_VALUE]
        for y in ys:
            for value in y:
                if not (isinstance(value, str) and value in allowed):
                    raise ValueError(
                        f"All target values must be strings ('0', '1') "
                        f"or missing ('{ClusModel.CLUS_MISSING_VALUE}')."
                    )


class ClusModelHierarchicalMultiLabelClassification(ClusModel):
    def _read_predictions(self, pred_file: str) -> Dict[str, List[List[Any]]]:
        """
        Format is regression-like, but output classification-like (predictions and probabilities).
        :param pred_file:
        :return:
        """
        attribute = "@attribute"
        model_names = []
        n_targets = 0
        n_keys = 0
        found_all_targets = False
        with open(pred_file) as f:
            for _ in range(2):  # @relation, empty line
                f.readline()
            hmlc_target_value = f.readline().strip()
            assert hmlc_target_value.endswith("string"), hmlc_target_value
            for line in f:
                line = line.strip()
                if line:
                    assert line.lower().startswith(attribute), line
                    if line.lower().startswith("key"):
                        n_keys += 1
                    elif line.lower().endswith("numeric"):
                        found_all_targets = True
                    elif line.lower().endswith("{1,0}") and not found_all_targets:
                        # second condition not necessary but kept for safety ...
                        n_targets += 1
                    elif line.lower().endswith("string"):
                        model_names.append(line[line.find(" "):line.find("-models")].strip())
                    else:
                        raise ValueError(f"Weird line: {line}")
                else:
                    break
            data_line = f.readline().strip()
            if data_line.lower() != "@data":
                raise ValueError(f"Expected @data (or @DATA) but got {data_line}.")
            ok_indices_true_values = range(n_keys, n_keys + n_targets + 1)
            ok_indices = [
                [m * (n_targets + 1) + t + (1 + n_keys + n_targets) for t in range(n_targets)]
                for m in range(len(model_names))
            ]
            results = {model_name: [] for model_name in model_names}
            results[ClusModel.TRUE_VALUES_MODEL] = []
            for line in f:
                line = line.strip().split(",")
                if not line:
                    # ignore empty lines
                    continue
                # true values are nominal ...
                true_values = [line[i] for i in ok_indices_true_values]
                results[ClusModel.TRUE_VALUES_MODEL].append(true_values)
                # predictions are numeric
                for model_name, ok_indices_model in zip(model_names, ok_indices):
                    predictions = [float(line[i]) for i in ok_indices_model]
                    results[model_name].append(predictions)
        return results

    def target_type_checker(self, ys: List[List[Any]], **kwargs) -> None:
        """
        We simply check whether ys are of the shape

        [list of labels for example1, list of labels for example2, ...]

        where lists of labels are of form [label1, label2, ...]

        where labels are strings.
        :param ys:
        :return:
        """
        hierarchy = kwargs["hierarchy"]
        shape = kwargs["hierarchy_shape"]
        allowed_values = {ClusModel.CLUS_MISSING_VALUE}
        # check hierarchy and its shape
        if shape == ClusModel.HMLC_SHAPE_DAG:
            h_message = "Hierarchy should be given as a list of 'parent/child' edges."
            # hierarchy should be a list of pairs parent/child
            for pair in hierarchy:
                if not isinstance(pair, str) or pair.count("/") != 1:
                    raise ValueError(h_message)
                parent_child = set(pair.split("/"))
                allowed_values |= parent_child
        elif shape == ClusModel.HMLC_SHAPE_TREE:
            # almost nothing to check ...
            h_message = "Hierarchy should be given as a list of paths " \
                        "'root/child1/.../childN' where N >= 0"
            for path in hierarchy:
                if not isinstance(path, str):
                    raise ValueError(h_message)
                allowed_values.add(path)
        else:
            raise ValueError(f"Shape '{shape}' is not an element of {ClusModel.HMLC_SHAPES}")
        # check target values
        message = "Labels for a given example should be given as " \
                  "'label1@label2@...@labelN' or ['label1@label2@...@labelN']"
        for y in ys:
            if len(y) != 1:
                raise ValueError(message)
            values = y[0]
            if not values:
                # no labels
                continue
            elif not isinstance(values, str):
                raise ValueError(message)
            for value in values.split(ClusModel.HMLC_LABEL_SEPARATOR):
                if value not in allowed_values:
                    raise ValueError(f"Label {value} not in allowed values {allowed_values}.")

    def fit(
            self,
            features: RawFeatures, target_raw: RawTarget,
            hierarchy_shape: str = None, hierarchy: List[str] = None, **kwargs
    ):
        if hierarchy_shape is None or hierarchy is None:
            raise ValueError("Provide hierarchy and its shape.")
        return super().fit(
            features, target_raw, hierarchy=hierarchy, hierarchy_shape=hierarchy_shape
        )


class ClusModelTree(ClusModel):
    def __init__(
            self,
            min_leaf_size=1,
            verbose=1,
            random_state: Union[None, int] = None,
            is_multi_target=True,
            java_parameters="",
            **kwargs):
        super().__init__(
            min_leaf_size=min_leaf_size, verbose=verbose, random_state=random_state,
            is_multi_target=is_multi_target, java_parameters=java_parameters,
            **kwargs
        )
        self.tree_parameter_check(**kwargs)

    def _read_predictions(self, pred_file: str) -> Dict[str, List[List[Any]]]:
        raise NotImplementedError("!")

    def target_type_checker(self, ys: List[List[Any]], **kwargs) -> None:
        raise NotImplementedError("!")

    def tree_parameter_check(self, **kwargs):
        forbidden_kwargs = [
            ClusModel.N_TREES, ClusModel.FEATURE_SUBSET,
            ClusModel.FEATURE_SUBSET, ClusModel.FEATURE_RANKING
        ]
        forbidden_cmd = [CommandLineSwitches.SWITCH_FOREST]
        for param in forbidden_kwargs:
            if param in kwargs:
                raise ValueError(f"Do not specify {param} when growing a tree!")
        for param in forbidden_cmd:
            if param in self.command_line_switches:
                raise ValueError(f"Do not specify {param} when growing a tree!")


class ClusModelEnsemble(ClusModel):
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
            java_parameters="",
            **kwargs):
        super().__init__(
            ensemble_method=ensemble_method, feature_subset=feature_subset,
            min_leaf_size=min_leaf_size, n_trees=n_trees,
            verbose=verbose, random_state=random_state,
            feature_ranking=feature_ranking,
            is_multi_target=is_multi_target, java_parameters=java_parameters,
            **kwargs
        )
        self.ensemble_parameter_check(n_trees=n_trees, **kwargs)

    def _read_predictions(self, pred_file: str) -> Dict[str, List[List[Any]]]:
        raise NotImplementedError("!")

    def target_type_checker(self, ys: List[List[Any]], **kwargs) -> None:
        raise NotImplementedError("!")

    def ensemble_parameter_check(self, **kwargs):
        if CommandLineSwitches.SWITCH_FOREST not in self.command_line_switches:
            self.command_line_switches[CommandLineSwitches.SWITCH_FOREST] = ""
        if ClusModel.N_TREES in kwargs:
            value = kwargs[ClusModel.N_TREES]
            if value is None:
                self._set_n_trees(ClusModel.N_TREES_DEFAULT)
            elif isinstance(value, int):
                if value <= 1:
                    raise ValueError("Number of trees must be > 1!")
            # other checks (for clus not to fail done in ClusModel)


class ClusModelRelief(ClusModel):
    def __init__(
            self,
            neighbours=10,
            iterations=-1,
            weight_neighbours=False,
            verbose=1,
            random_state: Union[None, int] = None,
            is_multi_target=True,
            java_parameters="",
            **kwargs):
        if CommandLineSwitches.SWITCH_RELIEF not in kwargs:
            kwargs[CommandLineSwitches.SWITCH_RELIEF] = []
        super().__init__(
            verbose=verbose, random_state=random_state, is_multi_target=is_multi_target,
            java_parameters=java_parameters,
            Relief_Neighbours=neighbours,
            Relief_Iterations=iterations,
            Relief_WeightNeighbours=weight_neighbours,
            **kwargs
        )
        self.relief_parameter_check(**kwargs)

    def _read_predictions(self, pred_file: str) -> Dict[str, List[List[Any]]]:
        raise NotImplementedError("Relief is not a predictive model!")

    def target_type_checker(self, ys: List[List[Any]], **kwargs) -> None:
        raise NotImplementedError("!")

    def relief_parameter_check(self, **kwargs):
        forbidden_kwargs = [
            ClusModel.N_TREES, ClusModel.ENSEMBLE_METHOD,
            ClusModel.FEATURE_RANKING, ClusModel.FEATURE_SUBSET
        ]
        allowed_cmd = [CommandLineSwitches.SWITCH_RELIEF, CommandLineSwitches.SWITCH_SILENT]
        for param in forbidden_kwargs:
            if param in kwargs:
                raise ValueError(f"Do not specify {param} when running Relief!")
        for param in self.command_line_switches:
            if param not in allowed_cmd:
                raise ValueError(f"Do not specify {param} when running Relief!")
