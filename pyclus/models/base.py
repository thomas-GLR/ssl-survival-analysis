import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import logging
from pyclus.helpers import Utilities


logger = logging.getLogger(__name__)


class PredictiveModel:
    TEMP_DIR = Utilities.find_an_available_root_temp_dir()

    def __init__(self, model=None):
        """
        Base predictive model class. When explicitly calling the constructor,
        it accepts any scikit learn predictive model.
        :param model: scikit learn predictive model
        """
        # in addition to scikit learn models, this may store various structures that somehow
        # correspond to a model (depending on the subclass). For example, Gradient Boosting's model
        # is a list of per-target models, whereas ClusForest's model is a list of (lists of) bytes
        # that define the java models.
        self.model = model
        self.descriptive_attribute_names = None  # set after fit
        self.target_attribute_names = None  # set after fit

    def fit(self, features, targets, **kwargs):
        """
        Fit the model to the provided data.
        :param features: array-like, shape = [n_samples, n_features]
            The features of the learning data.
        :param targets: array-like, shape = [n_samples, n_targets]
            The targets of the learning data.
        :return:
        """
        if PredictiveModel.TEMP_DIR in kwargs:
            del kwargs[PredictiveModel.TEMP_DIR]
        self.model.fit(features, targets, **kwargs)

    def set_attribute_names(self, descriptive_attribute_names, target_attribute_names):
        self.descriptive_attribute_names = descriptive_attribute_names[:]
        self.target_attribute_names = target_attribute_names[:]

    def predict(self, features):
        """
        Use the fitted model to make predictions on the provided data.
        :param features: array-like, shape = [n_samples, n_features]
            The features of the data to make predictions from.
        :return: array-like, shape = [n_samples, n_targets]
            Predicted values for the provided examples.
        """
        return self.model.predict(features)

    def pre_predict_check(self, descriptive_attribute_names, target_attribute_names):
        if self.descriptive_attribute_names is None \
                or self.target_attribute_names is None:
            raise ValueError(
                "Descriptive and target attributes should be set! Train (fit) the model first."
            )
        wrong_length_message = "The number of {0} attributes used for prediction ({1}) " \
                               "does not equal the number of {0} attributes " \
                               "used for training the model ({2})"
        wrong_names_message = "The names of {0} attributes used for prediction ({1}) " \
                              "do not equal the names of {0} attributes " \
                              "used for training the model ({2})"
        if len(descriptive_attribute_names) != len(self.descriptive_attribute_names):
            raise ValueError(
                wrong_length_message.format(
                    "descriptive",
                    len(descriptive_attribute_names), len(self.descriptive_attribute_names)
                )
            )
        elif descriptive_attribute_names != self.descriptive_attribute_names:
            logger.warning(
                wrong_names_message.format(
                    "descriptive", descriptive_attribute_names, self.descriptive_attribute_names)
            )
        if len(target_attribute_names) != len(self.target_attribute_names):
            raise ValueError(
                wrong_length_message.format(
                    "target",
                    len(target_attribute_names), len(self.target_attribute_names)
                )
            )
        elif target_attribute_names != self.target_attribute_names:
            logger.warning(
                wrong_names_message.format(
                    "target", target_attribute_names, self.target_attribute_names
                )
            )

    def get_params(self, deep=True):
        """
        Return the parameters of the model.
        :param deep: boolean, (default=True)
            If True, returns also the parameters of any sub-models (e.g., in ensembles).
        :return: dict
            Dictionary of parameters.
        """
        return self.model.get_params(deep)

    def set_params(self, **kwargs):
        """
        Sets the model parameters.
        :param kwargs: dict
            Dictionary with names of parameters and its selected values.
        :return:
        """
        self.model.set_params(**kwargs)

    def get_feature_ranking_scores(self, features, targets, error=mean_squared_error):
        """
        Returns the feature ranking scores. Computes them as follows:
        Tries to access feature_importances_ field of the model first, and return its value.
        If the field does not exist, it resorts to the call of computing permutation_feature_ranking
        scores.

        Depending on the ranking method itself, either a single ranking is returned
        (e.g., in the case of global MTR model), or multiple rankings (one for each target)
        are returned (e.g., in the case of local MTR model).

        :param features: array-like, shape = [n_samples, n_features]
            Passed to permutation_feature_ranking_scores
        :param targets: array-like, shape = [n_samples, n_targets]
            Passed to permutation_feature_ranking_scores
        :param error: callable
            Passed to permutation_feature_ranking_scores
        :return: array-like, shape = [n_rankings, n_features]
        """

        for o in [self, self.model]:
            importance_scores = getattr(o, "feature_importances_", None)
            if importance_scores is not None:
                if len(importance_scores.shape) == 1:
                    importance_scores = importance_scores.reshape((1, -1))
                return importance_scores
        logging.info("No embedded feature importance. Computing permutation scores")
        return self.permutation_feature_ranking_scores(features, targets, performance_measure=error)

    @staticmethod
    def multi_error_score(error, y_true, y_pred):
        try:
            # try first if per-target is possible:
            return error(y_true, y_pred, multioutput="raw_values")
        except ValueError:
            return error(y_true, y_pred)

    @staticmethod
    def breiman_ranking_score(e_permuted: np.ndarray, e_baseline: np.ndarray, more_is_better):
        sign = -1.0 if more_is_better else 1.0
        return sign * (e_permuted - e_baseline) / e_baseline

    @staticmethod
    def is_more_better(error_measure):
        y1 = np.array([1.0, 2.0, 3.0])
        y2 = np.array([1.1, 2.1, 3.1])
        value_optimal = error_measure(y1, y1)
        value_other = error_measure(y1, y2)
        return value_optimal > value_other

    def permutation_feature_ranking_scores(
            self, features, targets, performance_measure=mean_squared_error
    ):
        """
        Calculate the "Random forest" feature importances.
        The importance of a feature is determined by the drop in
        performance that occurs if the values of the features are permuted.
        :param features: array-like, shape = [n_samples, n_features]
            The features of the data to calculate the importances on.
        :param targets: array-like, shape = [n_samples, n_targets]
            The targets of the data to calculate the importances on.
        :param performance_measure: callable with a signature
        (y_true, y_pred) -> Union[np.ndarray, float]
        The performance measure to use. Can be loss (less is better) or quality (more is better).
        :return: array-like, shape = [n_rankings, n_features]
            The differences in performance with and without value permutation for each feature.
        """
        more_is_better = PredictiveModel.is_more_better(performance_measure)
        baseline = PredictiveModel.multi_error_score(
            performance_measure, targets, self.predict(features)
        )  # 1d array
        n_features = features.shape[1]
        n_rankings = baseline.shape[0]
        per_target_scores = np.zeros((n_rankings, n_features))
        permuted = features.copy()
        np.random.seed(404)
        for f in range(n_features):
            permuted[:, f] = np.random.permutation(features[:, f])
            permuted_error = PredictiveModel.multi_error_score(
                performance_measure, targets, self.predict(permuted)
            )
            per_target_scores[:, f] = PredictiveModel.breiman_ranking_score(
                permuted_error, baseline, more_is_better
            )
            permuted[:, f] = features[:, f]
        global_scores = np.mean(per_target_scores, axis=0).reshape((1, -1))
        if n_rankings == 1:
            logger.info("Aggregating one ranking ... Global scores equal to per-target ones.")
        return np.concatenate((global_scores, per_target_scores))

    def save(self, model_file):
        """
        The default method for saving the models to files.
        :param model_file: string
            Path to the file, e.g., '/home/model.bin'
        :return:
        """
        pickle.dump(self, open(model_file, "wb"))

    @staticmethod
    def load(model_file):
        """
        The default loading method for models.
        :param model_file: string
            Path to the file, e.g., '/home/model.bin'
        :return: model object
        """
        return pickle.load(open(model_file, "rb"))
