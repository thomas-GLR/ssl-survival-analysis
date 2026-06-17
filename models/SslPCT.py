from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from pyclus.models import ClusModelEnsemble, ClusModelRegression


class SslPCT(ClusModelRegression, ClusModelEnsemble):
	"""
	Python wrapper for SSL-PCT (Semi-Supervised Learning Predictive Clustering Trees).

	The objective is to reproduce the equivalent of:

	`java -jar ... -forest -ssl settings_FD001.s`

	but with a scikit-learn-like API:

		model = SslPCT(...)
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

	The parameters are exposed through the constructor and can be modified.
	Additional CLUS options can also be passed via `**kwargs`.
	"""

	def __init__(
		self,
		ensemble_method: str = "RForest",
		feature_subset: Any = "SQRT",
		min_leaf_size: int = 1,
		n_trees: int = 100,
		verbose: int = 0,
		random_state: Optional[int] = None,
		split_heuristic: str = "VarianceReduction",
		missing_clustering_attr_handling: str = "EstimateFromParentNode",
		missing_target_attr_handling: str = "ParentNode",
		pruning_method: str = "M5",
		semi_supervised_method: str = "PCT",
		percentage_labeled: int = 100,
		pruning_when_tuning: str = "No",
		internal_folds: int = 3,
		possible_weights: Iterable[float] = (0.25, 0.5, 0.75),
		write_ensemble_predictions: str = "Yes",
		number_of_threads: int = 8,
		is_multi_target: bool = True,
		java_parameters: str = "",
		**kwargs,
	):
		self._ssl_init_params = {
			"ensemble_method": ensemble_method,
			"feature_subset": feature_subset,
			"min_leaf_size": min_leaf_size,
			"n_trees": n_trees,
			"verbose": verbose,
			"random_state": random_state,
			"split_heuristic": split_heuristic,
			"missing_clustering_attr_handling": missing_clustering_attr_handling,
			"missing_target_attr_handling": missing_target_attr_handling,
			"pruning_method": pruning_method,
			"semi_supervised_method": semi_supervised_method,
			"percentage_labeled": percentage_labeled,
			"pruning_when_tuning": pruning_when_tuning,
			"internal_folds": internal_folds,
			"possible_weights": tuple(possible_weights),
			"write_ensemble_predictions": write_ensemble_predictions,
			"number_of_threads": number_of_threads,
			"is_multi_target": is_multi_target,
			"java_parameters": java_parameters,
		}

		clus_kwargs = dict(kwargs)
		clus_kwargs.setdefault("forest", [])
		clus_kwargs.setdefault("ssl", [])
		clus_kwargs.setdefault("Ensemble_WriteEnsemblePredictions", write_ensemble_predictions)
		clus_kwargs.setdefault("Ensemble_NumberOfThreads", number_of_threads)
		clus_kwargs.setdefault("Tree_Heuristic", split_heuristic)
		clus_kwargs.setdefault(
			"Tree_MissingClusteringAttrHandling", missing_clustering_attr_handling
		)
		clus_kwargs.setdefault("Tree_MissingTargetAttrHandling", missing_target_attr_handling)
		clus_kwargs.setdefault("Tree_PruningMethod", pruning_method)
		clus_kwargs.setdefault("SemiSupervised_SemiSupervisedMethod", semi_supervised_method)
		clus_kwargs.setdefault("SemiSupervised_PercentageLabeled", percentage_labeled)
		clus_kwargs.setdefault("SemiSupervised_PruningWhenTuning", pruning_when_tuning)
		clus_kwargs.setdefault("SemiSupervised_InternalFolds", internal_folds)
		clus_kwargs.setdefault("SemiSupervised_PossibleWeights", list(possible_weights))

		# `ClusModelEnsemble` fournit la mécanique CLUS complète et active le mode forest.
		ClusModelEnsemble.__init__(
			self,
			ensemble_method=ensemble_method,
			feature_subset=feature_subset,
			min_leaf_size=min_leaf_size,
			n_trees=n_trees,
			verbose=verbose,
			random_state=random_state,
			is_multi_target=is_multi_target,
			java_parameters=java_parameters,
			**clus_kwargs,
		)

	def fit(self, X, y, **kwargs):
		"""Entraîne le modèle sur `X` et `y` au format scikit-learn."""
		return super().fit(X, y, **kwargs)

	def get_params(self, deep: bool = True) -> Dict[str, Any]:
		"""Retourne les paramètres initiaux du wrapper, comme un estimateur sklearn."""
		return dict(self._ssl_init_params)

	def set_params(self, **params):
		"""Met à jour les paramètres du wrapper et synchronise le modèle CLUS."""
		if not params:
			return self

		self._ssl_init_params.update(params)

		mapping = {
			"ensemble_method": "ensemble_method",
			"feature_subset": "feature_subset",
			"min_leaf_size": "min_leaf_size",
			"n_trees": "n_trees",
			"verbose": "verbose",
			"random_state": "random_state",
			"split_heuristic": "Tree_Heuristic",
			"missing_clustering_attr_handling": "Tree_MissingClusteringAttrHandling",
			"missing_target_attr_handling": "Tree_MissingTargetAttrHandling",
			"pruning_method": "Tree_PruningMethod",
			"semi_supervised_method": "SemiSupervised_SemiSupervisedMethod",
			"percentage_labeled": "SemiSupervised_PercentageLabeled",
			"pruning_when_tuning": "SemiSupervised_PruningWhenTuning",
			"internal_folds": "SemiSupervised_InternalFolds",
			"possible_weights": "SemiSupervised_PossibleWeights",
			"write_ensemble_predictions": "Ensemble_WriteEnsemblePredictions",
			"number_of_threads": "Ensemble_NumberOfThreads",
			"is_multi_target": "is_mtr",
			"java_parameters": "java_parameters",
		}

		for key, value in params.items():
			if key == "possible_weights":
				value = list(value)
			if key in {"ensemble_method", "feature_subset", "min_leaf_size", "n_trees", "verbose", "random_state", "java_parameters", "is_multi_target"}:
				super().set_params(**{mapping[key]: value})
			elif key in mapping:
				super().set_params(**{mapping[key]: value})

		return self

