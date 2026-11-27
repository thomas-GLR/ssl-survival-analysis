"""Challenge-specific scoring and checkpoint selection for co-training."""
import copy
import csv
import math
from pathlib import Path

import numpy as np
import torch

from models.CoTrainingEnsemble_v2 import CoTrainingEnsemble_v2
from scania.metrics import rul_to_class, scania_cost_from_classes
from shared.enums import DatasetVersion


class ConformalCoTrainingEnsembleScaniaChallenge(CoTrainingEnsemble_v2):
    def __init__(self, models, **kwargs):
        fixed = dict(dataset_version=DatasetVersion.SCANIA, use_fine_tuning=False,
                     use_cotraining_ensemble_survival_loss_function=True,
                     peer_weighted_pseudo_label=False, use_average_window_confidence=True,
                     bagging_failure_data=False, keep_best_model_mode=None)
        for key, value in fixed.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"{key} must be {value!r}")
            kwargs[key] = value
        super().__init__(models=models, **kwargs)
        self.best_individual_costs = [math.inf] * len(models)
        self.best_individual_models = [None] * len(models)
        self.best_weighted_cost = math.inf
        self.best_weighted_models = None
        self.best_weighted_weights = None

    @staticmethod
    def weights_from_costs(costs):
        costs = np.asarray(costs, dtype=float)
        if not len(costs) or not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("Challenge costs must be finite and nonnegative")
        zero = costs == 0
        if zero.any():
            return (zero / zero.sum()).tolist()
        inverse = 1 / costs
        return (inverse / inverse.sum()).tolist()

    def calculate_weights(self, challenge_data, challenge_labels, models=None):
        models = self.lightning_modules if models is None else models
        costs = [self._challenge_cost(self._predict(m, challenge_data), challenge_labels) for m in models]
        self.weights = self.weights_from_costs(costs)
        return self.weights

    @staticmethod
    def _challenge_cost(prediction, labels):
        prediction = prediction.detach().cpu().numpy().reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        return float(scania_cost_from_classes(rul_to_class(prediction), labels))

    def configure_challenge(self, validation_data, validation_labels, test_data, test_labels,
                            checkpoints_path, model_names):
        self.challenge_validation = (validation_data, validation_labels)
        self.challenge_test = (test_data, test_labels)
        self.challenge_checkpoints = Path(checkpoints_path)
        self.challenge_model_names = list(model_names)

    def train(self, *args, metrics_file, **kwargs):
        if kwargs.pop('pretrained_models', None) is not None:
            raise ValueError('Pretrained models are not supported')
        if 'test_data' in kwargs or 'test_label' in kwargs:
            raise ValueError('Use configure_challenge for challenge data')
        test_data, test_labels = self.challenge_test
        super().train(*args, test_data=test_data, test_label=torch.as_tensor(test_labels),
                      score_callback=lambda *args: None, weight_callback=lambda *args: None,
                      metrics_file=metrics_file, **kwargs)
        return self.best_individual_models, self.best_weighted_models, self.best_weighted_weights

    def _log_stage_metrics(self, stage, h, metrics_file, iteration_time=None, **kwargs):
        vx, vy = self.challenge_validation
        tx, ty = self.challenge_test
        vp = [self._predict(model, vx) for model in h]
        tp = [self._predict(model, tx) for model in h]
        vc = [self._challenge_cost(p, vy) for p in vp]
        tc = [self._challenge_cost(p, ty) for p in tp]
        weights = self.weights_from_costs(vc)
        weighted_val = self._challenge_cost(sum(w * p for w, p in zip(weights, vp)), vy)
        weighted_test = self._challenge_cost(sum(w * p for w, p in zip(weights, tp)), ty)
        for i, (model, cost) in enumerate(zip(h, vc)):
            if cost < self.best_individual_costs[i]:
                self.best_individual_costs[i] = cost
                self.best_individual_models[i] = copy.deepcopy(model).cpu()
                directory = self.challenge_checkpoints / 'individual' / f'{i}_{self.challenge_model_names[i]}'
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_individual_models[i], directory / f'{self.challenge_model_names[i]}_individual_{cost:.6f}.pth')
        if weighted_val < self.best_weighted_cost:
            self.best_weighted_cost = weighted_val
            self.best_weighted_models = [copy.deepcopy(model).cpu() for model in h]
            self.best_weighted_weights = weights
            directory = self.challenge_checkpoints / 'weighted' / f'{weighted_val:.6f}'
            directory.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(self.best_weighted_models):
                torch.save(model, directory / f'{self.challenge_model_names[i]}_{i}_weighted_{weighted_val:.6f}.pth')
            torch.save(weights, directory / 'weights.pth')
        header = ['stage', 'iteration_time'] + [f'validation_cost_{i}' for i in range(len(h))] + [f'test_cost_{i}' for i in range(len(h))] + ['weighted_validation_cost', 'weighted_test_cost'] + [f'weight_{i}' for i in range(len(h))]
        path = Path(metrics_file)
        exists = path.exists()
        if exists:
            with path.open(newline='', encoding='utf-8') as stream:
                if next(csv.reader(stream), None) != header:
                    raise ValueError(f'Existing metrics file has an incompatible header: {path}')
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(header)
            writer.writerow([stage, iteration_time, *vc, *tc, weighted_val, weighted_test, *weights])
