"""Train and evaluate the challenge-specific conformal ensemble."""
from pathlib import Path
import json

from constants.scania_component_x_columns import VEHICLE_ID
from models.ConformalCoTrainingEnsembleScaniaChallenge import ConformalCoTrainingEnsembleScaniaChallenge
from scania.challenge import CLASS_LABEL, get_challenge_features, load_challenge_labels, run_challenge_evaluation
from scania.dataset import ScaniaDataModule
from shared.co_training.utils_cotraining_common import parse_models_config
from shared.utils import set_seed


def _challenge_data(module, split):
    features, ids = get_challenge_features(module, split)
    labels = load_challenge_labels(module.data_dir, split)
    if labels[VEHICLE_ID].duplicated().any():
        raise ValueError(f'Duplicate challenge labels in {split}')
    aligned = labels.set_index(VEHICLE_ID)[CLASS_LABEL].reindex(ids)
    if aligned.isna().any():
        raise ValueError(f'Missing challenge labels in {split}')
    return features, aligned.to_numpy()


def train_model(dataset_root, dataset_params, model_params, training_params,
                checkpoints_path, results_path):
    dataset_params = dict(dataset_params)
    if not dataset_params.get('cache_dir'):
        dataset_params['cache_dir'] = str(Path(dataset_root) / 'scania_challenge_cache')
    if dataset_params.get('test_rate', 0.0) != 0.0:
        raise ValueError('The Scania challenge workflow requires test_rate=0.0')
    dataset_params['test_rate'] = 0.0
    training_params = dict(training_params)
    set_seed(dataset_params['seed'])
    checkpoints_path = Path(checkpoints_path)
    results_path = Path(results_path)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    module = ScaniaDataModule(data_dir=dataset_root, **dataset_params)
    module.setup()
    failure_x, failure_y, censored_x, censored_ids = module.get_cotraining_tensors('train')
    val_x, val_y, _, _ = module.get_cotraining_tensors('val')
    calib_x, calib_y = (None, None)
    if module.calib_rate > 0:
        calib_x, calib_y, _, _ = module.get_cotraining_tensors('calib')
        if len(calib_x) == 0:
            raise ValueError('The calibration split contains no uncensored windows; increase the data fraction or calibration rate.')
    if len(val_x) == 0:
        raise ValueError('The validation split contains no uncensored windows; increase the data fraction or validation rate.')
    _, _, lower = module.get_censored_lower_bounds('train')
    times = None
    if training_params.get('isotonic_time_weighting', False):
        _, _, times = module.get_censored_time_steps('train')
    validation = _challenge_data(module, 'validation')
    test = _challenge_data(module, 'test')
    models, builders, meta = parse_models_config(model_params['models'], len(module.feature_cols),
                                                  dataset_params['sequence_len'], failure_y)
    iterations = training_params.pop('iterations')
    pool = training_params.pop('suspension_pool_size')
    ratio = training_params.pop('add_ratio')
    ensemble = ConformalCoTrainingEnsembleScaniaChallenge(models=models, verbose=1, **training_params)
    ensemble.setup_training_builder(module_builders=builders, max_epochs=meta['max_epochs'],
                                    patiences=meta['patiences'],
                                    batchs_size=[dataset_params['batch_size']] * len(models),
                                    shuffle_dataloaders=[True] * len(models), gpu_ids=None)
    ensemble.configure_challenge(*validation, *test, checkpoints_path, meta['version_strs'])
    with (results_path / 'parameters.json').open('w', encoding='utf-8') as stream:
        json.dump({'dataset_params': dataset_params, 'model_params': model_params,
                   'training_params': training_params}, stream, indent=2)
    individual, weighted, weighted_weights = ensemble.train(
        train_with_censored_data=False, iterations=iterations, suspension_pool_size=pool, add_ratio=ratio,
        failure_data=failure_x, failure_label=failure_y, suspension_data=censored_x,
        suspension_ids=censored_ids, val_data=val_x, val_label=val_y,
        calib_data=calib_x, calib_label=calib_y, suspension_lower_bounds=lower,
        suspension_time_steps=times, metrics_file=str(results_path / 'per_stage.csv'),
        log_file=str(results_path / 'log.txt'), pool_seed=dataset_params['seed'])
    individual_weights = ensemble.calculate_weights(*validation, models=individual)

    def predict(features):
        a = [ensemble._predict(m, features) for m in individual]
        b = [ensemble._predict(m, features) for m in weighted]
        result = {f'individual_{i}_{name}': prediction for i, (name, prediction) in
                  enumerate(zip(meta['version_strs'], a))}
        result['individual_weighted'] = sum(w * p for w, p in zip(individual_weights, a))
        result['best_weighted'] = sum(w * p for w, p in zip(weighted_weights, b))
        return result

    for split in ('validation', 'test'):
        result = run_challenge_evaluation(module, predict, 'conformal_co_training_ensemble_challenge',
                                          str(results_path), split=split)
        if result is None:
            raise RuntimeError(f'Challenge evaluation failed for {split}')
    return individual, weighted, weighted_weights
