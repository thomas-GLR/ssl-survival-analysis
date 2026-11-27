"""Run the Scania conformal co-training challenge."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from scania.utils.utils_conformal_cotraining_ensemble_challenge import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config-path', 'checkpoints-path', 'results-path', 'dataset-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cache-dir', default=None,
                        help='Dataset cache directory (defaults to <dataset-root>/scania_challenge_cache).')
    parser.add_argument('--benchmark-version', default='default')
    parser.add_argument('--run-name', default='')
    args = parser.parse_args()
    config_path = Path(args.config_path) / args.benchmark_version / 'conformal_co_training_ensemble_challenge.json'
    with config_path.open(encoding='utf-8') as stream:
        config = json.load(stream)
    if args.cache_dir is not None:
        config['dataset_params']['cache_dir'] = args.cache_dir
    run_name = args.run_name or datetime.now().strftime('challenge_%Y-%m-%d_%H-%M-%S')
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        parser.error('--run-name must be a single directory name')
    train_model(dataset_root=args.dataset_root, checkpoints_path=Path(args.checkpoints_path) / run_name,
                results_path=Path(args.results_path) / run_name, **config)


if __name__ == '__main__':
    main()
