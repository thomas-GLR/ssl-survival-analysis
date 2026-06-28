from C_MAPSS.utils import utils_cmapss
from C_MAPSS.utils.ModelVersion import ModelVersion

if __name__ == '__main__':
    import argparse

    dataset_root = "../../data/C_MAPSS"
    checkpoints_path = "../checkpoints"
    results_path = "../results"
    config_path = "../config"

    parser = argparse.ArgumentParser(
        description="Reproduce all semi-supervised experiments."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="device type",
    )
    parser.add_argument(
        "--model-version",
        default="lstm",
        choices=[
            "transformer",
            "lstm",
            "autoencoder",
            "metric",
            "rsf",
            "pyclus",
            "coprog",
        ],
        help="transformer or lstm model",
    )
    parser.add_argument(
        "--config-path",
        default=config_path,
        help="path to config",
    )
    parser.add_argument(
        "--checkpoints-path",
        default=checkpoints_path,
        help="path to checkpoints",
    )
    parser.add_argument(
        "--results-path",
        default=results_path,
        help="path to results",
    )
    parser.add_argument(
        "--dataset-root",
        default=dataset_root,
        help="path to dataset",
    )

    opt = parser.parse_args()

    model_version = ModelVersion(opt.model_version)

    utils_cmapss.reproduce_result(
        config_path=opt.config_path,
        checkpoints_path=opt.checkpoints_path,
        results_path=opt.results_path,
        dataset_root=opt.dataset_root,
        model_version=model_version,
        device=opt.device,
    )
