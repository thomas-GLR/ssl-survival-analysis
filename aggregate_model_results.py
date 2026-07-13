"""Aggregate per-run model outputs into a single benchmark summary CSV.

Some model output trees under ``outputs/`` (autoencoder baseline and the v1
co_training_ensemble) were produced without an aggregated summary CSV like the
other benchmarks have. This script walks those trees and rolls the per-run
metrics up into a summary shaped like the existing
``*_benchmark_default_*_results_turbofan.csv`` files.
"""

import glob
import os
import re

import pandas as pd

from constants import results_columns

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

COLUMNS = [
    results_columns.SUB_DATASET,
    results_columns.CENSORED_PERCENTAGE,
    results_columns.BROKEN_PERCENTAGE,
    results_columns.MODEL,
    results_columns.RMSE,
    results_columns.SCORE,
]


def parse_metric(value: str) -> float:
    """Turn a possibly ``tensor(...)``-wrapped string into a float."""
    return float(re.sub(r"tensor\((.*)\)", r"\1", str(value).strip()))


def parse_config(folder_name: str) -> tuple[float, float]:
    """Extract (censored, broken) percentages from ``censored-<c>-broken-<b>``."""
    match = re.match(r"censored-([0-9.]+)-broken-([0-9.]+)", folder_name)
    if not match:
        raise ValueError(f"Unexpected config folder name: {folder_name}")
    return float(match.group(1)), float(match.group(2))


def _sort_key(row: dict):
    """FD ascending; censored descending with 0.0 last; broken ascending with 0.0 last."""
    censored = row[results_columns.CENSORED_PERCENTAGE]
    broken = row[results_columns.BROKEN_PERCENTAGE]
    # 0.0 configs (the uncensored reference run) sort to the bottom of each FD.
    censored_is_zero = censored == 0.0
    broken_key = float("inf") if broken == 0.0 else broken
    return (row[results_columns.SUB_DATASET], censored_is_zero, -censored, broken_key)


def build_summary(model_glob: str, model_value: str, per_run_csv_prefix: str, out_name: str) -> None:
    model_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, model_glob)))
    if not model_dirs:
        raise FileNotFoundError(f"No model folders matched: {model_glob}")

    rows = []
    for model_dir in model_dirs:
        fd_match = re.search(r"turbofan-(FD\d+)", os.path.basename(model_dir))
        if not fd_match:
            raise ValueError(f"Cannot find sub-dataset in folder name: {model_dir}")
        sub_dataset = fd_match.group(1)

        config_dirs = sorted(glob.glob(os.path.join(model_dir, "censored-*-broken-*")))
        for config_dir in config_dirs:
            censored, broken = parse_config(os.path.basename(config_dir))
            per_run_csv = os.path.join(config_dir, f"{per_run_csv_prefix}-turbofan-{sub_dataset}.csv")
            if not os.path.isfile(per_run_csv):
                # Some runs only left a log.txt (no metrics produced); skip them.
                print(f"WARNING: no result CSV, skipping {config_dir}")
                continue

            per_run = pd.read_csv(per_run_csv)
            rows.append({
                results_columns.SUB_DATASET: sub_dataset,
                results_columns.CENSORED_PERCENTAGE: censored,
                results_columns.BROKEN_PERCENTAGE: broken,
                results_columns.MODEL: model_value,
                results_columns.RMSE: parse_metric(per_run["test_rmse"].iloc[0]),
                results_columns.SCORE: parse_metric(per_run["test_score"].iloc[0]),
            })

    rows.sort(key=_sort_key)
    df = pd.DataFrame(rows, columns=COLUMNS)

    out_path = os.path.join(OUTPUTS_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")


def main() -> None:
    build_summary(
        model_glob="model-baseline-with-autoencoder-turbofan-FD*",
        model_value="autoencoder",
        per_run_csv_prefix="autoencoder",
        out_name="autoencoder_benchmark_default_2026-07-07_11-22-16_results_turbofan.csv",
    )
    build_summary(
        model_glob="model-co_training_ensemble-turbofan-FD*",
        model_value="co_training_ensemble",
        per_run_csv_prefix="co_training_ensemble",
        out_name="co_training_ensemble_benchmark_default_2026-07-08_11-02-25_results_turbofan.csv",
    )


if __name__ == "__main__":
    main()
