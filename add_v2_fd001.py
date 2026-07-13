"""Add the missing FD001 rows to the co_training_ensemble_v2 benchmark summary.

The summary CSV was generated with FD002/FD003/FD004 only. The FD001 run lives in
``outputs/model-co_training_ensemble_v2-turbofan-FD001-2026-07-08_17-57-28/``; this
script reads its per-run CSVs and merges the FD001 rows into the existing summary,
keeping the same column order and sub-dataset ordering (FD001 first).
"""

import glob
import os

import pandas as pd

from aggregate_model_results import COLUMNS, OUTPUTS_DIR, _sort_key, parse_config, parse_metric
from constants import results_columns

SUMMARY_CSV = os.path.join(
    OUTPUTS_DIR, "co_training_ensemble_v2_benchmark_default_2026-07-09_10-22-11_results_turbofan.csv"
)
FD001_DIR = os.path.join(OUTPUTS_DIR, "model-co_training_ensemble_v2-turbofan-FD001-2026-07-08_17-57-28")
MODEL_VALUE = "co_training_ensemble_v2"
SUB_DATASET = "FD001"


def build_fd001_rows() -> list[dict]:
    rows = []
    for config_dir in sorted(glob.glob(os.path.join(FD001_DIR, "censored-*-broken-*"))):
        censored, broken = parse_config(os.path.basename(config_dir))
        per_run_csv = os.path.join(config_dir, f"{MODEL_VALUE}-turbofan-{SUB_DATASET}.csv")
        if not os.path.isfile(per_run_csv):
            print(f"WARNING: no result CSV, skipping {config_dir}")
            continue
        per_run = pd.read_csv(per_run_csv)
        rows.append({
            results_columns.SUB_DATASET: SUB_DATASET,
            results_columns.CENSORED_PERCENTAGE: censored,
            results_columns.BROKEN_PERCENTAGE: broken,
            results_columns.MODEL: MODEL_VALUE,
            results_columns.RMSE: parse_metric(per_run["test_rmse"].iloc[0]),
            results_columns.SCORE: parse_metric(per_run["test_score"].iloc[0]),
        })
    return rows


def main() -> None:
    existing = pd.read_csv(SUMMARY_CSV)[COLUMNS]
    if (existing[results_columns.SUB_DATASET] == SUB_DATASET).any():
        raise SystemExit(f"{SUB_DATASET} rows already present in {SUMMARY_CSV}; nothing to do.")

    new_rows = build_fd001_rows()
    combined = existing.to_dict("records") + new_rows
    combined.sort(key=_sort_key)

    df = pd.DataFrame(combined, columns=COLUMNS)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Added {len(new_rows)} {SUB_DATASET} rows -> {SUMMARY_CSV} (now {len(df)} rows)")


if __name__ == "__main__":
    main()
