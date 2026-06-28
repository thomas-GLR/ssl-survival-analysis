#!/bin/bash
# docker/hpo_entrypoint.sh
# ──────────────────────────────────────────────────────────────────────────────
# Entry point for the HPO container.
#
# Two modes:
#   1. Env-var mode (docker compose up / RunPod autostart)
#      The command is built from environment variables.
#      docker compose --profile hpo up
#
#   2. Passthrough mode (docker run / docker compose run with explicit args)
#      All arguments after the image name are forwarded directly to run_hpo.py.
#      docker compose --profile hpo run hpo transformer --subset FD001 --n_trials 30
#      docker run cmapss-hpo transformer --subset all --single_objective
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Passthrough mode: user supplied CLI args ───────────────────────────────────
if [ "$#" -gt 0 ]; then
    echo "▶ python run_hpo.py $*"
    exec python run_hpo.py "$@"
fi

# ── Env-var mode: build command from environment ───────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"

# Ensure output directory exists even if the volume was not pre-populated
mkdir -p "$OUTPUT_DIR"

ARGS=(
    "${MODEL:-transformer_lstm}"
    --subset     "${SUBSET:-all}"
    --n_trials   "${N_TRIALS:-50}"
    --max_epochs "${MAX_EPOCHS:-100}"
    --data_dir   "${DATA_DIR:-/data/CMAPSSData}"
    --output_dir "$OUTPUT_DIR"
)

# Optional explicit Optuna storage URL
# Falls back to sqlite:////<output_dir>/optuna.db inside run_hpo.py
if [ -n "${STORAGE:-}" ]; then
    ARGS+=(--storage "$STORAGE")
fi

# Single-objective mode
if [ "${SINGLE_OBJECTIVE:-false}" = "true" ]; then
    ARGS+=(--single_objective)
fi

echo "▶ python run_hpo.py ${ARGS[*]}"
exec python run_hpo.py "${ARGS[@]}"
