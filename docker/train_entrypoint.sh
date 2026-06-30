#!/bin/bash
# docker/train_entrypoint.sh
# ──────────────────────────────────────────────────────────────────────────────
# Entry point for the train container.
#
# Two modes:
#   1. Env-var mode (docker compose up / RunPod autostart)
#      The command is built from environment variables.
#      docker compose --profile train up
#
#   2. Passthrough mode (docker run / docker compose run with explicit args)
#      All arguments after the image name are forwarded directly to run_train_cmapss.py.
#      docker compose --profile train run train transformer --subset FD001
#      docker run cmapss-train transformer --subset all
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Passthrough mode: user supplied CLI args ───────────────────────────────────
if [ "$#" -gt 0 ]; then
    echo "▶ python run_train_cmapss.py $*"
    exec python run_train_cmapss.py "$@"
fi

# ── Env-var mode: build command from environment ───────────────────────────────
OUTPUT_DIR="${RESULT_DIR:-/workspace/outputs}"

# Ensure output directory exists even if the volume was not pre-populated
mkdir -p "$OUTPUT_DIR"

ARGS=(
    --model-version     "${MODEL_VERSION:-lstm}"
    --config-path       "${CONFIG_DIR:-/workspace/config}"
    --checkpoints-path  "${CHECKPOINT_DIR:-/workspace/checkpoints}"
    --results-path      "$OUTPUT_DIR"
    --dataset-root      "${DATA_DIR:-/workspace/data/CMAPSSData}"
    --benchmark-version "${BENCHMARK_VERSION:-default}"
)

echo "▶ python run_train_cmapss.py ${ARGS[*]}"
exec python run_train_cmapss.py "${ARGS[@]}"

# 3. Keep the container alive for debugging/inspection
echo "Python script finished. Keeping the pod alive..."
sleep infinity
