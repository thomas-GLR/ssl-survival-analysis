# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **Remaining-Useful-Life (RUL) / survival analysis** benchmarking across a family of models, on two datasets:

- **C-MAPSS** (NASA turbofan degradation, sub-datasets `FD001`–`FD004`) — the mature pipeline, everything under `C_MAPSS/`.
- **Scania Component X** (run-to-failure with censoring) — a newer, in-progress pipeline under `Scania/` (`ScaniaDataModule`, `ScaniaDataset`).

The core idea is a **config-driven benchmark**: for a given model and "benchmark version", train/evaluate across every sub-dataset × censored% × broken% combination, writing RMSE and the C-MAPSS score to CSV. Censoring/suspension of data (partially observed lifetimes) is central — several models are designed to exploit censored samples, while supervised models filter them out.

## Running

There is no test suite. Work is driven through two CLI entry points at the repo root, run either locally or via Docker.

**Local (needs repo root on `PYTHONPATH`; top-level packages `models/`, `dataset/`, `constants/`, `pyclus/` are imported absolutely):**

```bash
# Train / benchmark a model over the config matrix
python run_train_cmapss.py --model-version transformer --subset FD001 \
  --config-path C_MAPSS/config --checkpoints-path ./checkpoints \
  --results-path ./outputs --dataset-root ./data/C_MAPSS --benchmark-version default

# Hyperparameter search (Optuna)
python run_hpo.py transformer --subset FD001 --n_trials 50
python run_hpo.py lstm --subset all --n_trials 100 --single_objective
```

`--model-version` ∈ `transformer, lstm, autoencoder, metric, rsf, pyclus, coprog, cnn`.
HPO model names come from the `@register_model` registry in `C_MAPSS/hpo/optuna_search.py` (`lstm, transformer_lstm, cnn1d`) — a *different* namespace than the train entry point.

**Docker (two profiles, `train` and `hpo`):**

```bash
docker compose --profile train --env-file .env.train up
docker compose --profile hpo   --env-file .env.hpo   up
# Override inline or pass explicit CLI args (passthrough mode):
docker compose --profile train run --rm train transformer --subset FD001
```

The entrypoints (`docker/train_entrypoint.sh`, `docker/hpo_entrypoint.sh`) have two modes: **env-var mode** (no args → builds the command from `MODEL_VERSION`/`SUBSET`/… env vars, defined in `.env.train` / `.env.hpo`) and **passthrough mode** (any args → forwarded verbatim to the Python script). Container paths are fixed (`/workspace/data/CMAPSSData`, `/workspace/outputs`, `/workspace/checkpoints`, `/workspace/config`); the host side is set by `*_DIR_HOST` env vars (defaults point at local subdirs; RunPod uses a network-volume path).

## Architecture

### Dispatch by `ModelVersion`

`C_MAPSS/utils/ModelVersion.py` is the enum of supported models. `run_train_cmapss.py` → `reproduce_result()` is model-agnostic; it delegates via three `match` dispatchers in `C_MAPSS/utils/utils_cmapss.py`:

- `get_train_model_method` → the model's `train_model` function (in the matching `utils_*.py`),
- `get_necessary_dataset_keys` / `get_necessary_model_keys` → the required JSON config keys (validated against `constants/necessary_keys_cmapss.py`).

**To add a model:** add an enum member, a `utils_<x>.py` with a `train_model(...)` entry, wire up all three dispatchers, add its `NECESSARY_*` key lists in `constants/necessary_keys_cmapss.py`, and add per-subset config JSON.

The `utils_*.py` modules are the per-family training orchestrators:
- `utils_transformer_lstm.py` — supervised deep models (`transformer`, `lstm`, `cnn`), all wrapped in `TransformerLstmModule` (Lightning). **These drop censored samples** via `get_data_loader_without_censored_data`.
- `utils_self_supervised.py` — `autoencoder` / `metric` pretraining then downstream RUL.
- `utils_random_survival_forest.py` — `rsf` (scikit-survival, non-deep).
- `utils_coprog.py` — `coprog`, co-training that self-labels censored/suspension data (`models/Coprog.py`).
- `utils_pyclus.py` — `pyclus`, wraps the external **CLUS** Java tool.

### Config system

Configs live in `C_MAPSS/config/<benchmark_version>/`. The `--benchmark-version` (e.g. `default`, `test`, `coprog`) selects the folder. Each folder has:
- `benchmark.json` — the sweep matrix: `cmapss_files`, `censored_percentage[]`, `broken_percentage[]`.
- `<model_version>.json` — per-sub-dataset `dataset_params` + `model_params` blocks (keyed by `FD001`…`FD004`).

`reproduce_result` loops the matrix, pulls params per sub-dataset, calls the dispatched `train_model`, and writes both per-sub-dataset "secure" CSVs (incremental) and a final combined CSV to `results-path`. Individual run failures are caught and logged (see the `try/except` in `reproduce_result`) so one bad combination doesn't abort the sweep.

### Lightning layer (`C_MAPSS/lightning_module/`)

`TransformerLstmModule` is the shared supervised wrapper (takes any `nn.Module` in `.model`). Others: `AutoencoderPretrainingModule`, `MetricPretrainingModule`, `UnsupervisedPretrainingModule`, `BaselineModule`. `mixins.py` provides `LoadEncoderMixin` (loads a pretrained encoder's weights out of a checkpoint by stripping the `encoder.` prefix) — the bridge from self-supervised pretraining to downstream models. Raw `nn.Module` architectures are in `C_MAPSS/models/` (`Simple_LSTM`, `TransformerEncoder_LSTM_1`, `CNN1D`).

### Two package roots

- `C_MAPSS/` — self-contained C-MAPSS pipeline (own `dataset/`, `models/`, `lightning_module/`, `utils/`, `hpo/`, `scripts/`, `config/`).
- Top-level `models/`, `dataset/`, `pyclus/`, `constants/` — shared/domain-agnostic code imported absolutely (so **repo root must be on `PYTHONPATH`**; the Docker images set `PYTHONPATH=/app`). `models/self_supervised/` holds the encoder/decoder/RBM building blocks; `pyclus/` is a vendored Python wrapper around CLUS.

### Data

`data/C_MAPSS/` holds the turbofan `train_/test_/RUL_FDxxx` files; `data/Scania_component_X/` holds the Scania readouts/TTE/specifications CSVs (plus a `scania_cache/`). `CMAPSSLoader.get_datasets()` produces train/valid/test `CMAPSSDataset`s with windowing, normalization, operating-condition clustering, and censoring. `ScaniaDataModule` splits **by vehicle** from the train files only (the standalone `validation_*`/`test_*` files are intentionally ignored) and caches processed splits.

## Gotchas

- **CPU thread capping runs before `torch`/`numpy` import** in `run_train_cmapss.py` (`OMP_NUM_THREADS` etc.). Keep any new native-backend imports below that block.
- **`pyclus` needs Java + `clus.jar`.** The jar is expected at `pyclus/clus.jar` (see `pyclus/models/clus.py`); the Docker images install `default-jre`.
- **Shell scripts must stay LF** (`.gitattributes` enforces `*.sh text eol=lf`) or container entrypoints break on Linux. This is a Windows dev environment (`.venv`, PowerShell) targeting Linux/CUDA containers.
- **PyTorch 2.6 checkpoint loading:** custom model classes are registered via `add_safe_globals(...)` in `utils_transformer_lstm.py` before `load_from_checkpoint`; do the same for new architectures you serialize.
- The C-MAPSS "score" (asymmetric, penalizes late predictions harder) is `utils_cmapss.cmapss_score` — reported alongside RMSE everywhere.

## Rules

- The code is writen in english
- The python functions should have docstrings.
- The python functions parameters and return should be typed.