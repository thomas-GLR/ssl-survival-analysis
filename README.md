# Survival Analysis for CMAPSS and Scania Component X datasets

## CMAPSS

## Scania Component X

### Hyperparameter optimisation (HPO)

Hyperparameter search on the Scania Component X dataset is driven by
[`run_hpo_scania.py`](run_hpo_scania.py). It wraps an **Optuna** study:
single-objective, minimising the validation RMSE (`val_rmse`) with a TPE sampler
and Hyperband pruning. Scania has no sub-datasets, so there is a single study per
model. `val_score`, `test_rmse` and `test_score` are recorded per trial for
inspection but are **never** optimised.

#### 1. Running the script

```bash
python run_hpo_scania.py cnn \
  --n_trials 50 \
  --max_epochs 200 \
  --data_dir ./data/Scania_component_X \
  --cache_dir ./scania_cache \
  --output_dir ./outputs/hpo_scania
```

Inline command
```bash
python run_hpo_scania.py cnn --n_trials 50 --max_epochs 200 --data_dir ./data/Scania_component_X --cache_dir ./scania_cache --output_dir ./outputs/hpo_scania
```

The first (positional) argument is the **model to optimise**. The choices come
from the `@register_model` registry in `scania/hpo/optuna_search.py` — currently
only `cnn` is registered.

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `model` (positional) | yes | — | Model to optimise. Must be a registered model (`cnn`). |
| `--n_trials` | no | `50` (or `$N_TRIALS`) | Number of Optuna trials to run. |
| `--max_epochs` | no | `100` (or `$MAX_EPOCHS`) | Max epochs per trial; also the Hyperband `max_resource`. |
| `--data_dir` | no | `./data/Scania_component_X` (or `$DATA_DIR`) | Root folder of the Scania data files. |
| `--cache_dir` | no | `<data_dir>/scania_cache` (or `$CACHE_DIR`) | Base cache directory for the processed dataset splits. A per-config sub-folder (`cm=<counter_mode>_sl=<sequence_len>`) is created inside it. |
| `--output_dir` | no | `./outputs` (or `$OUTPUT_DIR`) | Directory for the Optuna DB and the result files. Created if missing. |
| `--storage` | no | `sqlite:////<output_dir>/optuna.db` | Optuna storage URL — see below. |

Every optional argument falls back to an environment variable (shown in the
Default column), which is convenient for Docker / RunPod runs.

#### 2. How `--storage` works (resuming & parallel runs)

The study is created with `load_if_exists=True`, so the storage URL determines
whether a run **starts fresh** or **resumes**:

- **Omitted** → a local SQLite DB at `sqlite:////<output_dir>/optuna.db`.
  Re-running the same command reuses that DB and **continues the existing study**
  (the study name is `<model>_scania`), adding more trials rather than starting
  over.
- **Explicit SQLite** (e.g. `--storage sqlite:////abs/path/optuna.db`) → same
  behaviour, at a path you control.
- **A shared database** (e.g. `--storage postgresql://user:pass@host/db`) → lets
  several processes/machines contribute trials to the **same** study in parallel.
  Use this for distributed HPO; SQLite is not suited to concurrent writers.

> Note the four slashes in the SQLite default (`sqlite:////...`): three for the
> URL scheme plus one for the absolute POSIX path.

#### 3. What gets searched

Per trial, the objective searches these training/data hyperparameters:

- `lr` — learning rate, log-uniform in `[1e-5, 1e-2]`.
- `batch_size` — one of `{64, 128, 256}`.
- `sequence_len` — window length, within the model's registered
  `seq_len_range` (e.g. `(30, 50)` for `cnn`).

Architecture hyperparameters (if the model has any) are suggested inside that
model's builder in `scania/hpo/optuna_search.py`. `CNN1D` is sequence-length
agnostic and has no tunable architecture params, so for `cnn` only the three
above are searched. Each trial trains with early stopping on `val_rmse`
(patience 20) and Hyperband pruning.

#### 4. Outputs

At the end of the run, under `--output_dir`:

- `optuna.db` — the Optuna study database (SQLite, unless `--storage` overrides it).
- `<model>_trials.csv` — every completed trial with its params and its
  `val_rmse` / `val_score` / `test_rmse` / `test_score`.
- `<model>_best_params.json` — the best trial's hyperparameters.

A summary table of the best trial is also logged to the console.

#### 5. Example

Run 100 trials for the CNN, capped at 80 epochs each, resuming into a named
SQLite DB, and write results under `./outputs/hpo_scania`:

```bash
python run_hpo_scania.py cnn \
  --n_trials 100 \
  --max_epochs 80 \
  --data_dir ./data/Scania_component_X \
  --output_dir ./outputs/hpo_scania \
  --storage sqlite:////workspace/outputs/hpo_scania/optuna.db
```

Once finished, feed the values from
`./outputs/hpo_scania/cnn_best_params.json` into the corresponding
`scania/config/<benchmark-version>/*.json` config to train the tuned model with
[`run_train_scania.py`](run_train_scania.py).
