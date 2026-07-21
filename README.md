The purpose of this branch is to search for ssl-state-of-the-art that work on scania and cmapss datasets.

# Survival Analysis for CMAPSS and Scania Component X datasets

## CMAPSS

## Scania Component X

The script to train and evaluate models on the Scania Component X dataset is
[`run_train_scania.py`](run_train_scania.py).

### Coprog

Coprog is a **co-training** model: two different base models (e.g. a CNN and a
Transformer) are trained on the labelled (uncensored) samples, then iteratively
**self-label the censored / suspension samples** for each other. Final
predictions are a validation-weighted ensemble of the two models. The two base
models can be trained **in parallel on separate GPUs** to speed up the process.

#### 1. Running the script

```bash
python run_train_scania.py \
  --model-version coprog \
  --config-path ./scania/config \
  --checkpoints-path ./checkpoints \
  --results-path ./outputs \
  --dataset-root ./data/Scania_component_X \
  --dataset-cache-dir ./scania_cache \
  --benchmark-version default \
  --run-name my_first_coprog_run \
  --gpu-ids 0 1
```

Inline command :
```bash
python run_train_scania.py --model-version coprog --config-path ./scania/config --checkpoints-path ./checkpoints --results-path ./outputs --dataset-root ./data/Scania_component_X --dataset-cache-dir ./scania_cache --benchmark-version default --run-name my_first_coprog_run --gpu-ids 0 1
```

| Argument | Required | Description |
| --- | --- | --- |
| `--model-version` | yes | The model to train. Use `coprog` for Coprog. |
| `--config-path` | yes | Path to the config **root** folder (the folder that contains the benchmark-version sub-folders, e.g. `./scania/config`). |
| `--checkpoints-path` | yes | Where the trained models (`coprog_cnn.pth`, `coprog_lstm.pth`) are saved. Created if missing. |
| `--results-path` | yes | Where the metrics CSVs and the run log are written. Created if missing. |
| `--dataset-root` | yes | Root folder of the Scania Component X data files (readouts / TTE / specifications CSVs). |
| `--dataset-cache-dir` | yes | Folder used to cache the processed dataset splits (built once, reused afterwards). |
| `--benchmark-version` | no (default `default`) | Selects the config **sub-folder** to read, i.e. `<config-path>/<benchmark-version>/coprog.json`. `default` is the full run; `test` is a fast smoke-test config (1 epoch, 2 iterations). |
| `--run-name` | no (default `""`) | Optional name; when set, results and checkpoints are written under a sub-folder of this name. |
| `--gpu-ids` | no (default `None`) | GPU selection — see below. **Coprog only.** |

The actual config file loaded by the command above is
`./scania/config/default/coprog.json` (i.e.
`<config-path>/<benchmark-version>/<model-version>.json`).

#### 2. How `--gpu-ids` works

`--gpu-ids` accepts zero, one, or several integer GPU ids (space-separated). It
only affects Coprog, because Coprog is the only model that trains two networks
that can run in parallel.

| Value | Behaviour |
| --- | --- |
| *(omitted)* | `None` → single GPU / automatic device selection. The two models are trained **sequentially**. |
| `--gpu-ids 0` | Pin all training to GPU `0`. The two models are trained **sequentially** on that GPU. |
| `--gpu-ids 0 1` | Train the **two models in parallel**, the first on GPU `0` and the second on GPU `1`. This is the fastest option when two GPUs are available. |

Notes:
- Give **at most two** ids — Coprog has exactly two base models, so extra ids are
  not used.
- The ids are physical GPU indices as seen by CUDA (`nvidia-smi`). On a
  single-GPU machine, simply omit the flag (or pass `--gpu-ids 0`).

#### 3. The Coprog config file

The config is a single JSON file with three top-level blocks:
`dataset_params`, `model_params`, and `training_params`. All keys listed below
are **required** (see `constants/necessary_keys_scania.py`).

##### `dataset_params` — how the data is loaded and windowed

| Key | Type | Meaning |
| --- | --- | --- |
| `sequence_len` | int | Length of the sliding time window fed to the models. |
| `seed` | int | Random seed for the split (reproducibility). |
| `val_rate` | float | Fraction of vehicles used for validation. **Must be > 0** (Coprog needs a validation set for early stopping and for the ensemble weights). |
| `test_rate` | float | Fraction of vehicles used for the test set. |
| `stratify` | bool | Whether to stratify the vehicle split. |
| `norm_type` | `"z-score"` or `null` | Feature normalization. |
| `num_workers` | int | DataLoader worker processes. |
| `pin_memory` | bool | DataLoader `pin_memory`. |
| `return_sequence_label` | bool | Whether the dataset returns a per-step label sequence (keep `false` for Coprog). |
| `batch_size` | int | Training/eval batch size. |
| `shuffle_loader` | bool | Whether to shuffle the training DataLoader. |
| `counter_mode` | `"cumulative"`, `"delta"`, or `"both"` | How the Scania counter variables are encoded. |

##### `model_params` — the two base models

`model_params` has exactly two keys, `first_model` and `second_model`. Each one
contains **exactly one** key naming the model version, whose value is that
model's hyper-parameter dict. Supported model versions for Coprog:

- `cnn` — no hyper-parameters (`{}`).
- `lstm` — `hidden_dim`, `lstm_num_layers`, `lstm_dropout`, `fc_layer_dim`, `fc_dropout`.
- `transformer_features` — `transformer_encoder_head_num`, `transformer_num_layer`, `fc_layer_dim`, `fc_dropout`.
- `transformer_time_sequence` — same keys as `transformer_features`.

(`feature_num`, `sequence_len`, and `d_model` are injected automatically from
the dataset, so you don't set them here.)

##### `training_params` — the co-training loop

All list-valued keys have **2 entries**, one per base model (`[first_model, second_model]`).

| Key | Type | Meaning |
| --- | --- | --- |
| `lr` | list[float] (len 2) | Learning rate for each base model. |
| `patiences` | list[int] (len 2) | Early-stopping patience (in epochs) for each base model. |
| `max_epochs` | list[int] (len 2) | Max training epochs for each base model. |
| `coprog_iterations` | int | Number of co-training rounds (self-labelling passes). |
| `coprog_suspension_pool_size` | int | How many censored/suspension samples are pulled into the self-labelling pool each iteration. |
| `rul_target_standardization` | list[bool] (len 2) | Whether to standardize (z-score) the RUL target for each model. Stats are computed on the uncensored training labels only; predictions are de-normalized back to real RUL units for all metrics. |

##### Full example (`scania/config/default/coprog.json`)

A CNN + Transformer ensemble, trained in a 10-round co-training loop:

```json
{
  "dataset_params": {
    "sequence_len": 32,
    "seed": 42,
    "val_rate": 0.2,
    "test_rate": 0.1,
    "stratify": true,
    "norm_type": "z-score",
    "num_workers": 4,
    "pin_memory": true,
    "return_sequence_label": false,
    "batch_size": 128,
    "shuffle_loader": true,
    "counter_mode": "cumulative"
  },
  "model_params": {
    "first_model": {
      "cnn": {}
    },
    "second_model": {
      "transformer_features": {
        "transformer_encoder_head_num": 8,
        "transformer_num_layer": 2,
        "fc_layer_dim": 128,
        "fc_dropout": 0.2
      }
    }
  },
  "training_params": {
    "lr": [0.0002, 0.0002],
    "patiences": [50, 50],
    "max_epochs": [500, 500],
    "coprog_iterations": 10,
    "coprog_suspension_pool_size": 25,
    "rul_target_standardization": [true, true]
  }
}
```

For a quick end-to-end check without a long run, use the `test` benchmark
version (`--benchmark-version test`), which is the same config with
`max_epochs: [1, 1]`, `coprog_iterations: 2`, and `coprog_suspension_pool_size: 4`.

#### 4. Outputs

After a run you will find, under `--results-path` (inside the `--run-name`
sub-folder if given):

- `coprog-scania.csv` — per-model and weighted-ensemble RMSE / score plus the
  ensemble weights `weight_h1` / `weight_h2`.
- `coprog-per-stage-scania.csv` — metrics tracked at each stage of the
  co-training loop (initial / per-iteration / final).
- the saved training parameters and a run log;

and, under `--checkpoints-path`, the two trained models `coprog_cnn.pth` and
`coprog_lstm.pth`.

### CoTrainingEnsemble (v1)

CoTrainingEnsemble is the generalisation of Coprog to a **configurable number of
base models** (2 or more) drawn from the four allowed architectures (`cnn`,
`lstm`, `transformer_features`, `transformer_time_sequence`). Like Coprog, the
models are trained on the labelled (uncensored) samples and then iteratively
**self-label the censored / suspension samples** for each other; final
predictions are a validation-weighted ensemble of all models. **v1 selects which
self-labelled censored units to add by voting** across the models
(`SelectionMode.VOTING`) and can optionally **fine-tune** rather than retrain
from scratch at each step. The base models can be trained **in parallel across
several GPUs** to speed up the process.

#### 1. Running the script

```bash
python run_train_scania.py \
  --model-version co_training_ensemble \
  --config-path ./scania/config \
  --checkpoints-path ./checkpoints \
  --results-path ./outputs \
  --dataset-root ./data/Scania_component_X \
  --dataset-cache-dir ./scania_cache \
  --benchmark-version default \
  --run-name my_first_cotraining_run \
  --gpu-ids 0 1
```

Inline command :
```bash
python run_train_scania.py --model-version co_training_ensemble --config-path ./scania/config --checkpoints-path ./checkpoints --results-path ./outputs --dataset-root ./data/Scania_component_X --dataset-cache-dir ./scania_cache --benchmark-version default --run-name my_first_cotraining_run --gpu-ids 0 1
```

| Argument | Required | Description |
| --- | --- | --- |
| `--model-version` | yes | The model to train. Use `co_training_ensemble` for CoTrainingEnsemble v1. |
| `--config-path` | yes | Path to the config **root** folder (the folder that contains the benchmark-version sub-folders, e.g. `./scania/config`). |
| `--checkpoints-path` | yes | Where the trained models (one `.pth` per base model, e.g. `co_training_ensemble_cnn_0.pth`) are saved. Created if missing. |
| `--results-path` | yes | Where the metrics CSVs and the run log are written. Created if missing. |
| `--dataset-root` | yes | Root folder of the Scania Component X data files (readouts / TTE / specifications CSVs). |
| `--dataset-cache-dir` | yes | Folder used to cache the processed dataset splits (built once, reused afterwards). |
| `--benchmark-version` | no (default `default`) | Selects the config **sub-folder** to read, i.e. `<config-path>/<benchmark-version>/co_training_ensemble.json`. `default` is the full run; `test` is a fast smoke-test config (1 epoch, 2 iterations). |
| `--run-name` | no (default `""`) | Optional name; when set, results and checkpoints are written under a sub-folder of this name. |
| `--gpu-ids` | no (default `None`) | GPU selection — see below. |

The actual config file loaded by the command above is
`./scania/config/default/co_training_ensemble.json` (i.e.
`<config-path>/<benchmark-version>/<model-version>.json`).

#### 2. How `--gpu-ids` works

`--gpu-ids` accepts zero, one, or several integer GPU ids (space-separated). For
the co-training ensembles it distributes the base models **round-robin** across
the given GPUs.

| Value | Behaviour |
| --- | --- |
| *(omitted)* | `None` → single GPU / automatic device selection. The models are trained **sequentially**. |
| `--gpu-ids 0` | Pin all training to GPU `0`. The models are trained **sequentially** on that GPU. |
| `--gpu-ids 0 1` | Train the models **in parallel**, distributed round-robin across GPUs `0` and `1`. This is the fastest option when several GPUs are available. |

Notes:
- Unlike Coprog (exactly two models), CoTrainingEnsemble can have any number of
  models, so you may pass as many ids as you have GPUs; models are assigned
  round-robin (extra ids beyond the model count are simply unused).
- The ids are physical GPU indices as seen by CUDA (`nvidia-smi`). On a
  single-GPU machine, simply omit the flag (or pass `--gpu-ids 0`).

#### 3. The CoTrainingEnsemble config file

The config is a single JSON file with three top-level blocks:
`dataset_params`, `model_params`, and `training_params`. All keys listed below
are **required** (see `constants/necessary_keys_scania.py`).

##### `dataset_params` — how the data is loaded and windowed

Identical to Coprog (see the Coprog table above). `val_rate` **must be > 0** (the
ensemble needs a validation set for early stopping / best-model selection and for
the ensemble weights).

##### `model_params` — the list of base models

`model_params` has exactly one key, `models`, whose value is a **list of at least
two** model entries. Each entry is a **single-key dict** `{model_version: {...}}`
where `model_version` is one of `cnn`, `lstm`, `transformer_features`,
`transformer_time_sequence`. Adding a model to the ensemble is just adding an
entry to this list.

Each entry is **self-contained** and must provide these keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `model_params` | dict | The architecture hyper-parameters for that model version (`{}` for `cnn`; the transformer/LSTM keys otherwise — see the Coprog `model_params` notes). |
| `lr` | float | Learning rate for this model. |
| `max_epochs` | int | Max training epochs for this model. |
| `patience` | int | Early-stopping patience (in epochs) for this model. |
| `rul_target_standardization` | bool | Whether to standardize (z-score) the RUL target for this model. Stats are computed on the uncensored training labels only; predictions are de-normalized back to real RUL units for all metrics. |

(`feature_num`, `sequence_len`, and `d_model` are injected automatically from the
dataset, so you don't set them here.)

##### `training_params` — the co-training loop

| Key | Type | Meaning |
| --- | --- | --- |
| `iterations` | int | Number of co-training rounds (self-labelling passes). |
| `suspension_pool_size` | float in `(0, 1]` | Fraction of censored/suspension units sampled into the self-labelling pool each iteration. (Note: a **fraction**, unlike Coprog's integer count.) |
| `add_ratio` | float in `(0, 1]` | Fraction of the pool actually added (self-labelled) per iteration. |
| `is_fine_tuning_during_finding_best_suspension_data` | bool | Fine-tune (vs. retrain from scratch) while searching for the best censored candidates. |
| `is_fine_tuning_for_last_step` | bool | Fine-tune (vs. retrain from scratch) after the censored data has been selected. |
| `fine_tune_lr_factor` | float | LR multiplier applied while fine-tuning. |
| `fine_tune_max_epochs` | int | Max epochs for fine-tuning. |

##### Full example (`scania/config/default/co_training_ensemble.json`)

A four-model ensemble (CNN + LSTM + two transformers), trained in a 10-round
co-training loop with fine-tuning disabled:

```json
{
  "dataset_params": {
    "sequence_len": 32,
    "seed": 42,
    "val_rate": 0.2,
    "test_rate": 0.1,
    "stratify": true,
    "norm_type": "z-score",
    "num_workers": 4,
    "pin_memory": true,
    "return_sequence_label": false,
    "batch_size": 128,
    "shuffle_loader": true,
    "counter_mode": "both"
  },
  "model_params": {
    "models": [
      {
        "cnn": {
          "model_params": {},
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      },
      {
        "lstm": {
          "model_params": {
            "hidden_dim": 256,
            "lstm_num_layers": 3,
            "lstm_dropout": 0.3,
            "fc_layer_dim": 128,
            "fc_dropout": 0.2
          },
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      },
      {
        "transformer_features": {
          "model_params": {
            "transformer_encoder_head_num": 8,
            "transformer_num_layer": 2,
            "fc_layer_dim": 128,
            "fc_dropout": 0.2
          },
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      },
      {
        "transformer_time_sequence": {
          "model_params": {
            "transformer_encoder_head_num": 8,
            "transformer_num_layer": 2,
            "fc_layer_dim": 128,
            "fc_dropout": 0.2
          },
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      }
    ]
  },
  "training_params": {
    "iterations": 10,
    "suspension_pool_size": 0.5,
    "add_ratio": 0.3,
    "is_fine_tuning_during_finding_best_suspension_data": false,
    "is_fine_tuning_for_last_step": false,
    "fine_tune_lr_factor": 0.1
  }
}
```

For a quick end-to-end check without a long run, use the `test` benchmark
version (`--benchmark-version test`), which is a smaller config with
`max_epochs: 1`, `iterations: 2`, and a tiny `suspension_pool_size`.

#### 4. Outputs

After a run you will find, under `--results-path` (inside the `--run-name`
sub-folder if given):

- `co_training_ensemble-scania.csv` — per-model and weighted-ensemble RMSE /
  score plus the ensemble weights (`test_rmse_h{i}` / `test_score_h{i}` /
  `weight_h{i}` per model, and `test_rmse_weighted` / `test_score_weighted`).
- `co_training_ensemble-per-stage-scania.csv` — metrics tracked at each stage of
  the co-training loop (initial / per-iteration / final).
- `predictions_co_training_ensemble_test_h{i}_scania.csv` — one per-model
  prediction file (targets vs. predictions), plus
  `predictions_co_training_ensemble_test_weighted_scania.csv` for the weighted
  ensemble.
- the saved training parameters and a run log;

and, under `--checkpoints-path`, one trained model per base model, named
`co_training_ensemble_<version>_<i>.pth` (e.g. `co_training_ensemble_cnn_0.pth`).

### CoTrainingEnsemble v2

CoTrainingEnsemble v2 has the **same structure** as v1 (a configurable list of 2+
base models, per-model prediction files, validation-weighted ensemble, parallel
GPU training). The differences are in **how censored units are selected** and in
the training loop:

- v2 always trains **from scratch** (no fine-tuning), so it drops the
  fine-tuning knobs.
- Instead of voting, v2 ranks censored units by **`crepes` conformal-interval
  width** and self-labels the most confident (narrowest-interval) ones, so it
  takes a single `confidence` parameter.

#### 1. Running the script

Same as v1, but with `--model-version co_training_ensemble_v2`:

```bash
python run_train_scania.py \
  --model-version co_training_ensemble_v2 \
  --config-path ./scania/config \
  --checkpoints-path ./checkpoints \
  --results-path ./outputs \
  --dataset-root ./data/Scania_component_X \
  --dataset-cache-dir ./scania_cache \
  --benchmark-version default \
  --run-name my_first_cotraining_v2_run \
  --gpu-ids 0 1
```

Inline command :
```bash
python run_train_scania.py --model-version co_training_ensemble_v2 --config-path ./scania/config --checkpoints-path ./checkpoints --results-path ./outputs --dataset-root ./data/Scania_component_X --dataset-cache-dir ./scania_cache --benchmark-version default --run-name my_first_cotraining_v2_run --gpu-ids 0 1
```

The arguments are identical to v1 (see the table above); the config file loaded
is `<config-path>/<benchmark-version>/co_training_ensemble_v2.json`, and the
checkpoints/outputs are named with the `co_training_ensemble_v2` prefix.
`--gpu-ids` behaves exactly as for v1.

#### 2. The CoTrainingEnsemble v2 config file

`dataset_params` and `model_params` are **identical to v1** (same `models` list
of self-contained model entries). Only `training_params` differs:

| Key | Type | Meaning |
| --- | --- | --- |
| `iterations` | int | Number of co-training rounds (self-labelling passes). |
| `suspension_pool_size` | float in `(0, 1]` | Fraction of censored/suspension units sampled into the self-labelling pool each iteration. |
| `add_ratio` | float in `(0, 1]` | Fraction of the pool actually added (self-labelled) per iteration. |
| `confidence` | float in `(0, 1)` | Confidence level for the `crepes` conformal intervals used to rank censored units (higher → wider intervals). |

##### Full example (`scania/config/default/co_training_ensemble_v2.json`)

A three-model ensemble with 95%-confidence conformal selection:

```json
{
  "dataset_params": {
    "sequence_len": 32,
    "seed": 42,
    "val_rate": 0.2,
    "test_rate": 0.1,
    "stratify": true,
    "norm_type": "z-score",
    "num_workers": 4,
    "pin_memory": true,
    "return_sequence_label": false,
    "batch_size": 128,
    "shuffle_loader": true,
    "counter_mode": "both"
  },
  "model_params": {
    "models": [
      {
        "cnn": {
          "model_params": {},
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      },
      {
        "lstm": {
          "model_params": {
            "hidden_dim": 256,
            "lstm_num_layers": 3,
            "lstm_dropout": 0.3,
            "fc_layer_dim": 128,
            "fc_dropout": 0.2
          },
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      },
      {
        "transformer_features": {
          "model_params": {
            "transformer_encoder_head_num": 8,
            "transformer_num_layer": 2,
            "fc_layer_dim": 128,
            "fc_dropout": 0.2
          },
          "lr": 0.0002,
          "max_epochs": 500,
          "patience": 50,
          "rul_target_standardization": true
        }
      }
    ]
  },
  "training_params": {
    "iterations": 10,
    "suspension_pool_size": 0.5,
    "add_ratio": 0.3,
    "confidence": 0.95
  }
}
```

As for v1, `--benchmark-version test` runs a fast smoke-test config
(`max_epochs: 1`, `iterations: 2`, small `suspension_pool_size`).

#### 3. Outputs

Same as v1, but with the `co_training_ensemble_v2` prefix:

- `co_training_ensemble_v2-scania.csv` — per-model and weighted-ensemble RMSE /
  score plus the ensemble weights.
- `co_training_ensemble_v2-per-stage-scania.csv` — per-stage co-training metrics.
- `predictions_co_training_ensemble_v2_test_h{i}_scania.csv` (per model) and
  `predictions_co_training_ensemble_v2_test_weighted_scania.csv`.
- the saved training parameters and a run log;

and, under `--checkpoints-path`, one trained model per base model, named
`co_training_ensemble_v2_<version>_<i>.pth`.

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
