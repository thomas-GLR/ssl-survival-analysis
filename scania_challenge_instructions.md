# Scania challenge instructions

## ConformalCoTrainingEnsembleScaniaChallenge

- DO NOT MODIFY "CoTrainingEnsemble_v2"
- Create a class named ConformalCoTrainingEnsembleScaniaChallenge that extend CoTrainingEnsemble_v2
- The class ConformalCoTrainingEnsembleScaniaChallenge will be used to test our method on scania challenge
- Scania challenge is based on the scania_score() in file metrics.py
- The ConformalCoTrainingEnsembleScaniaChallenge class will use a validation_challenge and test_challenge set that come from the data (same preprocessing as train data) :
  - test_challenge_label = data/Scania_component_X/test_labels
  - test_challenge_data = data/Scania_component_X/test_operational_readouts.csv
  - validation_challenge_label = data/Scania_component_X/validation_labels.csv
  - validation_challenge_data data/Scania_component_X/validation_operational_readouts.csv
- As we use regressor model the purpose is to convert the RUL to the target class same as run_challenge_evaluation() from challenge.py
- The weight will be computed based on the validation_challenge scania_score
- Don't use anymore test_data and test_label anymore
- We keep using val_data and val_label as regression for early stopping and checkpoints save
- parameters list are (init in parent __init__ if same as parent) :
  - models
  - weights
  - verbose
  - confidence
  - inference_batch_size
  - use_average_window_confidence
  - confidence_width_threshold
  - use_monotone_projection
  - monotone_residual_weight
  - disable_isotonic_regression
  - isotonic_time_weighting
  - cotraining_survival_loss_lambda
  - use_mondrian_categorizer
  - mondrian_no_bins
  - use_cps
  - difficulty_estimator_k

- CoTrainingEnsemble_v2 (parent) parameters rules (this rules can be applied in utils_conformal_cotraining_ensemble_challenge) :
  - "dataset_version" should always be equals to DatasetVersion.SCANIA
  - "use_fine_tuning" always equal to False can't be changed
  - "use_cotraining_ensemble_survival_loss_function" always equal to True can't be changed
  - "peer_weighted_pseudo_label" always equal to False can't be changed
  - "use_average_window_confidence" always equal to True can't be changed
  - "bagging_failure_data" always equal to False can't be changed
  - "keep_best_model_mode" always equal to null can't be changed

- "computing_weight_mode" don't follow any rules of parent

- Override function :
  - calculate_weights : always compute based on validation_challenge dataset scania score
  - _log_stage_metrics : should just compute iteration_time, scania score on validation_challenge and scania score on test_challenge. No RMSE computation

- Don't take care of parallel run in ConformalCoTrainingEnsembleScaniaChallenge
- Don't take care of pretrained model in ConformalCoTrainingEnsembleScaniaChallenge
- Save checkpoints of each model when they beat its best validation_challenge scania score
- Save ckeckpoints of all models when the weighted_validation_score based on validation_challenge beat its own best scania_score
- to save different checkpoint separate the checkpoints for each specific models and the weighted validation score in different folder
- name the checkpoint with the scania score when it is for each model and with wieghted_scania_score write in the name of the file the model (CNN, LSTM, Transformer) and if its for single model or weighted
- When the train() finish return each model that have the best scania score and all the models that compute the best scania score

## conformal_co_training_ensemble_challenge configuration
```json
{
  "dataset_params": {
    "sequence_len": 64,
    "seed": 42,
    "data_fraction": 1.0,
    "val_rate": 0.15,
    "calib_rate": 0.1,
    "stratify": true,
    "norm_type": "z-score",
    "num_workers": 20,
    "pin_memory": true,
    "return_sequence_label": false,
    "batch_size": 128,
    "shuffle_loader": true,
    "counter_mode": "cumulative",
    "include_histograms": true,
    "histogram_mode": "sum"
  },
  "model_params": {
    "models": [
      {
        "cnn": {
          "model_params": {},
          "lr": 0.001,
          "weight_decay": 0,
          "max_epochs": 500,
          "patience": 100,
          "rul_target_standardization": false
        }
      },
      {
        "lstm": {
          "model_params": {
            "hidden_dim": 64,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.2,
            "fc_layer_dim": 256,
            "fc_dropout": 0.2
          },
          "lr": 0.002,
          "weight_decay": 0,
          "max_epochs": 500,
          "patience": 100,
          "rul_target_standardization": false
        }
      },
      {
        "transformer_features": {
          "model_params": {
            "transformer_encoder_head_num": 4,
            "transformer_num_layer": 2,
            "fc_layer_dim": 256,
            "fc_dropout": 0.3
          },
          "lr": 0.0004,
          "weight_decay": 0,
          "max_epochs": 500,
          "patience": 100,
          "rul_target_standardization": true
        }
      }
    ]
  },
  "training_params": {
    "iterations": 20,
    "suspension_pool_size": 0.35,
    "add_ratio": 0.1,
    "confidence": 0.95,
    "inference_batch_size": 1024,
    "confidence_width_threshold": 3.0,
    "use_monotone_projection": true,
    "monotone_residual_weight": 1.0,
    "disable_isotonic_regression": false,
    "bagging_failure_data": false,
    "isotonic_time_weighting":  true,
    "cotraining_survival_loss_lambda": 0.7,
    "use_mondrian_categorizer": true,
    "mondrian_no_bins": 7,
    "use_cps": true,
    "difficulty_estimator_k": 25
  }
}
```

## utils_conformal_cotraining_ensemble_challenge.py

- create file utils_conformal_cotraining_ensemble_challenge.py
- define train_model() method same as utils_conformal_cotraining_ensemble.py (without cmapss parameter)
- For the final result just compute the result for validation challenge and test challenge with each best model weighted by the validation score and with the weighted models 

## run_scania_challenge.py

- Create a new file run_scania_challenge.py that run utils_conformal_cotraining_ensemble_challenge.py
- can only run ConformalCoTrainingEnsembleScaniaChallenge.py
- Wanted parameters :
  - "--config-path"
  - "--checkpoints-path"
  - "--results-path"
  - "--dataset-root"
  - "--benchmark-version"
  - "--run-name"
