# Ablation Study

This directory contains the code and configuration files used to train and evaluate the different model variants considered in the ablation study.

The configurations for the different ablation groups are provided in the `ablation_study_configurations` directory.

## Training

All models included in the ablation study can be trained by running:

```bash
python3 run_ablation_trainings.py \
    --ablation_configs_path ablation_study_configurations/
```

The `--ablation_configs_path` argument specifies the directory containing the configuration files for the different ablation groups.

Models can be trained individually using *train_ablation.py*

## Evaluation

The trained models can be evaluated using:

```bash
python3 run_ablation_evaluation.py \
    --models_path MODELS_PATH \
    --context CONTEXT_FILE \
    --dataset TEST_SET_FILE
    --dataroot DATASET_ROOT \
    --trainconfig_path ablation_study_configurations/
```

where:

- `--models_path` specifies the directory containing the trained models.
- `--context` specifies the context file used for evaluation.
- `--dataset` specifies the file containing the test set (txt).
- `--dataroot` specifies the root directory of the dataset.
- `--trainconfig_path` specifies the directory containing the ablation configuration files.

Evaluation results can optionally be saved to a CSV file using the `--saveresults` argument.

Models can be evaluated individually using *evaluate_model_ablation.py*

## Trained Models

The trained models used in the ablation study are available at:

**[Trained models](https://www.dropbox.com/scl/fi/n7exz7j1366v3pcmypuaq/ablation_Groups_models.zip?rlkey=1uaqhe9pf9nbjxdevdsduguu3&st=wi59q2i3&dl=0)**
