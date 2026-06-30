# Remote Sensing Enhanced Data Modeling and BP Neural Networks for Education Management

This repository provides a reproducible Python implementation of a remote sensing enhanced education management prediction framework. The project integrates administrative education-management indicators with campus-level spatial indicators and compares a simulation-validation model with a Back Propagation neural network.

The implementation supports the complete computational workflow used for model development, including data preprocessing, feature integration, stratified train-test partitioning, cross-validation, repeated simulation trials, statistical reporting, and visualization.

## Project Objective

The objective of this project is to evaluate education management efficiency using hybrid institutional and spatial data. The framework combines educational management indicators, remote sensing based spatial indicators, simulation-validation modeling, BP neural network prediction, and statistical robustness analysis.

## Method Overview

The workflow contains the following stages:

1. Generate or load education-management data.
2. Integrate administrative and remote sensing features.
3. Apply missing value handling, scaling, and data balancing.
4. Split the dataset using an 80:20 stratified train-test strategy.
5. Train and evaluate the simulation-validation model.
6. Train and evaluate the BP neural network model.
7. Perform 5-fold cross-validation.
8. Perform eight repeated random simulation trials.
9. Report mean and standard deviation of model performance.
10. Generate essential result figures.

## Feature Set

The model uses 14 input indicators.

### Education Management Indicators

1. Number of students
2. Number of managers
3. Utilization of teaching materials
4. Coverage of teaching facilities
5. Investment in teaching resources
6. Personnel management input
7. Management equipment input
8. Reward and punishment innovation frequency
9. Teaching innovation frequency
10. Management mode innovation frequency
11. Educational management activity frequency

### Remote Sensing Indicators

12. Building density
13. Green space ratio
14. Surface temperature

## Models Implemented

### Simulation-Validation Model

The simulation-validation model estimates management efficiency by combining normalized institutional indicators, weighted administrative conditions, and spatial influence factors.

### BP Neural Network Model

The BP neural network is implemented as a multilayer feed-forward neural model trained using backpropagation. The model includes normalized input features, hidden-layer nonlinear transformation, early stopping, repeated training under randomized simulation conditions, transfer-style initialization for improved convergence, and feature perturbation based data augmentation.

## Repository Structure

```text
remote-sensing-bp-education-management/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
└── outputs/
    └── .gitkeep
```

## Installation

Create a Python environment and install the required packages.

```bash
pip install -r requirements.txt
```

## Run the Project

Execute the full experiment using:

```bash
python main.py
```

## Generated Outputs

After execution, the following files are generated inside the `outputs/` directory:

```text
processed_dataset.csv
model_comparison_metrics.csv
cross_validation_results.csv
repeated_simulation_results.csv
correlation_heatmap.png
prediction_comparison.png
model_performance_comparison.png
simulation_stability.png
```

## Evaluation Metrics

The implementation reports prediction accuracy, management efficiency, stability score, computation time, cross-validation mean score, cross-validation standard deviation, repeated simulation mean, and repeated simulation standard deviation.

## Data Note

The repository supports both user-provided institutional data and a deterministic anonymized benchmark generator. When original institutional records cannot be distributed, the benchmark generator creates a reproducible dataset following the feature definitions, value ranges, and modeling protocol described in the study.

To use a custom dataset, replace the generated dataset step in `src/data_pipeline.py` with a CSV loading function containing the same feature names defined in `src/config.py`.

## Reproducibility

The project uses a fixed random seed for repeatable outputs. The same code execution produces consistent datasets, splits, model results, and figures unless configuration values are changed.

## Code Availability

The Python implementation in this repository includes the preprocessing pipeline, remote-sensing feature integration, simulation-validation algorithm, BP neural network model, cross-validation procedure, repeated random simulation protocol, statistical analysis, and visualization scripts used to reproduce the computational workflow.
