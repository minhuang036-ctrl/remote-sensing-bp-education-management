"""Evaluation, cross-validation, and repeated simulation protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from src.config import CATEGORY, CV_FOLDS, N_SIMULATION_RUNS, SEED, TARGET
from src.data_pipeline import (
    augment_training_data,
    create_stratification_labels,
    stratified_train_test_split,
    transform_train_test,
)
from src.models import BPNeuralNetworkModel, SimulationValidationModel


@dataclass
class PredictionBundle:
    y_true: np.ndarray
    simulation_prediction: np.ndarray
    bp_prediction: np.ndarray


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, elapsed_seconds: float) -> dict[str, float]:
    """Calculate manuscript-aligned model evaluation metrics."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    prediction_accuracy = max(0.0, 100.0 - mae)
    management_efficiency = float(np.mean(y_pred))
    error_std = float(np.std(y_true - y_pred))
    stability_score = max(0.0, 100.0 - 4.5 * rmse)

    return {
        "management_efficiency": round(management_efficiency, 4),
        "prediction_accuracy": round(prediction_accuracy, 4),
        "stability_score": round(stability_score, 4),
        "computation_time_seconds": round(float(elapsed_seconds), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_and_evaluate_once(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, PredictionBundle]:
    """Train both models once and return metric table with predictions."""

    X_train, X_test, y_train, y_test, _ = transform_train_test(train_df, test_df)
    train_labels = create_stratification_labels(train_df)

    sim_model = SimulationValidationModel()
    start = time.perf_counter()
    sim_model.fit(X_train, y_train)
    sim_prediction = sim_model.predict(X_test)
    sim_elapsed = time.perf_counter() - start

    X_aug, y_aug = augment_training_data(X_train, y_train, train_labels, random_state=random_state)
    bp_model = BPNeuralNetworkModel(random_state=random_state)
    start = time.perf_counter()
    bp_model.fit_with_transfer_style_initialization(X_train, y_train, X_aug, y_aug)
    bp_prediction = bp_model.predict(X_test)
    bp_elapsed = time.perf_counter() - start

    rows = []
    for model_name, prediction, elapsed in [
        ("Simulation-Validation", sim_prediction, sim_elapsed),
        ("BP Neural Network", bp_prediction, bp_elapsed),
    ]:
        row = {"model": model_name}
        row.update(calculate_metrics(y_test, prediction, elapsed))
        rows.append(row)

    return pd.DataFrame(rows), PredictionBundle(y_test, sim_prediction, bp_prediction)


def run_cross_validation(df: pd.DataFrame, random_state: int = SEED) -> pd.DataFrame:
    """Run leakage-safe 5-fold stratified cross-validation for both models."""

    labels = create_stratification_labels(df)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | int | str]] = []

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(df, labels), start=1):
        train_df = df.iloc[train_index].reset_index(drop=True)
        valid_df = df.iloc[valid_index].reset_index(drop=True)
        metrics_df, _ = train_and_evaluate_once(
            train_df,
            valid_df,
            random_state=random_state + fold_index,
        )
        metrics_df.insert(0, "fold", fold_index)
        rows.extend(metrics_df.to_dict(orient="records"))

    return pd.DataFrame(rows)


def summarize_cross_validation(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level metrics using mean and standard deviation."""

    numeric_columns = [
        "management_efficiency",
        "prediction_accuracy",
        "stability_score",
        "computation_time_seconds",
        "mae",
        "rmse",
        "r2_score",
    ]
    summary = (
        cv_results.groupby("model")[numeric_columns]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def run_repeated_simulations(
    df: pd.DataFrame,
    n_runs: int = N_SIMULATION_RUNS,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Run eight randomized simulation trials while preserving stratified balance."""

    rows = []
    for run_index in range(1, n_runs + 1):
        train_df, test_df = stratified_train_test_split(
            df,
            random_state=random_state + run_index,
        )
        metrics_df, _ = train_and_evaluate_once(
            train_df,
            test_df,
            random_state=random_state + run_index,
        )
        metrics_df.insert(0, "simulation_run", run_index)
        rows.extend(metrics_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def summarize_repeated_simulations(simulation_results: pd.DataFrame) -> pd.DataFrame:
    """Report mean ± standard deviation for the repeated simulation protocol."""

    numeric_columns = [
        "management_efficiency",
        "prediction_accuracy",
        "stability_score",
        "computation_time_seconds",
        "mae",
        "rmse",
        "r2_score",
    ]
    summary = (
        simulation_results.groupby("model")[numeric_columns]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def add_classification_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure category labels are present when a custom dataset is used."""

    if CATEGORY in df.columns:
        return df
    result = df.copy()
    result[CATEGORY] = pd.qcut(
        result[TARGET],
        q=4,
        labels=["very_low", "low", "medium", "high"],
    ).astype(str)
    return result
