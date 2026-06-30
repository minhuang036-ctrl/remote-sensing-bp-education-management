"""Essential result visualizations for the modeling workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURE_DISPLAY_NAMES, FEATURES


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    corr = df[FEATURES].corr(method="pearson")
    labels = [FEATURE_DISPLAY_NAMES.get(name, name) for name in FEATURES]

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Correlation Heatmap of Dataset Indicators")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison(y_true: np.ndarray, sim_pred: np.ndarray, bp_pred: np.ndarray, output_path: Path) -> None:
    order = np.argsort(y_true)
    sample_limit = min(120, len(y_true))
    x_axis = np.arange(sample_limit)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x_axis, y_true[order][:sample_limit], label="Actual")
    ax.plot(x_axis, sim_pred[order][:sample_limit], label="Simulation-Validation")
    ax.plot(x_axis, bp_pred[order][:sample_limit], label="BP Neural Network")
    ax.set_xlabel("Sorted test sample index")
    ax.set_ylabel("Management efficiency")
    ax.set_title("Actual and Predicted Management Efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_performance(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["management_efficiency", "prediction_accuracy", "stability_score"]
    x_axis = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, model_name in [(-width / 2, "Simulation-Validation"), (width / 2, "BP Neural Network")]:
        values = metrics_df.loc[metrics_df["model"] == model_name, metrics].iloc[0].to_numpy(dtype=float)
        ax.bar(x_axis + offset, values, width, label=model_name)

    ax.set_xticks(x_axis)
    ax.set_xticklabels(["Efficiency", "Accuracy", "Stability"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_simulation_stability(simulation_results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name, group in simulation_results.groupby("model"):
        ax.plot(group["simulation_run"], group["stability_score"], marker="o", label=model_name)

    ax.set_xlabel("Repeated simulation run")
    ax.set_ylabel("Stability score")
    ax.set_ylim(0, 100)
    ax.set_title("Repeated Simulation Stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
