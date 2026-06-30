"""Run the complete remote-sensing enhanced education management experiment."""

from __future__ import annotations

import pandas as pd

from src.config import OUTPUT_DIR, SEED
from src.data_pipeline import generate_benchmark_dataset, stratified_train_test_split
from src.evaluation import (
    run_cross_validation,
    run_repeated_simulations,
    summarize_cross_validation,
    summarize_repeated_simulations,
    train_and_evaluate_once,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_model_performance,
    plot_prediction_comparison,
    plot_simulation_stability,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_benchmark_dataset(random_state=SEED)
    dataset.to_csv(OUTPUT_DIR / "processed_dataset.csv", index=False)

    train_df, test_df = stratified_train_test_split(dataset, random_state=SEED)
    metrics_df, predictions = train_and_evaluate_once(train_df, test_df, random_state=SEED)
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

    cv_results = run_cross_validation(dataset, random_state=SEED)
    cv_summary = summarize_cross_validation(cv_results)
    cv_results.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "cross_validation_summary.csv", index=False)

    simulation_results = run_repeated_simulations(dataset, random_state=SEED)
    simulation_summary = summarize_repeated_simulations(simulation_results)
    simulation_results.to_csv(OUTPUT_DIR / "repeated_simulation_results.csv", index=False)
    simulation_summary.to_csv(OUTPUT_DIR / "repeated_simulation_summary.csv", index=False)

    plot_correlation_heatmap(dataset, OUTPUT_DIR / "correlation_heatmap.png")
    plot_prediction_comparison(
        predictions.y_true,
        predictions.simulation_prediction,
        predictions.bp_prediction,
        OUTPUT_DIR / "prediction_comparison.png",
    )
    plot_model_performance(metrics_df, OUTPUT_DIR / "model_performance_comparison.png")
    plot_simulation_stability(simulation_results, OUTPUT_DIR / "simulation_stability.png")

    print("\nRemote Sensing Enhanced BP Education Management Experiment Completed\n")
    print("Model Comparison")
    print(metrics_df.to_string(index=False))
    print("\nRepeated Simulation Summary")
    print(simulation_summary.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
