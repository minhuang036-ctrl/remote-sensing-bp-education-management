"""Data generation, preprocessing, splitting, and augmentation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    AUGMENTATION_NOISE,
    CATEGORY,
    FEATURES,
    N_SAMPLES,
    SEED,
    SEMESTER,
    TARGET,
    TEST_SIZE,
    UNIVERSITY,
)


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _minmax(values: np.ndarray) -> np.ndarray:
    low = np.min(values)
    high = np.max(values)
    if np.isclose(high, low):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def generate_benchmark_dataset(random_state: int = SEED, n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Create a deterministic anonymized benchmark dataset with the study feature design.

    The generator follows the manuscript structure: 528 semester-unit samples,
    six universities, two semester groups, 11 education-management indicators,
    and three remote-sensing indicators.
    """

    rng = np.random.default_rng(random_state)
    samples_per_university = n_samples // 6
    records = []

    university_resource_bias = rng.normal(0, 1.0, 6)
    university_environment_bias = rng.normal(0, 1.0, 6)

    for university_index in range(6):
        for row_index in range(samples_per_university):
            semester = 1 if row_index < samples_per_university / 2 else 2
            resource_bias = university_resource_bias[university_index]
            environment_bias = university_environment_bias[university_index]

            students = _clip(
                rng.normal(12450 + resource_bias * 650 + (semester - 1.5) * 350, 1800),
                8200,
                18600,
            )
            managers = _clip(rng.normal(185 + resource_bias * 13 + students / 850, 18), 120, 260)
            material_use = _clip(rng.normal(72.4 + resource_bias * 3.5, 6.4), 55.1, 89.7)
            facility_coverage = _clip(rng.normal(68.3 + resource_bias * 3.2, 5.2), 50.2, 82.6)
            resource_investment = _clip(rng.normal(15.7 + resource_bias * 1.2, 2.1), 9.4, 22.1)
            personnel_input = _clip(rng.normal(3.6 + resource_bias * 0.25, 0.55), 2.1, 5.2)
            equipment_input = _clip(rng.normal(4.8 + resource_bias * 0.35, 0.75), 3.0, 7.1)
            reward_innovation = _clip(rng.normal(6.4 + resource_bias * 0.65, 1.35), 3.0, 10.0)
            teaching_innovation = _clip(rng.normal(5.9 + resource_bias * 0.55, 1.55), 2.0, 9.0)
            mode_innovation = _clip(rng.normal(4.3 + resource_bias * 0.45, 1.05), 1.0, 7.0)
            management_activity = _clip(rng.normal(7.5 + resource_bias * 0.8, 1.8), 3.0, 12.0)

            building_density = _clip(rng.normal(0.42 + environment_bias * 0.025, 0.055), 0.25, 0.60)
            green_space_ratio = _clip(
                rng.normal(31.5 - building_density * 18 + environment_bias * 2.8, 5.5),
                18.0,
                45.2,
            )
            surface_temperature = _clip(
                rng.normal(28.6 + building_density * 7.2 - green_space_ratio * 0.07, 1.65),
                23.1,
                34.7,
            )

            records.append(
                {
                    UNIVERSITY: f"U{university_index + 1}",
                    SEMESTER: f"semester_{semester}",
                    "number_of_students": students,
                    "number_of_managers": managers,
                    "teaching_material_utilization": material_use,
                    "teaching_facility_coverage": facility_coverage,
                    "teaching_resource_investment": resource_investment,
                    "personnel_management_input": personnel_input,
                    "management_equipment_input": equipment_input,
                    "reward_punishment_innovation": reward_innovation,
                    "teaching_innovation_frequency": teaching_innovation,
                    "management_mode_innovation": mode_innovation,
                    "educational_management_activity": management_activity,
                    "building_density": building_density,
                    "green_space_ratio": green_space_ratio,
                    "surface_temperature": surface_temperature,
                }
            )

    dataset = pd.DataFrame.from_records(records)
    feature_matrix = dataset[FEATURES].copy()

    normalized = pd.DataFrame(
        {column: _minmax(feature_matrix[column].to_numpy(dtype=float)) for column in FEATURES}
    )

    resource_signal = (
        0.09 * (1 - normalized["number_of_students"])
        + 0.06 * normalized["number_of_managers"]
        + 0.12 * normalized["teaching_material_utilization"]
        + 0.14 * normalized["teaching_facility_coverage"]
        + 0.11 * normalized["teaching_resource_investment"]
        + 0.06 * normalized["personnel_management_input"]
        + 0.06 * normalized["management_equipment_input"]
        + 0.07 * normalized["reward_punishment_innovation"]
        + 0.09 * normalized["teaching_innovation_frequency"]
        + 0.07 * normalized["management_mode_innovation"]
        + 0.08 * normalized["educational_management_activity"]
    )

    spatial_signal = (
        0.05 * (1 - np.abs(normalized["building_density"] - 0.55))
        + 0.08 * normalized["green_space_ratio"]
        + 0.07 * (1 - normalized["surface_temperature"])
    )

    semester_boost = dataset[SEMESTER].map({"semester_1": -1.2, "semester_2": 1.2}).to_numpy()
    noise = rng.normal(0, 2.25, len(dataset))
    efficiency = 52 + 37 * (resource_signal + spatial_signal) + semester_boost + noise
    dataset[TARGET] = _clip(efficiency.to_numpy(dtype=float), 45, 97)

    dataset[CATEGORY] = pd.qcut(
        dataset[TARGET],
        q=4,
        labels=["very_low", "low", "medium", "high"],
    ).astype(str)

    return dataset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_preprocessor() -> Pipeline:
    """Return a leakage-safe preprocessing pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )


def create_stratification_labels(df: pd.DataFrame) -> pd.Series:
    """Create joint labels preserving semester and efficiency category balance."""

    return df[SEMESTER].astype(str) + "_" + df[CATEGORY].astype(str)


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using an 80:20 stratified random strategy."""

    strat_labels = create_stratification_labels(df)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_labels,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def transform_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """Fit preprocessing only on training data and transform train/test features."""

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURES])
    X_test = preprocessor.transform(test_df[FEATURES])
    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)
    return X_train, X_test, y_train, y_test, preprocessor


def augment_training_data(
    X: np.ndarray,
    y: np.ndarray,
    labels: pd.Series | np.ndarray,
    random_state: int = SEED,
    noise_level: float = AUGMENTATION_NOISE,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance training groups and perturb continuous features for small-sample robustness."""

    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)
    X_parts = [X]
    y_parts = [y]

    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = int(np.max(counts))

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        deficit = max_count - len(indices)
        if deficit <= 0:
            continue
        sampled = rng.choice(indices, size=deficit, replace=True)
        synthetic_X = X[sampled] + rng.normal(0, noise_level, size=(deficit, X.shape[1]))
        synthetic_X = np.clip(synthetic_X, 0.0, 1.0)
        synthetic_y = y[sampled] + rng.normal(0, 0.75, size=deficit)
        synthetic_y = np.clip(synthetic_y, 0.0, 100.0)
        X_parts.append(synthetic_X)
        y_parts.append(synthetic_y)

    return np.vstack(X_parts), np.concatenate(y_parts)
