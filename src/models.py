"""Simulation-validation and BP neural network model definitions."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from src.config import BP_MODEL_PARAMS, FEATURES, SEED


class SimulationValidationModel:
    """Interpretable simulation-validation baseline for management efficiency.

    The model combines normalized administrative and spatial indicators using
    study-aligned weights and then calibrates the score using the training mean.
    """

    def __init__(self) -> None:
        self.feature_names = FEATURES
        self.train_mean_: float | None = None
        self.weights_ = np.array(
            [0.06, 0.05, 0.12, 0.13, 0.10, 0.06, 0.06, 0.07, 0.08, 0.07, 0.08, 0.04, 0.05, 0.03],
            dtype=float,
        )
        self.weights_ = self.weights_ / self.weights_.sum()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimulationValidationModel":
        self.train_mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.train_mean_ is None:
            raise RuntimeError("SimulationValidationModel must be fitted before prediction.")

        X_adjusted = np.array(X, dtype=float, copy=True)
        surface_temperature_index = self.feature_names.index("surface_temperature")
        building_density_index = self.feature_names.index("building_density")
        students_index = self.feature_names.index("number_of_students")

        X_adjusted[:, surface_temperature_index] = 1.0 - X_adjusted[:, surface_temperature_index]
        X_adjusted[:, students_index] = 1.0 - X_adjusted[:, students_index]
        X_adjusted[:, building_density_index] = 1.0 - np.abs(X_adjusted[:, building_density_index] - 0.55)

        simulation_score = X_adjusted @ self.weights_
        raw_efficiency = 48.0 + 44.0 * simulation_score
        calibrated = 0.78 * raw_efficiency + 0.22 * self.train_mean_
        return np.clip(calibrated, 0.0, 100.0)


class BPNeuralNetworkModel:
    """Back Propagation neural network with transfer-style initialization."""

    def __init__(self, random_state: int = SEED) -> None:
        self.random_state = random_state
        self.model = MLPRegressor(random_state=random_state, warm_start=True, **BP_MODEL_PARAMS)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BPNeuralNetworkModel":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X, y)
        return self

    def fit_with_transfer_style_initialization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_augmented: np.ndarray,
        y_augmented: np.ndarray,
    ) -> "BPNeuralNetworkModel":
        """Pretrain on augmented data, then fine-tune on original training data."""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_augmented, y_augmented)
            self.model.set_params(max_iter=250, learning_rate_init=0.001, warm_start=True)
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0.0, 100.0)
