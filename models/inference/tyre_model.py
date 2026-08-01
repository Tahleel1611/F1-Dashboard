"""XGBoost tyre degradation regressor with deterministic synthetic training data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


class F1TyrePredictor:
    feature_columns = ["tyre_age", "track_temp", "fuel_load", "compound_encoded"]

    compound_curve_map = {
        0: lambda age, temp: 0.14 * np.power(age, 1.22) * np.exp((temp - 35.0) / 32.0),
        1: lambda age, temp: 0.068 * age * (0.92 + temp / 120.0),
        2: lambda age, temp: 0.028 * np.power(age, 0.87) * np.power(temp / 35.0, 0.68),
    }

    def __init__(self) -> None:
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
        )
        self._is_trained = False

    @staticmethod
    def generate_synthetic_telemetry_dataset(samples: int = 2400) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        tyre_age = rng.integers(1, 40, size=samples)
        track_temp = rng.uniform(25.0, 48.0, size=samples)
        fuel_load = rng.uniform(10.0, 110.0, size=samples)
        compound_encoded = rng.integers(0, 3, size=samples)

        base_time = 88.5 + (fuel_load * 0.034)
        soft_curve = F1TyrePredictor.compound_curve_map[0](tyre_age, track_temp)
        medium_curve = F1TyrePredictor.compound_curve_map[1](tyre_age, track_temp)
        hard_curve = F1TyrePredictor.compound_curve_map[2](tyre_age, track_temp)
        degradation = np.select(
            [compound_encoded == 0, compound_encoded == 1, compound_encoded == 2],
            [soft_curve, medium_curve, hard_curve],
        )

        target_lap_time = base_time + degradation + rng.normal(0.0, 0.04, size=samples)
        return pd.DataFrame(
            {
                "tyre_age": tyre_age,
                "track_temp": track_temp,
                "fuel_load": fuel_load,
                "compound_encoded": compound_encoded,
                "target_lap_time": target_lap_time,
            }
        )

    def train_model(self, df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty:
            raise ValueError("Training data must not be empty")

        features = df[self.feature_columns]
        target = df["target_lap_time"]
        split_idx = max(int(len(df) * 0.8), 1)
        x_train, x_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        self.model.fit(x_train, y_train)
        self._is_trained = True

        if len(x_test) == 0:
            return 0.0, 1.0

        predictions = self.model.predict(x_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))
        return rmse, r2

    def ensure_trained(self) -> None:
        if not self._is_trained:
            dataset = self.generate_synthetic_telemetry_dataset()
            self.train_model(dataset)

    def retrain_from_historical_session(self, df: pd.DataFrame) -> Tuple[float, float]:
        normalized = df.copy()
        if "compound_code" in normalized.columns and "compound_encoded" not in normalized.columns:
            normalized = normalized.rename(columns={"compound_code": "compound_encoded"})
        if "target_lap_time" not in normalized.columns:
            raise ValueError("Historical data must include target_lap_time")
        rmse, r2 = self.train_model(normalized)
        return rmse, r2

    @staticmethod
    def estimate_compound_degradation(compound: int, tyre_age: np.ndarray, track_temp: np.ndarray) -> np.ndarray:
        curve = F1TyrePredictor.compound_curve_map.get(compound, F1TyrePredictor.compound_curve_map[1])
        return np.asarray(curve(tyre_age, track_temp), dtype=float)

    def predict_stint_trajectory(
        self, start_lap: int, end_lap: int, track_temp: float, fuel_load: float, compound: int
    ) -> np.ndarray:
        self.ensure_trained()
        laps = np.arange(start_lap, end_lap + 1)
        fuel_curve = np.clip(fuel_load - ((laps - start_lap) * 1.6), 0.0, None)
        stint_features = pd.DataFrame(
            {
                "tyre_age": laps,
                "track_temp": np.full_like(laps, track_temp, dtype=float),
                "fuel_load": fuel_curve,
                "compound_encoded": np.full_like(laps, compound, dtype=float),
            }
        )
        baseline_prediction = self.model.predict(stint_features[self.feature_columns])
        compound_adjustment = self.estimate_compound_degradation(compound, laps, np.full_like(laps, track_temp, dtype=float))
        return baseline_prediction + 0.12 * compound_adjustment


if __name__ == "__main__":
    predictor = F1TyrePredictor()
    data = predictor.generate_synthetic_telemetry_dataset()
    rmse_metric, r2_metric = predictor.train_model(data)
    print(rmse_metric, r2_metric)
    print(predictor.predict_stint_trajectory(1, 5, 38.5, 65.0, 0))