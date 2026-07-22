"""Pure telemetry math utilities for acceleration, delta time, and braking analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_array(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def calculate_acceleration(speed_kph: np.ndarray | pd.Series | list[float], time_seconds: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    speed_ms = _as_float_array(speed_kph) / 3.6
    time_seconds_array = _as_float_array(time_seconds)

    if speed_ms.size < 2:
        return np.zeros_like(speed_ms)

    dt = np.diff(time_seconds_array, prepend=time_seconds_array[0])
    dt = np.where(np.abs(dt) < 1e-6, 1e-3, dt)
    dv = np.diff(speed_ms, prepend=speed_ms[0])
    acceleration_ms2 = dv / dt
    return np.round(acceleration_ms2 / 9.81, 3)


def compute_delta_time(
    driver_one: pd.DataFrame,
    driver_two: pd.DataFrame | None = None,
    num_points: int = 2000,
    distance_column: str = "Distance",
    speed_column: str = "Speed",
) -> pd.DataFrame:
    if driver_two is None:
        frame = driver_one.copy()
        if distance_column not in frame.columns:
            distance_column = "distance_meters"
        if "d1_Speed" in frame.columns and "d2_Speed" in frame.columns:
            driver_one = frame[[distance_column, "d1_Speed"]].rename(columns={distance_column: "distance", "d1_Speed": "speed"})
            driver_two = frame[[distance_column, "d2_Speed"]].rename(columns={distance_column: "distance", "d2_Speed": "speed"})
        else:
            raise ValueError("compute_delta_time requires either two frames or a synchronized frame with d1_Speed and d2_Speed columns")
    else:
        driver_one = driver_one[[distance_column, speed_column]].rename(columns={distance_column: "distance", speed_column: "speed"})
        driver_two = driver_two[[distance_column, speed_column]].rename(columns={distance_column: "distance", speed_column: "speed"})

    driver_one = driver_one.sort_values("distance")
    driver_two = driver_two.sort_values("distance")

    common_start = max(float(driver_one["distance"].min()), float(driver_two["distance"].min()))
    common_end = min(float(driver_one["distance"].max()), float(driver_two["distance"].max()))
    common_distance = np.linspace(common_start, common_end, num_points)

    speed_one = np.interp(common_distance, driver_one["distance"].to_numpy(), driver_one["speed"].to_numpy()) / 3.6
    speed_two = np.interp(common_distance, driver_two["distance"].to_numpy(), driver_two["speed"].to_numpy()) / 3.6

    speed_one = np.clip(speed_one, 0.5, None)
    speed_two = np.clip(speed_two, 0.5, None)

    dx = np.diff(common_distance, prepend=common_distance[0])
    time_one = np.cumsum(dx / speed_one)
    time_two = np.cumsum(dx / speed_two)
    delta_time = time_two - time_one

    return pd.DataFrame(
        {
            "distance_meters": common_distance,
            "driver_one_time_s": time_one,
            "driver_two_time_s": time_two,
            "delta_time_s": np.round(delta_time, 4),
        }
    )


def isolate_braking_zones(brake_channel: np.ndarray | pd.Series | list[float], g_force: np.ndarray | pd.Series | list[float], threshold_g: float = -0.5) -> np.ndarray:
    brake_array = _as_float_array(brake_channel)
    g_force_array = _as_float_array(g_force)
    braking_mask = (brake_array > 0) & (g_force_array < threshold_g)
    return braking_mask.astype(int)


class F1TelemetryEngine:
    calculate_acceleration = staticmethod(calculate_acceleration)
    compute_delta_time = staticmethod(compute_delta_time)
    isolate_braking_zones = staticmethod(isolate_braking_zones)


if __name__ == "__main__":
    mock_distance = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
    mock_v1 = np.array([300.0, 280.0, 180.0, 120.0, 90.0])
    mock_v2 = np.array([295.0, 270.0, 160.0, 115.0, 92.0])
    mock_time = np.array([0.0, 0.24, 0.53, 0.98, 1.62])
    mock_brake = np.array([0, 1, 1, 1, 0])

    g_traces = calculate_acceleration(mock_v1, mock_time)
    df_mock = pd.DataFrame({"Distance": mock_distance, "d1_Speed": mock_v1, "d2_Speed": mock_v2})
    delta_profile = compute_delta_time(df_mock)
    brake_zones = isolate_braking_zones(mock_brake, g_traces)

    print(g_traces)
    print(delta_profile.head())
    print(brake_zones)