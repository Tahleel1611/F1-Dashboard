"""Dual-driver telemetry ingestion and distance-aligned interpolation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import fastf1 as ff1
except Exception:  # pragma: no cover
    ff1 = None

from telemetry.engine.config import get_runtime_config


LOGGER = logging.getLogger(__name__)


class F1DualIngestionEngine:
    def __init__(self, cache_dir: str | None = None, data_mode: str | None = None):
        runtime_config = get_runtime_config()
        self.cache_dir = Path(cache_dir or runtime_config.fastf1_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_mode = (data_mode or runtime_config.data_mode).upper()
        if ff1 is not None:
            ff1.Cache.enable_cache(str(self.cache_dir))

    @staticmethod
    def _extract_channel(frame: pd.DataFrame, candidates: tuple[str, ...], fallback: float = 0.0) -> np.ndarray:
        for column_name in candidates:
            if column_name in frame.columns:
                return frame[column_name].to_numpy(dtype=float)
        return np.full(len(frame), fallback, dtype=float)

    def _build_synthetic_lap(self, driver_code: str, bias: float = 0.0, num_points: int = 2000) -> pd.DataFrame:
        distance = np.linspace(0.0, 5412.0, num_points)
        speed = 295.0 - 145.0 * np.exp(-((distance - (1100.0 + bias)) / 280.0) ** 2) - 165.0 * np.exp(-((distance - (3500.0 - bias)) / 420.0) ** 2)
        throttle = np.clip(100.0 - np.where(speed < 220.0, 75.0, 0.0), 0.0, 100.0)
        brake = (speed < 210.0).astype(float)
        rpm = np.clip(12000.0 + (speed * 8.0), 8000.0, 15500.0)
        gear = np.clip(np.round(speed / 38.0), 1.0, 8.0)
        drs = np.where(speed > 250.0, 14.0, 0.0)
        angle = (distance / distance.max()) * np.pi * 2.0
        x_coord = 900.0 * np.cos(angle + bias / 2000.0)
        y_coord = 580.0 * np.sin(angle)

        return pd.DataFrame(
            {
                "Distance": distance,
                "Speed": speed,
                "Throttle": throttle,
                "Brake": brake,
                "RPM": rpm,
                "Gear": gear,
                "DRS": drs,
                "X": x_coord,
                "Y": y_coord,
                "driver_code": driver_code,
            }
        )

    def _standardize_telemetry(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        frame = telemetry.copy().sort_values("Distance")
        return pd.DataFrame(
            {
                "Distance": frame["Distance"].to_numpy(dtype=float),
                "Speed": self._extract_channel(frame, ("Speed",)),
                "Throttle": self._extract_channel(frame, ("Throttle", "Throttle%", "ThrottlePct")),
                "Brake": self._extract_channel(frame, ("Brake", "BrakeActive")),
                "RPM": self._extract_channel(frame, ("RPM", "EngineRPM")),
                "Gear": self._extract_channel(frame, ("Gear", "nGear")),
                "DRS": self._extract_channel(frame, ("DRS", "drs", "DRSStatus")),
                "X": self._extract_channel(frame, ("X", "CoordinateX")),
                "Y": self._extract_channel(frame, ("Y", "CoordinateY")),
            }
        )

    def _align_frames(self, driver_one: pd.DataFrame, driver_two: pd.DataFrame, num_points: int = 2000) -> pd.DataFrame:
        driver_one = self._standardize_telemetry(driver_one)
        driver_two = self._standardize_telemetry(driver_two)

        start_distance = max(float(driver_one["Distance"].min()), float(driver_two["Distance"].min()))
        end_distance = min(float(driver_one["Distance"].max()), float(driver_two["Distance"].max()))
        common_distance = np.linspace(start_distance, end_distance, num_points)

        def interpolate(frame: pd.DataFrame, column_name: str) -> np.ndarray:
            return np.interp(common_distance, frame["Distance"].to_numpy(), frame[column_name].to_numpy())

        synchronized = pd.DataFrame({"Distance": common_distance})
        for prefix, frame in (("d1_", driver_one), ("d2_", driver_two)):
            synchronized[f"{prefix}Speed"] = interpolate(frame, "Speed")
            synchronized[f"{prefix}Throttle"] = interpolate(frame, "Throttle")
            synchronized[f"{prefix}Brake"] = interpolate(frame, "Brake")
            synchronized[f"{prefix}RPM"] = interpolate(frame, "RPM")
            synchronized[f"{prefix}Gear"] = np.round(interpolate(frame, "Gear")).astype(int)
            synchronized[f"{prefix}DRS"] = np.round(interpolate(frame, "DRS")).astype(int)
            synchronized[f"{prefix}X"] = interpolate(frame, "X")
            synchronized[f"{prefix}Y"] = interpolate(frame, "Y")

        return synchronized

    def _fetch_live_dataset(self, year: int, round_id: int | str, session_code: str, d1: str, d2: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        if ff1 is None:
            raise RuntimeError("FastF1 unavailable")

        session = ff1.get_session(year, round_id, session_code)
        session.load(telemetry=True, laps=True, weather=False)

        selected_laps = session.laps.pick_drivers([d1, d2])
        lap_d1 = selected_laps[selected_laps["Driver"] == d1].pick_fastest()
        lap_d2 = selected_laps[selected_laps["Driver"] == d2].pick_fastest()

        telemetry_d1 = self._standardize_telemetry(lap_d1.get_telemetry())
        telemetry_d2 = self._standardize_telemetry(lap_d2.get_telemetry())
        aligned_telemetry = self._align_frames(telemetry_d1, telemetry_d2)

        metadata = {
            "session_name": f"{year} {session.event['EventName']} - {session_code}",
            "d1_metadata": {
                "driver": d1,
                "team": lap_d1["Team"],
                "lap_time": str(lap_d1["LapTime"]).split()[-1][:9],
                "compound": lap_d1["Compound"],
                "life": int(lap_d1["TyreLife"]),
            },
            "d2_metadata": {
                "driver": d2,
                "team": lap_d2["Team"],
                "lap_time": str(lap_d2["LapTime"]).split()[-1][:9],
                "compound": lap_d2["Compound"],
                "life": int(lap_d2["TyreLife"]),
            },
        }
        return aligned_telemetry, metadata

    def _fetch_offline_dataset(self, year: int, session_code: str, d1: str, d2: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        driver_one = self._build_synthetic_lap(d1, bias=0.0)
        driver_two = self._build_synthetic_lap(d2, bias=24.0)
        aligned_telemetry = self._align_frames(driver_one, driver_two)
        metadata = {
            "session_name": f"{year} Synthetic Session - {session_code}",
            "d1_metadata": {"driver": d1, "team": "SYN", "lap_time": "1:28.000", "compound": "SOFT", "life": 3},
            "d2_metadata": {"driver": d2, "team": "SYN", "lap_time": "1:28.600", "compound": "MEDIUM", "life": 4},
        }
        return aligned_telemetry, metadata

    def fetch_comparison_dataset(
        self, year: int, round_id: int | str, session_code: str, d1: str, d2: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        try:
            if self.data_mode == "OFFLINE":
                return self._fetch_offline_dataset(year, session_code, d1, d2)

            return self._fetch_live_dataset(year, round_id, session_code, d1, d2)
        except Exception:
            LOGGER.warning("Falling back to synthetic telemetry for %s/%s/%s", year, round_id, session_code, exc_info=True)
            return self._fetch_offline_dataset(year, session_code, d1, d2)


def get_telemetry_broker(cache_dir: str | None = None, data_mode: str | None = None) -> F1DualIngestionEngine:
    return F1DualIngestionEngine(cache_dir=cache_dir, data_mode=data_mode)


if __name__ == "__main__":
    engine = F1DualIngestionEngine()
    data, meta = engine.fetch_comparison_dataset(year=2024, round_id="Bahrain", session_code="Q", d1="VER", d2="LEC")
    print(meta["session_name"])
    print(data.head())