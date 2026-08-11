"""Strict request and telemetry schemas for the TRK-72 FastAPI gateway."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SessionContext(BaseModel):
    year: int = Field(..., ge=2020, le=2026)
    round_number: int = Field(..., ge=1, le=25)
    session_type: str = Field(..., description="FP1, FP2, FP3, Q, SQ, Sprint, Race")

    @field_validator("session_type")
    @classmethod
    def validate_session_type(cls, value: str) -> str:
        valid_sessions = {"FP1", "FP2", "FP3", "Q", "SQ", "Sprint", "Race"}
        if value not in valid_sessions:
            raise ValueError(f"Invalid session type: {value}")
        return value


class TelemetryChannelPacket(BaseModel):
    timestamp: datetime
    lap_time_ms: float = Field(..., ge=0.0)
    distance_m: float = Field(..., ge=0.0)
    speed_kph: int = Field(..., ge=0, le=380)
    throttle_pct: float = Field(..., ge=0.0, le=100.0)
    brake_pressure_bar: float = Field(..., ge=0.0, le=150.0)
    rpm: int = Field(..., ge=0, le=15000)
    gear: int = Field(..., ge=1, le=8)
    drs_status: int = Field(..., ge=0, le=14)

    model_config = {"frozen": True}


class TelemetryDerivedFrame(BaseModel):
    distance_m: float = Field(..., ge=0.0)
    speed_kph: float = Field(..., ge=0.0)
    throttle_pct: float = Field(..., ge=0.0, le=100.0)
    brake_active: int = Field(..., ge=0, le=1)
    gear: int = Field(..., ge=1, le=8)
    rpm: int = Field(..., ge=0, le=20000)
    x_coord: float
    y_coord: float
    z_coord: float = 0.0
    longitudinal_g: float
    lateral_g: float
    throttle_smoothness: float
    braking_zone: int = Field(..., ge=0, le=1)
    mguk_output_kw: float = Field(..., ge=0.0, le=350.0)
    battery_soc_pct: float = Field(..., ge=0.0, le=100.0)
    derating_active: int = Field(..., ge=0, le=1)
    aero_mode: Literal["X", "Z"] = "Z"
    aero_switch: int = Field(..., ge=0, le=1)
    boost_mode_active: int = Field(..., ge=0, le=1)
    overtake_mode_active: int = Field(..., ge=0, le=1)
    electric_energy_kwh: float = Field(..., ge=0.0)


class TelemetryStreamPacket(BaseModel):
    timestamp: datetime
    driver_one: TelemetryDerivedFrame
    driver_two: TelemetryDerivedFrame
    delta_time_s: float
    session_name: str
    regulation_context: str = "2026-Hybrid"


class StrategyOptimizationRequest(BaseModel):
    total_laps: int = Field(57, ge=10, le=80)
    initial_fuel: float = Field(100.0, ge=5.0, le=115.0)
    pit_loss_seconds: float = Field(23.0, ge=15.0, le=35.0)
    track_temp_c: float = Field(35.0, ge=10.0, le=55.0)
    fuel_burn_kg_per_lap: float = Field(1.6, ge=0.5, le=3.0)
    track_evolution_per_lap: float = Field(0.0, ge=-0.5, le=0.5)
    compound_matrix: list[list[int]] = Field(
        default_factory=lambda: [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    )
    degradation_slopes: dict[str, float] = Field(default_factory=lambda: {"soft": 0.125, "medium": 0.065, "hard": 0.024})


class ModelTrainingRequest(BaseModel):
    training_mode: Literal["synthetic", "historical"] = "synthetic"
    session_context: SessionContext | None = None


class ModelTrainingResponse(BaseModel):
    success: bool
    training_mode: str
    rmse: float | None = None
    r2_score: float | None = None
    records_used: int


class TelemetryComparisonRequest(BaseModel):
    season_year: int = Field(2024, ge=2020, le=2026)
    round_id: int | str = Field(...)
    session_code: str = Field("Q")
    driver_one: str = Field(..., min_length=3, max_length=3)
    driver_two: str = Field(..., min_length=3, max_length=3)


class TelemetryComparisonResponse(BaseModel):
    success: bool
    session_name: str
    aligned_points: int
    delta_end_s: float
    preview_rows: list[dict[str, object]]
    driver_metadata: dict[str, object]


class DashboardFilterState(BaseModel):
    season_year: int = Field(2024, ge=2020, le=2026)
    circuit_venue: str = Field(default="Bahrain Grand Prix")
    session_type: str = Field(default="Q")
    driver_one: str = Field(default="VER", min_length=3, max_length=3)
    driver_two: str = Field(default="LEC", min_length=3, max_length=3)