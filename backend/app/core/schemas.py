"""Strict request and telemetry schemas for the TRK-72 FastAPI gateway."""

from __future__ import annotations

from datetime import date, datetime

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


class StrategyOptimizationRequest(BaseModel):
    total_laps: int = Field(57, ge=10, le=80)
    initial_fuel: float = Field(100.0, ge=5.0, le=115.0)
    pit_loss_seconds: float = Field(23.0, ge=15.0, le=35.0)
    track_temp_c: float = Field(35.0, ge=10.0, le=55.0)


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


class DashboardFilterState(BaseModel):
    season_year: int = Field(2024, ge=2020, le=2026)
    circuit_venue: str = Field(default="Bahrain Grand Prix")
    session_type: str = Field(default="Q")
    driver_one: str = Field(default="VER", min_length=3, max_length=3)
    driver_two: str = Field(default="LEC", min_length=3, max_length=3)