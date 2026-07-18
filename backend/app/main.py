"""FastAPI gateway for telemetry, strategy, and performance analytics services."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from backend.app.core.schemas import StrategyOptimizationRequest, TelemetryComparisonRequest
from telemetry.engine.math_ops import F1TelemetryEngine
from telemetry.engine.pipeline import F1DualIngestionEngine
from telemetry.engine.strategy_sim import F1StrategySimulator

app = FastAPI(
    title="MISSION CONTROL | TRK-72",
    version="3.0.0",
    description="Telemetry and performance analytics gateway for Formula 1 operations.",
)

telemetry_engine = F1DualIngestionEngine()


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "ONLINE", "subsystems": {"database": "EXPECTED", "ml_inference": "READY", "telemetry": "READY"}}


@app.post("/api/v1/strategy/optimize")
async def optimize_race_strategy(payload: StrategyOptimizationRequest) -> dict[str, object]:
    try:
        solver = F1StrategySimulator(total_race_laps=payload.total_laps, pit_loss_seconds=payload.pit_loss_seconds)
        pit_lap, compound_sequence, total_time = solver.evaluate_one_stop_strategies(
            initial_fuel=payload.initial_fuel,
            track_temp=payload.track_temp_c,
        )
        return {
            "success": True,
            "optimal_strategy": {
                "pit_lap": pit_lap,
                "compound_sequence": compound_sequence,
                "projected_race_duration_seconds": round(total_time, 3),
                "formatted_pace": f"{int(total_time // 60)}m {round(total_time % 60, 2)}s",
            },
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Strategic engine solver fault: {error}") from error


@app.post("/api/v1/telemetry/comparison")
async def compare_telemetry(payload: TelemetryComparisonRequest) -> dict[str, object]:
    try:
        aligned_telemetry, metadata = telemetry_engine.fetch_comparison_dataset(
            year=payload.season_year,
            round_id=payload.round_id,
            session_code=payload.session_code,
            d1=payload.driver_one,
            d2=payload.driver_two,
        )
        delta_profile = F1TelemetryEngine.compute_delta_time(aligned_telemetry)
        return {
            "success": True,
            "session_name": metadata["session_name"],
            "aligned_points": int(len(aligned_telemetry)),
            "delta_end_s": float(delta_profile["delta_time_s"].iloc[-1]),
            "preview_rows": aligned_telemetry.head(5).to_dict(orient="records"),
            "driver_metadata": metadata,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Telemetry comparison fault: {error}") from error