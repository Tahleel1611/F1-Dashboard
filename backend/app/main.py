"""FastAPI gateway for telemetry, strategy, and performance analytics services."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from backend.app.core.schemas import ModelTrainingRequest, ModelTrainingResponse, StrategyOptimizationRequest, TelemetryComparisonRequest, TelemetryComparisonResponse, TelemetryDerivedFrame, TelemetryStreamPacket
from telemetry.engine.config import get_runtime_config
from telemetry.engine.math_ops import F1TelemetryEngine
from telemetry.engine.pipeline import F1DualIngestionEngine, get_telemetry_broker
from telemetry.engine.spatial_maps import F1SpatialMappingEngine
from telemetry.engine.strategy_sim import F1StrategySimulator
from models.inference.tyre_model import F1TyrePredictor

app = FastAPI(
    title="MISSION CONTROL | TRK-72",
    version="3.0.0",
    description="Telemetry and performance analytics gateway for Formula 1 operations.",
)

runtime_config = get_runtime_config()
telemetry_engine = get_telemetry_broker()
spatial_engine = F1SpatialMappingEngine()
tyre_predictor = F1TyrePredictor()


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {
        "status": "ONLINE",
        "data_mode": runtime_config.data_mode,
        "subsystems": {"database": "EXPECTED", "ml_inference": "READY", "telemetry": "READY"},
    }


@app.post("/api/v1/strategy/optimize")
async def optimize_race_strategy(payload: StrategyOptimizationRequest) -> dict[str, object]:
    try:
        solver = F1StrategySimulator(
            total_race_laps=payload.total_laps,
            pit_loss_seconds=payload.pit_loss_seconds,
            fuel_burn_kg_per_lap=payload.fuel_burn_kg_per_lap,
            track_evolution_per_lap=payload.track_evolution_per_lap,
        )
        best_candidate = solver.evaluate_grid_strategies(
            initial_fuel=payload.initial_fuel,
            track_temp=payload.track_temp_c,
            candidate_compound_sets=tuple(tuple(candidate) for candidate in payload.compound_matrix),
        )
        pit_lap, compound_sequence, total_time = solver.evaluate_one_stop_strategies(
            initial_fuel=payload.initial_fuel,
            track_temp=payload.track_temp_c,
        )
        return {
            "success": True,
            "best_strategy": {
                "stops": best_candidate.stops,
                "pit_laps": list(best_candidate.pit_laps),
                "compounds": list(best_candidate.compounds),
                "projected_race_duration_seconds": round(best_candidate.total_race_time, 3),
            },
            "optimal_strategy": {
                "pit_lap": pit_lap,
                "compound_sequence": compound_sequence,
                "projected_race_duration_seconds": round(total_time, 3),
                "formatted_pace": f"{int(total_time // 60)}m {round(total_time % 60, 2)}s",
            },
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Strategic engine solver fault: {error}") from error


@app.post("/api/v1/models/train", response_model=ModelTrainingResponse)
async def train_model(payload: ModelTrainingRequest) -> ModelTrainingResponse:
    try:
        if payload.training_mode == "historical":
            synthetic_dataset = tyre_predictor.generate_synthetic_telemetry_dataset(samples=1200)
            rmse, r2 = tyre_predictor.retrain_from_historical_session(synthetic_dataset)
            return ModelTrainingResponse(success=True, training_mode=payload.training_mode, rmse=rmse, r2_score=r2, records_used=len(synthetic_dataset))

        synthetic_dataset = tyre_predictor.generate_synthetic_telemetry_dataset(samples=2400)
        rmse, r2 = tyre_predictor.train_model(synthetic_dataset)
        return ModelTrainingResponse(success=True, training_mode=payload.training_mode, rmse=rmse, r2_score=r2, records_used=len(synthetic_dataset))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model training fault: {error}") from error


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
        response = TelemetryComparisonResponse(
            success=True,
            session_name=metadata["session_name"],
            aligned_points=int(len(aligned_telemetry)),
            delta_end_s=float(delta_profile["delta_time_s"].iloc[-1]),
            preview_rows=aligned_telemetry.head(5).to_dict(orient="records"),
            driver_metadata=metadata,
        )
        return response.model_dump()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Telemetry comparison fault: {error}") from error


def _frame_to_packet(frame_row, partner_row, session_name: str, distance_index: int, delta_time_s: float) -> TelemetryStreamPacket:
    driver_one_frame = TelemetryDerivedFrame(
        distance_m=float(frame_row["Distance"]),
        speed_kph=float(frame_row["d1_Speed"]),
        throttle_pct=float(frame_row["d1_Throttle"]),
        brake_active=int(frame_row["d1_Brake"]),
        gear=int(frame_row["d1_Gear"]),
        rpm=int(frame_row["d1_RPM"]),
        x_coord=float(frame_row["d1_X"]),
        y_coord=float(frame_row["d1_Y"]),
        z_coord=float(frame_row.get("d1_Z", 0.0)),
        longitudinal_g=float(F1TelemetryEngine.calculate_longitudinal_g_force([frame_row["d1_Speed"]], [frame_row["Distance"]])[0]),
        lateral_g=float(F1TelemetryEngine.calculate_lateral_g_force([frame_row["d1_X"]], [frame_row["d1_Y"]], [frame_row["d1_Speed"]])[0]),
        throttle_smoothness=float(F1TelemetryEngine.calculate_throttle_derivative([frame_row["d1_Throttle"]], [frame_row["Distance"]], [frame_row["d1_Speed"]])[0]),
        braking_zone=int(frame_row["d1_Brake"] > 0),
        mguk_output_kw=float(frame_row.get("d1_MGUK_Output_kW", 0.0)),
        battery_soc_pct=float(frame_row.get("d1_Battery_SoC_pct", 100.0)),
        derating_active=int(frame_row.get("d1_Derating_Active", 0)),
        aero_mode=str(frame_row.get("d1_Aero_Mode", "Z")),
        aero_switch=int(frame_row.get("d1_Aero_Switch", 0)),
        boost_mode_active=int(frame_row.get("d1_Boost_Mode_Active", 0)),
        overtake_mode_active=int(frame_row.get("d1_Overtake_Mode_Active", 0)),
        electric_energy_kwh=float(frame_row.get("d1_Electric_Energy_kWh", 0.0)),
    )
    driver_two_frame = TelemetryDerivedFrame(
        distance_m=float(partner_row["Distance"]),
        speed_kph=float(partner_row["d2_Speed"]),
        throttle_pct=float(partner_row["d2_Throttle"]),
        brake_active=int(partner_row["d2_Brake"]),
        gear=int(partner_row["d2_Gear"]),
        rpm=int(partner_row["d2_RPM"]),
        x_coord=float(partner_row["d2_X"]),
        y_coord=float(partner_row["d2_Y"]),
        z_coord=float(partner_row.get("d2_Z", 0.0)),
        longitudinal_g=float(F1TelemetryEngine.calculate_longitudinal_g_force([partner_row["d2_Speed"]], [partner_row["Distance"]])[0]),
        lateral_g=float(F1TelemetryEngine.calculate_lateral_g_force([partner_row["d2_X"]], [partner_row["d2_Y"]], [partner_row["d2_Speed"]])[0]),
        throttle_smoothness=float(F1TelemetryEngine.calculate_throttle_derivative([partner_row["d2_Throttle"]], [partner_row["Distance"]], [partner_row["d2_Speed"]])[0]),
        braking_zone=int(partner_row["d2_Brake"] > 0),
        mguk_output_kw=float(partner_row.get("d2_MGUK_Output_kW", 0.0)),
        battery_soc_pct=float(partner_row.get("d2_Battery_SoC_pct", 100.0)),
        derating_active=int(partner_row.get("d2_Derating_Active", 0)),
        aero_mode=str(partner_row.get("d2_Aero_Mode", "X")),
        aero_switch=int(partner_row.get("d2_Aero_Switch", 0)),
        boost_mode_active=int(partner_row.get("d2_Boost_Mode_Active", 0)),
        overtake_mode_active=int(partner_row.get("d2_Overtake_Mode_Active", 0)),
        electric_energy_kwh=float(partner_row.get("d2_Electric_Energy_kWh", 0.0)),
    )
    return TelemetryStreamPacket(
        timestamp=datetime.now(tz=timezone.utc),
        driver_one=driver_one_frame,
        driver_two=driver_two_frame,
        delta_time_s=float(delta_time_s),
        session_name=session_name,
        regulation_context="2026-Hybrid",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        query_params = dict(websocket.query_params)
        year = int(query_params.get("year", 2024))
        round_id = query_params.get("round_id", "Bahrain")
        session_code = query_params.get("session_code", "Q")
        driver_one = query_params.get("driver_one", "VER")
        driver_two = query_params.get("driver_two", "LEC")
        sample_size = max(1, min(int(query_params.get("sample_size", 120)), 240))

        aligned_telemetry, metadata = telemetry_engine.fetch_comparison_dataset(year, round_id, session_code, driver_one, driver_two)
        aligned_telemetry = F1TelemetryEngine.calculate_2026_regulation_channels(
            aligned_telemetry,
            prefix="d1_",
            distance_column="Distance",
            speed_column="d1_Speed",
            throttle_column="d1_Throttle",
            brake_column="d1_Brake",
            x_column="d1_X",
            y_column="d1_Y",
        )
        aligned_telemetry = F1TelemetryEngine.calculate_2026_regulation_channels(
            aligned_telemetry,
            prefix="d2_",
            distance_column="Distance",
            speed_column="d2_Speed",
            throttle_column="d2_Throttle",
            brake_column="d2_Brake",
            x_column="d2_X",
            y_column="d2_Y",
        )
        delta_profile = F1TelemetryEngine.compute_delta_time(aligned_telemetry)
        sample_step = max(len(aligned_telemetry) // sample_size, 1)

        for index in range(0, len(aligned_telemetry), sample_step):
            try:
                incoming_message = await asyncio.wait_for(websocket.receive_text(), timeout=0.0)
                if incoming_message.strip().lower() in {"stop", "close"}:
                    break
            except asyncio.TimeoutError:
                pass

            partner_index = min(index, len(aligned_telemetry) - 1)
            packet = _frame_to_packet(
                aligned_telemetry.iloc[index],
                aligned_telemetry.iloc[partner_index],
                metadata["session_name"],
                index,
                delta_profile.iloc[index]["delta_time_s"],
            )
            await websocket.send_json(packet.model_dump(mode="json"))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return
    except Exception as error:
        await websocket.close(code=1011, reason=str(error))