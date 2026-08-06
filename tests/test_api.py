from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint_reports_online() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ONLINE"
    assert "data_mode" in payload


def test_strategy_optimize_returns_strategy() -> None:
    response = client.post("/api/v1/strategy/optimize", json={"total_laps": 57, "initial_fuel": 100.0, "pit_loss_seconds": 23.0, "track_temp_c": 35.0, "fuel_burn_kg_per_lap": 1.6, "track_evolution_per_lap": 0.0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "best_strategy" in payload


def test_model_training_endpoint_returns_metrics() -> None:
    response = client.post("/api/v1/models/train", json={"training_mode": "synthetic", "session_context": None})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["records_used"] > 0


def test_telemetry_comparison_returns_preview() -> None:
    response = client.post("/api/v1/telemetry/comparison", json={"season_year": 2024, "round_id": "Bahrain", "session_code": "Q", "driver_one": "VER", "driver_two": "LEC"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["aligned_points"] > 0


def test_websocket_telemetry_handshake_streams_packets() -> None:
    with client.websocket_connect("/ws/telemetry?data_mode=OFFLINE&sample_size=2&year=2024&round_id=Bahrain&session_code=Q&driver_one=VER&driver_two=LEC") as websocket:
        first_packet = websocket.receive_json()

    assert first_packet["session_name"].startswith("2024")
    assert "driver_one" in first_packet