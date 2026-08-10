"""MISSION CONTROL | TRK-72 dashboard."""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import httpx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import websockets

import sys
from pathlib import Path

# Ensure the project root is on sys.path so local packages (like `telemetry`) are
# importable when Streamlit changes the working directory/environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetry.engine.math_ops import F1TelemetryEngine
from telemetry.engine.pipeline import F1DualIngestionEngine
from telemetry.engine.spatial_maps import F1SpatialMappingEngine
from telemetry.engine.strategy_sim import F1StrategySimulator


TEAM_COLORS: dict[str, str] = {
    "RED BULL RACING": "#3671C6",
    "RED BULL": "#3671C6",
    "SCUDERIA FERRARI": "#F10808",
    "FERRARI": "#F10808",
    "MERCEDES": "#00A19B",
    "MCLAREN": "#FF8000",
    "ASTON MARTIN": "#006F62",
    "ALPINE": "#2293D1",
    "WILLIAMS": "#64C4FF",
    "RB": "#5E8FAA",
    "SAUBER": "#52E252",
    "HAAS": "#B6BABD",
    "SYN": "#9D9D9D",
}


st.set_page_config(
    page_title="MISSION CONTROL | TRK-72",
    page_icon="TRK-72",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --canvas-black: #0B0C10;
        --card-charcoal: #1F2833;
        --teal: #66FCF1;
        --text: #FFFFFF;
        --muted: #45A29E;
        --red: #F10808;
    }
    .stApp {
        background: radial-gradient(circle at top, #12161d 0%, var(--canvas-black) 52%);
        color: var(--text);
        font-family: "JetBrains Mono", "Roboto Mono", Consolas, monospace;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11161d 0%, #161d26 100%);
        border-right: 1px solid rgba(102, 252, 241, 0.18);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(31, 40, 51, 0.96) 0%, rgba(17, 22, 29, 0.98) 100%);
        border: 1px solid rgba(102, 252, 241, 0.18);
        border-radius: 4px;
        padding: 0.65rem 0.85rem;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"] {
        font-family: "JetBrains Mono", "Roboto Mono", Consolas, monospace !important;
        white-space: nowrap;
    }
    .panel {
        background: rgba(31, 40, 51, 0.92);
        border: 1px solid rgba(102, 252, 241, 0.16);
        border-radius: 4px;
        padding: 0.75rem;
    }
    h1, h2, h3, h4, p, label, span, div {
        font-family: "JetBrains Mono", "Roboto Mono", Consolas, monospace !important;
    }
    .stPlotlyChart {
        border: 1px solid rgba(102, 252, 241, 0.14);
        border-radius: 4px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@lru_cache(maxsize=1)
def get_engines() -> tuple[F1DualIngestionEngine, F1SpatialMappingEngine, F1StrategySimulator]:
    ingestion_engine = F1DualIngestionEngine()
    map_engine = F1SpatialMappingEngine()
    strategy_engine = F1StrategySimulator()
    return ingestion_engine, map_engine, strategy_engine


def _resolve_api_base_url() -> str:
    raw_value = os.getenv("F1_API_BASE_URL", "http://localhost:8000").strip().rstrip("/")
    return raw_value or "http://localhost:8000"


async def _fetch_ws_snapshot(api_base_url: str, query_params: dict[str, str], max_packets: int = 1) -> list[dict[str, object]]:
    websocket_url = api_base_url.replace("http://", "ws://").replace("https://", "wss://")
    websocket_url = f"{websocket_url}/ws/telemetry?{urlencode(query_params)}"
    packets: list[dict[str, object]] = []
    async with websockets.connect(websocket_url) as websocket:
        while len(packets) < max_packets:
            packets.append(json.loads(await websocket.recv()))
    return packets


def _fetch_remote_comparison_data(
    api_base_url: str,
    season_year: int,
    round_id: str,
    session_code: str,
    driver_one: str,
    driver_two: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    response = httpx.post(
        f"{api_base_url}/api/v1/telemetry/comparison",
        json={
            "season_year": season_year,
            "round_id": round_id,
            "session_code": session_code,
            "driver_one": driver_one,
            "driver_two": driver_two,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    preview_rows = pd.DataFrame(payload["preview_rows"])
    metadata = payload["driver_metadata"]
    return preview_rows, metadata


def _fetch_remote_strategy_data(api_base_url: str, total_laps: int, initial_fuel: float, pit_loss: float, track_temp: float) -> dict[str, object]:
    response = httpx.post(
        f"{api_base_url}/api/v1/strategy/optimize",
        json={
            "total_laps": total_laps,
            "initial_fuel": initial_fuel,
            "pit_loss_seconds": pit_loss,
            "track_temp_c": track_temp,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _build_live_stream_packets(frame: pd.DataFrame, max_packets: int = 8, driver_one: str = "VER", driver_two: str = "LEC") -> list[dict[str, object]]:
    if frame.empty:
        return []

    count = max(1, min(int(max_packets), len(frame)))
    step = max(1, len(frame) // count)
    packets: list[dict[str, object]] = []

    for index in range(0, len(frame), step):
        if len(packets) >= count:
            break

        row = frame.iloc[index]
        packets.append(
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "session_name": "LIVE-FALLBACK",
                "driver_one": {
                    "driver_code": driver_one,
                    "distance_m": float(row["Distance"]),
                    "speed_kph": float(row["d1_Speed"]),
                    "throttle_pct": float(row["d1_Throttle"]),
                    "brake_active": int(row["d1_Brake"]),
                    "gear": int(row["d1_Gear"]),
                    "rpm": int(row["d1_RPM"]),
                    "x_coord": float(row["d1_X"]),
                    "y_coord": float(row["d1_Y"]),
                    "mguk_output_kw": 0.0,
                    "derating_active": False,
                    "aero_mode": "X",
                },
                "driver_two": {
                    "driver_code": driver_two,
                    "distance_m": float(row["Distance"]),
                    "speed_kph": float(row["d2_Speed"]),
                    "throttle_pct": float(row["d2_Throttle"]),
                    "brake_active": int(row["d2_Brake"]),
                    "gear": int(row["d2_Gear"]),
                    "rpm": int(row["d2_RPM"]),
                    "x_coord": float(row["d2_X"]),
                    "y_coord": float(row["d2_Y"]),
                    "mguk_output_kw": 0.0,
                    "derating_active": False,
                    "aero_mode": "Z",
                },
                "delta_time_s": 0.0,
                "regulation_context": "2026-Hybrid",
            }
        )

    return packets


def _build_live_stream_figure(packets: list[dict[str, object]], driver_one: str = "VER", driver_two: str = "LEC") -> go.Figure:
    distances = [float(packet["driver_one"]["distance_m"]) for packet in packets]
    d1_speeds = [float(packet["driver_one"]["speed_kph"]) for packet in packets]
    d2_speeds = [float(packet["driver_two"]["speed_kph"]) for packet in packets]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=distances, y=d1_speeds, mode="lines+markers", name=f"{driver_one} Speed", line=dict(color="#66FCF1", width=2.2)))
    figure.add_trace(go.Scatter(x=distances, y=d2_speeds, mode="lines+markers", name=f"{driver_two} Speed", line=dict(color="#F10808", width=2.0)))
    figure.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0B0C10",
        plot_bgcolor="#0B0C10",
        font=dict(color="#FFFFFF", family="JetBrains Mono, Consolas, monospace"),
        legend=dict(orientation="h", y=1.08, x=0.02),
        hovermode="x unified",
    )
    figure.update_yaxes(title_text="Speed (km/h)", gridcolor="#26313f", zeroline=False)
    figure.update_xaxes(title_text="Distance (m)", gridcolor="#26313f", zeroline=False)
    return figure


def _packet_to_frame(packet: dict[str, object]) -> pd.DataFrame:
    driver_one = packet["driver_one"]
    driver_two = packet["driver_two"]
    return pd.DataFrame(
        {
            "Distance": [float(driver_one["distance_m"]), float(driver_two["distance_m"])],
            "d1_Speed": [float(driver_one["speed_kph"]), float(driver_one["speed_kph"])],
            "d2_Speed": [float(driver_two["speed_kph"]), float(driver_two["speed_kph"])],
            "d1_Throttle": [float(driver_one["throttle_pct"]), float(driver_one["throttle_pct"])],
            "d2_Throttle": [float(driver_two["throttle_pct"]), float(driver_two["throttle_pct"])],
            "d1_Brake": [float(driver_one["brake_active"]), float(driver_one["brake_active"])],
            "d2_Brake": [float(driver_two["brake_active"]), float(driver_two["brake_active"])],
            "d1_Gear": [int(driver_one["gear"]), int(driver_one["gear"])],
            "d2_Gear": [int(driver_two["gear"]), int(driver_two["gear"])],
            "d1_RPM": [int(driver_one["rpm"]), int(driver_one["rpm"])],
            "d2_RPM": [int(driver_two["rpm"]), int(driver_two["rpm"])],
            "d1_X": [float(driver_one["x_coord"]), float(driver_one["x_coord"])],
            "d1_Y": [float(driver_one["y_coord"]), float(driver_one["y_coord"])],
            "d2_X": [float(driver_two["x_coord"]), float(driver_two["x_coord"])],
            "d2_Y": [float(driver_two["y_coord"]), float(driver_two["y_coord"])],
        }
    )


@st.cache_data(show_spinner=False)
def load_comparison_data(season_year: int, round_id: str, session_code: str, driver_one: str, driver_two: str) -> tuple[pd.DataFrame, dict[str, object]]:
    ingestion_engine, _, _ = get_engines()
    return ingestion_engine.fetch_comparison_dataset(season_year, round_id, session_code, driver_one, driver_two)


@st.cache_data(show_spinner=False)
def load_comparison_data_remote(api_base_url: str, season_year: int, round_id: str, session_code: str, driver_one: str, driver_two: str) -> tuple[pd.DataFrame, dict[str, object]]:
    return _fetch_remote_comparison_data(api_base_url, season_year, round_id, session_code, driver_one, driver_two)


@st.cache_data(show_spinner=False)
def load_strategy_data_remote(api_base_url: str, total_laps: int, initial_fuel: float, pit_loss: float, track_temp: float) -> dict[str, object]:
    return _fetch_remote_strategy_data(api_base_url, total_laps, initial_fuel, pit_loss, track_temp)


def _team_color(team_name: str | None, driver_code: str | None = None, fallback_index: int = 0) -> str:
    if team_name:
        normalized_team = team_name.upper().strip()
        if normalized_team in TEAM_COLORS:
            return TEAM_COLORS[normalized_team]
    fallback_palette = ["#66FCF1", "#F10808", "#9D9D9D", "#FFFFFF"]
    if driver_code and driver_code.upper() == "LEC":
        return "#F10808"
    return fallback_palette[fallback_index % len(fallback_palette)]


def _lttb_downsample(x_values: np.ndarray, y_values: np.ndarray, threshold: int) -> np.ndarray:
    if threshold >= len(x_values) or threshold < 3:
        return np.arange(len(x_values))

    sampled_indices = [0]
    bucket_size = (len(x_values) - 2) / (threshold - 2)
    a_index = 0

    for bucket_index in range(threshold - 2):
        bucket_start = int(np.floor((bucket_index + 1) * bucket_size)) + 1
        bucket_end = int(np.floor((bucket_index + 2) * bucket_size)) + 1
        bucket_end = min(bucket_end, len(x_values))
        next_bucket_start = int(np.floor((bucket_index + 2) * bucket_size)) + 1
        next_bucket_end = int(np.floor((bucket_index + 3) * bucket_size)) + 1
        next_bucket_end = min(next_bucket_end, len(x_values))

        if bucket_start >= bucket_end:
            continue

        bucket_x = x_values[bucket_start:bucket_end]
        bucket_y = y_values[bucket_start:bucket_end]

        if next_bucket_start >= next_bucket_end:
            next_bucket_mean_x = float(x_values[-1])
            next_bucket_mean_y = float(y_values[-1])
        else:
            next_bucket_mean_x = float(np.mean(x_values[next_bucket_start:next_bucket_end]))
            next_bucket_mean_y = float(np.mean(y_values[next_bucket_start:next_bucket_end]))

        ax = float(x_values[a_index])
        ay = float(y_values[a_index])
        area = np.abs((ax - bucket_x) * (next_bucket_mean_y - ay) - (ax - next_bucket_mean_x) * (bucket_y - ay))
        selected_local_index = int(np.argmax(area)) + bucket_start
        sampled_indices.append(selected_local_index)
        a_index = selected_local_index

    sampled_indices.append(len(x_values) - 1)
    return np.unique(np.asarray(sampled_indices, dtype=int))


def _downsample_frame(frame: pd.DataFrame, x_column: str = "Distance", reference_column: str = "d1_Speed", max_points: int = 1200) -> pd.DataFrame:
    if frame.empty:
        return frame

    x_values = frame[x_column].to_numpy(dtype=float)
    reference_values = frame[reference_column].to_numpy(dtype=float)
    selected_indices = _lttb_downsample(x_values, reference_values, min(max_points, len(frame)))
    return frame.iloc[selected_indices].reset_index(drop=True)


def _enrich_physics_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched_frame = frame.copy()
    enriched_frame["d1_Longitudinal_G"] = F1TelemetryEngine.calculate_longitudinal_g_force(enriched_frame["d1_Speed"], enriched_frame["Distance"])
    enriched_frame["d2_Longitudinal_G"] = F1TelemetryEngine.calculate_longitudinal_g_force(enriched_frame["d2_Speed"], enriched_frame["Distance"])
    enriched_frame["d1_Throttle_Smoothness"] = F1TelemetryEngine.calculate_throttle_derivative(
        enriched_frame["d1_Throttle"], enriched_frame["Distance"], enriched_frame["d1_Speed"]
    )
    enriched_frame["d2_Throttle_Smoothness"] = F1TelemetryEngine.calculate_throttle_derivative(
        enriched_frame["d2_Throttle"], enriched_frame["Distance"], enriched_frame["d2_Speed"]
    )
    return enriched_frame


def build_telemetry_figure(df: pd.DataFrame, delta_profile: pd.DataFrame, metadata: dict[str, object]) -> go.Figure:
    d1_metadata = metadata.get("d1_metadata", {}) if isinstance(metadata, dict) else {}
    d2_metadata = metadata.get("d2_metadata", {}) if isinstance(metadata, dict) else {}
    d1_color = _team_color(d1_metadata.get("team"), d1_metadata.get("driver"), 0)
    d2_color = _team_color(d2_metadata.get("team"), d2_metadata.get("driver"), 1)

    figure = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.30, 0.16, 0.18, 0.18, 0.18],
    )

    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Speed"], name=f"{d1_metadata.get('driver', 'D1')} Speed", line=dict(color=d1_color, width=2.4)), row=1, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Speed"], name=f"{d2_metadata.get('driver', 'D2')} Speed", line=dict(color=d2_color, width=2.0)), row=1, col=1)

    figure.add_trace(go.Scatter(x=delta_profile["distance_meters"], y=delta_profile["delta_time_s"], name="Delta Time", line=dict(color="#FFFFFF", width=1.8, dash="dash")), row=2, col=1)

    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Throttle"], name=f"{d1_metadata.get('driver', 'D1')} Throttle", line=dict(color=d1_color, width=1.9)), row=3, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Throttle"], name=f"{d2_metadata.get('driver', 'D2')} Throttle", line=dict(color=d2_color, width=1.9)), row=3, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Brake"] * 100.0, name=f"{d1_metadata.get('driver', 'D1')} Brake", line=dict(color="#9D9D9D", width=1.2, dash="dot")), row=3, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Brake"] * 100.0, name=f"{d2_metadata.get('driver', 'D2')} Brake", line=dict(color="#B24C4C", width=1.2, dash="dot")), row=3, col=1)

    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Longitudinal_G"], name=f"{d1_metadata.get('driver', 'D1')} G", line=dict(color=d1_color, width=1.8)), row=4, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Longitudinal_G"], name=f"{d2_metadata.get('driver', 'D2')} G", line=dict(color=d2_color, width=1.8)), row=4, col=1)

    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Gear"], name=f"{d1_metadata.get('driver', 'D1')} Gear", line=dict(color=d1_color, width=2.0, shape="hv")), row=5, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Gear"], name=f"{d2_metadata.get('driver', 'D2')} Gear", line=dict(color=d2_color, width=2.0, shape="hv")), row=5, col=1)

    figure.update_layout(
        height=980,
        margin=dict(l=24, r=18, t=20, b=18),
        paper_bgcolor="#1F2833",
        plot_bgcolor="#0B0C10",
        font=dict(family="JetBrains Mono, Consolas, monospace", color="#FFFFFF"),
        legend=dict(orientation="h", y=1.02, x=0.01, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0B0C10", font=dict(color="#FFFFFF", family="JetBrains Mono, Consolas, monospace")),
    )
    figure.update_xaxes(title_text="Track Distance", row=5, col=1, gridcolor="#26313f", zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    figure.update_yaxes(title_text="Speed (km/h)", row=1, col=1, gridcolor="#26313f", zeroline=False)
    figure.update_yaxes(title_text="Δt (s)", row=2, col=1, gridcolor="#26313f", zeroline=False)
    figure.update_yaxes(title_text="Pedals %", row=3, col=1, gridcolor="#26313f", zeroline=False, range=[0, 100])
    figure.update_yaxes(title_text="G Force", row=4, col=1, gridcolor="#26313f", zeroline=False)
    figure.update_yaxes(title_text="Gear", row=5, col=1, gridcolor="#26313f", zeroline=False, range=[0.5, 8.5], tickmode="linear", dtick=1)
    return figure


def compute_sector_delta(delta_profile: pd.DataFrame, sector_index: int) -> float:
    if delta_profile.empty:
        return 0.0
    edges = np.linspace(float(delta_profile["distance_meters"].min()), float(delta_profile["distance_meters"].max()), 4)
    lower = edges[sector_index - 1]
    upper = edges[sector_index]
    sector = delta_profile[(delta_profile["distance_meters"] >= lower) & (delta_profile["distance_meters"] <= upper)]
    if sector.empty:
        return 0.0
    return float(sector["delta_time_s"].iloc[-1])


def main() -> None:
    st.title("MISSION CONTROL | TRK-72 ENGINEERING WORKSPACE")
    st.caption("High-density telemetry, strategy, and spatial analytics for multi-screen monitoring.")

    season_options = [2026, 2025, 2024, 2023]
    circuit_options = ["Bahrain", "Silverstone", "Monaco", "Spa-Francorchamps"]
    session_options = ["Q", "Race", "FP1", "FP2", "FP3"]
    driver_options = ["VER", "LEC", "NOR", "HAM", "SAI", "RUS"]

    with st.sidebar:
        st.subheader("SYSTEM LOGISTICS")
        execution_source = st.radio("Execution Source", ["Local Engine", "API Gateway", "WebSocket Snapshot"], index=0)
        api_base_url = st.text_input("API Base URL", value=_resolve_api_base_url())
        season_year = st.selectbox("Season", season_options, index=2)
        circuit_venue = st.selectbox("Circuit Venue", circuit_options, index=0)
        session_type = st.selectbox("Session Type", session_options, index=0)
        driver_one = st.selectbox("Driver 1", driver_options, index=0)
        driver_two = st.selectbox("Driver 2", driver_options, index=1)
        track_temp = st.slider("Track Temperature (°C)", 15.0, 55.0, 35.0, 0.5)
        cursor_distance = st.slider("Cursor Distance (m)", 0.0, 5412.0, 2706.0, 1.0)
        st.divider()
        st.subheader("Predictive Strategy Simulator")
        total_laps = st.slider("Race Laps", 10, 80, 57)
        initial_fuel = st.slider("Initial Fuel (kg)", 10.0, 115.0, 100.0, 0.5)
        pit_loss = st.slider("Pit Loss (s)", 15.0, 35.0, 23.0, 0.1)

    df, metadata = load_comparison_data(season_year, circuit_venue, session_type, driver_one, driver_two)

    remote_comparison_summary: dict[str, object] | None = None
    if execution_source == "API Gateway":
        try:
            remote_comparison_summary = load_comparison_data_remote(api_base_url, season_year, circuit_venue, session_type, driver_one, driver_two)[1]
        except Exception as error:
            st.warning(f"API comparison summary unavailable: {error}")
    elif execution_source == "WebSocket Snapshot":
        try:
            ws_query = {
                "year": str(season_year),
                "round_id": circuit_venue,
                "session_code": session_type,
                "driver_one": driver_one,
                "driver_two": driver_two,
                "sample_size": "2",
            }
            packets = asyncio.run(_fetch_ws_snapshot(api_base_url, ws_query, max_packets=1))
            if packets:
                packet = pd.json_normalize([packets[0]])
                st.sidebar.success("WebSocket telemetry stream reachable")
                st.sidebar.json(packet.iloc[0].to_dict())
        except Exception as error:
            st.warning(f"WebSocket snapshot failed, using local engine fallback: {error}")
            fallback_packets = _build_live_stream_packets(df, max_packets=3, driver_one=driver_one, driver_two=driver_two)
            if fallback_packets:
                st.sidebar.info("Live fallback packets generated from local telemetry sample")
                st.sidebar.json(fallback_packets[0])
                st.sidebar.plotly_chart(_build_live_stream_figure(fallback_packets, driver_one=driver_one, driver_two=driver_two), use_container_width=True, config={"displayModeBar": False})

    delta_profile = F1TelemetryEngine.compute_delta_time(df)
    physics_df = _enrich_physics_columns(df)
    downsampled_df = _downsample_frame(physics_df, max_points=1200)
    downsampled_delta = _downsample_frame(delta_profile.rename(columns={"distance_meters": "Distance", "delta_time_s": "d1_Speed"}), reference_column="d1_Speed", max_points=1200)
    downsampled_delta = downsampled_delta.rename(columns={"Distance": "distance_meters", "d1_Speed": "delta_time_s"})
    _, map_engine, _ = get_engines()
    if execution_source == "API Gateway":
        try:
            strategy_payload = load_strategy_data_remote(api_base_url, total_laps, initial_fuel, pit_loss, track_temp)
            best_strategy = strategy_payload.get("best_strategy", {})
            optimal_strategy = strategy_payload.get("optimal_strategy", {})
            pit_lap = int(optimal_strategy.get("pit_lap", -1))
            compound_sequence = str(optimal_strategy.get("compound_sequence", ""))
            total_time_seconds = float(optimal_strategy.get("projected_race_duration_seconds", 0.0))
        except Exception as error:
            st.warning(f"Strategy API unavailable, using local solver fallback: {error}")
            strategy_engine = F1StrategySimulator(total_race_laps=total_laps, pit_loss_seconds=pit_loss)
            pit_lap, compound_sequence, total_time_seconds = strategy_engine.evaluate_one_stop_strategies(initial_fuel, track_temp=track_temp)
            best_strategy = {"stops": 1, "pit_laps": [pit_lap], "compounds": compound_sequence.split(" -> ") if compound_sequence else [], "projected_race_duration_seconds": total_time_seconds}
    else:
        strategy_engine = F1StrategySimulator(total_race_laps=total_laps, pit_loss_seconds=pit_loss)
        pit_lap, compound_sequence, total_time_seconds = strategy_engine.evaluate_one_stop_strategies(initial_fuel, track_temp=track_temp)
        best_strategy = {"stops": 1, "pit_laps": [pit_lap], "compounds": compound_sequence.split(" -> ") if compound_sequence else [], "projected_race_duration_seconds": total_time_seconds}

    tyre_compound = metadata["d1_metadata"]["compound"] if isinstance(metadata, dict) else "SOFT"

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("SECTOR 1 DELTA", f"{compute_sector_delta(delta_profile, 1):+.3f} s", delta=f"{driver_one} vs {driver_two}")
    with metric_cols[1]:
        st.metric("SECTOR 2 DELTA", f"{compute_sector_delta(delta_profile, 2):+.3f} s", delta=f"{driver_one} vs {driver_two}")
    with metric_cols[2]:
        st.metric("TYRE COMPOUND", f"{tyre_compound}", delta=f"Stint target: Lap {pit_lap}")
    with metric_cols[3]:
        st.metric("TRACK TEMPERATURE", f"{track_temp:.1f} °C", delta=circuit_venue)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Telemetry Overview")
    telemetry_figure = build_telemetry_figure(downsampled_df, downsampled_delta, metadata)
    st.plotly_chart(telemetry_figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Spatial Vector Map")
    spatial_figure = map_engine.render_interactive_vector_map(df, selected_distance_m=cursor_distance)
    st.plotly_chart(spatial_figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    result_col, preview_col = st.columns([0.78, 1.22], gap="large")
    with result_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Strategy Output")
        st.write(f"Optimal pit lap: **{pit_lap}**")
        st.write(f"Compound sequence: **{compound_sequence}**")
        st.write(f"Projected duration: **{total_time_seconds:.3f} s**")
        st.write(f"Solver path: **{best_strategy.get('stops', 1)} stop(s)**")
        st.write(f"Session: **{metadata['session_name']}**")
        if remote_comparison_summary is not None:
            st.caption(f"API telemetry summary: {remote_comparison_summary.get('aligned_points', 0)} points, delta {remote_comparison_summary.get('delta_end_s', 0.0):+.3f} s")
        st.markdown('</div>', unsafe_allow_html=True)

    with preview_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Telemetry Preview")
        preview_frame = df[["Distance", "d1_Speed", "d2_Speed", "d1_Throttle", "d2_Throttle", "d1_Brake", "d2_Brake"]].head(12).copy()
        preview_frame.columns = ["Distance", f"{driver_one} Speed", f"{driver_two} Speed", f"{driver_one} Throttle", f"{driver_two} Throttle", f"{driver_one} Brake", f"{driver_two} Brake"]
        st.dataframe(preview_frame, use_container_width=True, height=260)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()