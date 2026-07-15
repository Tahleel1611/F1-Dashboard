"""MISSION CONTROL | TRK-72 dashboard."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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


@st.cache_data(show_spinner=False)
def load_comparison_data(season_year: int, round_id: str, session_code: str, driver_one: str, driver_two: str) -> tuple[pd.DataFrame, dict[str, object]]:
    ingestion_engine, _, _ = get_engines()
    return ingestion_engine.fetch_comparison_dataset(season_year, round_id, session_code, driver_one, driver_two)


def build_telemetry_figure(df: pd.DataFrame, delta_profile: pd.DataFrame) -> go.Figure:
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.46, 0.24, 0.30])

    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Speed"], name="VER Speed", line=dict(color="#66FCF1", width=2.2)), row=1, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Speed"], name="LEC Speed", line=dict(color="#F10808", width=2.0)), row=1, col=1)

    figure.add_trace(go.Scatter(x=delta_profile["distance_meters"], y=delta_profile["delta_time_s"], name="Delta Time", line=dict(color="#FFFFFF", width=1.8, dash="dash")), row=2, col=1)

    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Throttle"], name="VER Throttle", line=dict(color="#66FCF1", width=1.8)), row=3, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Throttle"], name="LEC Throttle", line=dict(color="#F10808", width=1.8)), row=3, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d1_Brake"] * 100.0, name="VER Brake", line=dict(color="#9D9D9D", width=1.2, dash="dot")), row=3, col=1)
    figure.add_trace(go.Scatter(x=df["Distance"], y=df["d2_Brake"] * 100.0, name="LEC Brake", line=dict(color="#B24C4C", width=1.2, dash="dot")), row=3, col=1)

    figure.update_layout(
        height=780,
        margin=dict(l=24, r=18, t=20, b=18),
        paper_bgcolor="#1F2833",
        plot_bgcolor="#0B0C10",
        font=dict(family="JetBrains Mono, Consolas, monospace", color="#FFFFFF"),
        legend=dict(orientation="h", y=1.02, x=0.01, font=dict(size=11)),
        hovermode="x unified",
    )
    figure.update_xaxes(title_text="Track Distance", row=3, col=1, gridcolor="#26313f", zeroline=False)
    figure.update_yaxes(title_text="Speed (km/h)", row=1, col=1, gridcolor="#26313f", zeroline=False)
    figure.update_yaxes(title_text="Δt (s)", row=2, col=1, gridcolor="#26313f", zeroline=False)
    figure.update_yaxes(title_text="Pedals %", row=3, col=1, gridcolor="#26313f", zeroline=False, range=[0, 100])
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
    delta_profile = F1TelemetryEngine.compute_delta_time(df)
    _, map_engine, _ = get_engines()
    strategy_engine = F1StrategySimulator(total_race_laps=total_laps, pit_loss_seconds=pit_loss)
    pit_lap, compound_sequence, total_time_seconds = strategy_engine.evaluate_one_stop_strategies(initial_fuel, track_temp=track_temp)
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
    telemetry_figure = build_telemetry_figure(df, delta_profile)
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
        st.write(f"Session: **{metadata['session_name']}**")
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