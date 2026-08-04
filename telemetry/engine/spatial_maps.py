"""Programmatic circuit mapping and delta-dominance sector visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class F1SpatialMappingEngine:
    @staticmethod
    def _dominance_color(is_driver_one_faster: bool) -> str:
        return "#66FCF1" if is_driver_one_faster else "#F10808"

    @staticmethod
    def generate_track_layout_sectors(df_synced: pd.DataFrame, total_sectors: int = 25) -> pd.DataFrame:
        processed_df = df_synced.copy().sort_values("Distance")
        sector_edges = np.linspace(float(processed_df["Distance"].min()), float(processed_df["Distance"].max()), total_sectors + 1)
        processed_df["sector_index"] = pd.cut(processed_df["Distance"], bins=sector_edges, labels=False, include_lowest=True)
        processed_df["d1_faster"] = processed_df["d1_Speed"] > processed_df["d2_Speed"]
        sector_dom = processed_df.groupby("sector_index")["d1_faster"].transform("mean")
        processed_df["dominance_color"] = np.where(sector_dom >= 0.5, "#66FCF1", "#F10808")
        processed_df["dominance_label"] = np.where(sector_dom >= 0.5, "Driver 1 faster", "Driver 2 faster")
        return processed_df

    @staticmethod
    def _nearest_row(frame: pd.DataFrame, distance_m: float) -> pd.Series:
        index = (frame["Distance"] - distance_m).abs().idxmin()
        return frame.loc[index]

    @staticmethod
    def _estimate_corner_indices(frame: pd.DataFrame, max_corners: int = 8) -> list[int]:
        x_coords = frame["d1_X"].to_numpy(dtype=float)
        y_coords = frame["d1_Y"].to_numpy(dtype=float)
        if x_coords.size < 5 or y_coords.size < 5:
            return []

        dx = np.gradient(x_coords)
        dy = np.gradient(y_coords)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.nan_to_num(np.abs(dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5), nan=0.0, posinf=0.0, neginf=0.0)

        candidate_indices = np.argsort(curvature)[::-1]
        selected_indices: list[int] = []
        minimum_separation = max(frame.shape[0] // (max_corners * 4), 25)

        for index in candidate_indices:
            if all(abs(index - chosen_index) >= minimum_separation for chosen_index in selected_indices):
                selected_indices.append(int(index))
            if len(selected_indices) >= max_corners:
                break

        return sorted(selected_indices)

    def render_interactive_vector_map(self, df_synced: pd.DataFrame, selected_distance_m: float | None = None) -> go.Figure:
        segmented_data = self.generate_track_layout_sectors(df_synced, total_sectors=25)
        fig = go.Figure()

        for sector_index in sorted(segmented_data["sector_index"].dropna().unique()):
            sector_frame = segmented_data[segmented_data["sector_index"] == sector_index]
            if sector_frame.empty:
                continue
            color = str(sector_frame["dominance_color"].iloc[0])
            fig.add_trace(
                go.Scattergl(
                    x=sector_frame["d1_X"],
                    y=sector_frame["d1_Y"],
                    mode="lines",
                    line=dict(color=color, width=4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for corner_number, row_index in enumerate(self._estimate_corner_indices(segmented_data), start=1):
            corner_row = segmented_data.iloc[row_index]
            fig.add_annotation(
                x=float(corner_row["d1_X"]),
                y=float(corner_row["d1_Y"]),
                text=f"Turn {corner_number}",
                showarrow=True,
                arrowhead=2,
                arrowwidth=1,
                arrowsize=1,
                arrowcolor="#FFFFFF",
                font=dict(family="JetBrains Mono", size=11, color="#FFFFFF"),
                bgcolor="rgba(11,12,16,0.75)",
                bordercolor="#66FCF1",
                borderpad=4,
            )

        if selected_distance_m is not None:
            marker_one = self._nearest_row(segmented_data, selected_distance_m)
            marker_two = self._nearest_row(segmented_data, selected_distance_m)
            fig.add_trace(
                go.Scatter(
                    x=[marker_one["d1_X"]],
                    y=[marker_one["d1_Y"]],
                    mode="markers",
                    marker=dict(size=11, color="#66FCF1", line=dict(color="#FFFFFF", width=1)),
                    name="Driver 1",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[marker_two["d2_X"]],
                    y=[marker_two["d2_Y"]],
                    mode="markers",
                    marker=dict(size=11, color="#F10808", line=dict(color="#FFFFFF", width=1)),
                    name="Driver 2",
                )
            )

        fig.update_layout(
            title=dict(text="TRACK DOMINANCE HEATMAP | DELTA-VECTOR OVERLAY", font=dict(family="JetBrains Mono", size=16, color="#66FCF1")),
            paper_bgcolor="#1F2833",
            plot_bgcolor="#0B0C10",
            height=620,
            margin=dict(l=24, r=24, t=56, b=24),
            showlegend=True,
            legend=dict(font=dict(family="JetBrains Mono", color="#FFFFFF"), orientation="h", y=1.04),
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
        )
        return fig


if __name__ == "__main__":
    theta = np.linspace(0, 2 * np.pi, 1000)
    mock_distance = np.linspace(0, 5412, 1000)
    mock_df = pd.DataFrame(
        {
            "Distance": mock_distance,
            "d1_X": 1000 * np.cos(theta),
            "d1_Y": 600 * np.sin(theta),
            "d2_X": 1000 * np.cos(theta + 0.06),
            "d2_Y": 600 * np.sin(theta + 0.03),
            "d1_Speed": 250 + 60 * np.sin(theta * 4),
            "d2_Speed": 248 + 60 * np.cos(theta * 4),
        }
    )
    engine = F1SpatialMappingEngine()
    print(engine.render_interactive_vector_map(mock_df).to_dict().keys())