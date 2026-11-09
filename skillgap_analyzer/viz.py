from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass(frozen=True)
class Charts:
    coverage_bar: go.Figure
    skills_radar: go.Figure
    progression_line: go.Figure


def make_charts(*, matched: set[str], missing: set[str], extra: set[str], predicted_weeks: float) -> Charts:
    counts = pd.DataFrame(
        [
            {"category": "Matched", "count": len(matched)},
            {"category": "Missing", "count": len(missing)},
            {"category": "Extra", "count": len(extra)},
        ]
    )
    coverage_bar = px.bar(counts, x="category", y="count", color="category", text="count")
    coverage_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)

    # Radar: show proportions of matched vs missing (simple 2-axis, but still visual).
    total_target = max(1, len(matched) + len(missing))
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=[len(matched) / total_target, len(missing) / total_target],
            theta=["Matched", "Missing"],
            fill="toself",
            name="Target Coverage",
        )
    )
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=260)

    # Progression curve: logistic-ish growth to 1 over predicted weeks.
    horizon = max(4, int(math.ceil(predicted_weeks)))
    weeks = np.arange(0, horizon + 1)
    k = 6.0 / max(1.0, predicted_weeks)
    y = 1.0 / (1.0 + np.exp(-k * (weeks - predicted_weeks / 2.0)))
    progression = go.Figure()
    progression.add_trace(go.Scatter(x=weeks, y=y, mode="lines+markers", name="Readiness (0-1)"))
    progression.update_layout(
        xaxis_title="Weeks",
        yaxis_title="Estimated readiness",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
    )

    return Charts(coverage_bar=coverage_bar, skills_radar=radar, progression_line=progression)

