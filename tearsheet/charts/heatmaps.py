"""Heatmap builders for sector performance and cross-asset correlation.

Both use the same diverging scale, anchored so that zero always sits at the
neutral midpoint. Anchoring matters: with an auto-scaled range a day on which
every sector fell would paint the least-bad sector green.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tearsheet.charts.template import DIVERGING, MUTED, empty_figure


def performance_heatmap(
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: Sequence[Sequence[float | None]],
) -> go.Figure:
    """Sector returns across several windows.

    Args:
        row_labels: Sector names, top to bottom.
        column_labels: Window names, e.g. ``["1D", "1W", "1M", "YTD"]``.
        values: ``values[row][column]`` in percent; None renders blank.

    Returns:
        The heatmap figure.
    """
    if not row_labels or not column_labels:
        return empty_figure()

    matrix = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in values], dtype=float
    )
    if np.all(np.isnan(matrix)):
        return empty_figure()

    # Symmetric around zero so the colour always encodes the sign, never the
    # rank within the day.
    bound = float(np.nanmax(np.abs(matrix))) or 1.0
    text = np.where(
        np.isnan(matrix),
        "",
        np.char.add(np.where(matrix >= 0, "+", ""), np.round(matrix, 1).astype(str)),
    )

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=list(column_labels),
            y=list(row_labels),
            colorscale=DIVERGING,
            zmid=0,
            zmin=-bound,
            zmax=bound,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="%{y} · %{x}: %{z:+.2f}%<extra></extra>",
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    figure.update_layout(
        xaxis={"side": "top", "showgrid": False, "showline": False, "ticks": ""},
        yaxis={"autorange": "reversed", "showgrid": False, "showline": False, "ticks": ""},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="closest",
    )
    return figure


def correlation_heatmap(matrix: pd.DataFrame, window: int) -> go.Figure:
    """A correlation matrix with the upper triangle masked.

    A correlation matrix is symmetric, so showing both halves doubles the ink
    and halves the legibility. The diagonal is dropped for the same reason: a
    row of 1.00s conveys nothing.
    """
    if matrix.empty:
        return empty_figure()

    masked = matrix.to_numpy(dtype=float).copy()
    masked[np.triu_indices_from(masked, k=0)] = np.nan

    text = np.where(
        np.isnan(masked),
        "",
        np.char.add(np.where(masked >= 0, "+", ""), np.round(masked, 2).astype(str)),
    )

    figure = go.Figure(
        go.Heatmap(
            z=masked,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=DIVERGING,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} vs %{x}: %{z:+.2f}<extra></extra>",
            xgap=2,
            ygap=2,
            colorbar={
                "title": {"text": f"{window}d ρ", "font": {"size": 10, "color": MUTED}},
                "tickfont": {"size": 10, "color": MUTED},
                "thickness": 10,
                "len": 0.7,
                "outlinewidth": 0,
            },
        )
    )
    figure.update_layout(
        xaxis={"showgrid": False, "showline": False, "ticks": ""},
        yaxis={"autorange": "reversed", "showgrid": False, "showline": False, "ticks": ""},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        hovermode="closest",
    )
    return figure
