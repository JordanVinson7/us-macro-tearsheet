"""The shared Plotly template and chart helpers.

Every chart in the tear-sheet is built through this module so the whole page
reads as one document rather than a gallery of defaults. The palette, type and
grid weights match the stylesheet, and the same tokens are defined in both
places (see ``render/static/screen.css``).

Charts are serialised with ``include_plotlyjs=False``: the Plotly library is
loaded once from a CDN in the page head. Embedding it per chart would add
roughly 3MB *per figure* to a page that carries a dozen of them.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE_NAME = "tearsheet"

# --- Palette ----------------------------------------------------------------
# Mirrored in screen.css as CSS custom properties.
NAVY = "#14375f"
NAVY_LIGHT = "#2f6296"
INK = "#1c1e21"
MUTED = "#6b7280"
GRID = "#e7e9ee"
AXIS = "#c3c8d2"
PAPER = "#ffffff"

POSITIVE = "#0b7a5a"
NEGATIVE = "#b3261e"
NEUTRAL = "#8a8f98"

#: Ordered categorical palette. Navy leads so a single-series chart is navy.
CATEGORICAL = [
    NAVY,
    "#c2762a",
    "#3f8f7a",
    "#8c4a6b",
    NAVY_LIGHT,
    "#7a7f3f",
    "#a3543a",
    "#5c5f8f",
    "#2f7d95",
    "#96683f",
    "#4a7a4a",
]

#: Diverging scale for correlations and heatmaps: red (negative) through a
#: near-white midpoint to navy (positive). Chosen to stay legible in greyscale
#: when the sheet is printed.
DIVERGING = [
    [0.0, "#8c2318"],
    [0.25, "#d08a80"],
    [0.5, "#f4f5f7"],
    [0.75, "#7ea3c4"],
    [1.0, NAVY],
]

FONT_BODY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
FONT_SERIF = "'Source Serif 4', Georgia, 'Times New Roman', serif"

#: Plotly's default toolbar is visual noise in a research note.
CHART_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}


def build_template() -> go.layout.Template:
    """Construct the tear-sheet Plotly template."""
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": AXIS,
        "ticklen": 4,
        "tickfont": {"size": 11, "color": MUTED, "family": FONT_BODY},
        "title": {"font": {"size": 12, "color": MUTED, "family": FONT_BODY}},
        "automargin": True,
    }

    return go.layout.Template(
        layout={
            "colorway": CATEGORICAL,
            "font": {"family": FONT_BODY, "size": 12, "color": INK},
            "paper_bgcolor": PAPER,
            "plot_bgcolor": PAPER,
            "margin": {"l": 56, "r": 20, "t": 28, "b": 40},
            "hovermode": "x unified",
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": AXIS,
                "font": {"family": FONT_BODY, "size": 12, "color": INK},
            },
            "xaxis": axis,
            "yaxis": {**axis, "ticks": ""},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "left",
                "x": 0,
                "font": {"size": 11, "color": MUTED},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "colorscale": {"diverging": DIVERGING},
            "title": {
                "font": {"family": FONT_SERIF, "size": 14, "color": INK},
                "x": 0,
                "xanchor": "left",
            },
        }
    )


def register_template() -> None:
    """Register the template and make it Plotly's default."""
    pio.templates[TEMPLATE_NAME] = build_template()
    pio.templates.default = TEMPLATE_NAME


def change_colour(value: float | None) -> str:
    """Colour for a change figure, neutral when the value is unknown."""
    if value is None:
        return NEUTRAL
    return POSITIVE if value >= 0 else NEGATIVE


def figure_to_html(figure: go.Figure, div_id: str, height: int = 320) -> str:
    """Serialise a figure to an embeddable ``<div>``.

    Args:
        figure: The figure to render.
        div_id: Stable DOM id, used by the PDF step to confirm every chart has
            finished drawing before printing.
        height: Chart height in pixels.

    Returns:
        An HTML fragment with no ``<script src>`` for Plotly itself — the
        library is loaded once per page from a CDN.
    """
    figure.update_layout(height=height, autosize=True)
    return pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config=CHART_CONFIG,
        default_height=f"{height}px",
    )


def empty_figure(message: str = "Data unavailable") -> go.Figure:
    """A placeholder figure for a chart whose data could not be sourced.

    Rendering an empty frame with an explicit note is better than omitting the
    chart: the reader can see that something is meant to be there.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 13, "color": MUTED, "family": FONT_BODY},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, plot_bgcolor="#fbfbfc")
    return figure
