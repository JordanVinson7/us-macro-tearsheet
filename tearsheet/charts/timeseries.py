"""Line, area and small-multiple chart builders.

Each function takes plain pandas objects and returns a Plotly figure. None of
them touch configuration or the data bundle, so a chart can be built and
eyeballed in isolation from a notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tearsheet.charts.template import (
    AXIS,
    CATEGORICAL,
    GRID,
    MUTED,
    NAVY,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    change_colour,
    empty_figure,
)

#: Shared axis format for daily date axes.
_DATE_AXIS = {"tickformat": "%b %y", "showgrid": False}


def _trim(series: pd.Series, days: int | None) -> pd.Series:
    """Restrict a series to its trailing ``days`` calendar days."""
    values = series.dropna().sort_index()
    if days is None or values.empty:
        return values
    cutoff = values.index[-1] - pd.Timedelta(days=days)
    return values.loc[cutoff:]


def line_chart(
    series_by_label: Mapping[str, pd.Series],
    *,
    yaxis_title: str = "",
    days: int | None = None,
    percent: bool = False,
    zero_line: bool = False,
    fill: bool = False,
) -> go.Figure:
    """A multi-series line chart.

    Args:
        series_by_label: Series keyed by legend label.
        yaxis_title: Axis label.
        days: Trailing window to display.
        percent: Suffix values with ``%``.
        zero_line: Draw an emphasised line at zero, for spreads and z-scores.
        fill: Shade beneath a single series.

    Returns:
        The figure, or a placeholder when nothing could be plotted.
    """
    usable = {k: _trim(v, days) for k, v in series_by_label.items()}
    usable = {k: v for k, v in usable.items() if not v.empty}
    if not usable:
        return empty_figure()

    figure = go.Figure()
    for index, (label, values) in enumerate(usable.items()):
        colour = CATEGORICAL[index % len(CATEGORICAL)]
        figure.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(),
                name=label,
                mode="lines",
                line={"color": colour, "width": 1.8},
                fill="tozeroy" if fill and len(usable) == 1 else None,
                fillcolor="rgba(20,55,95,0.06)" if fill else None,
                hovertemplate=f"{label}: %{{y:.2f}}{'%' if percent else ''}<extra></extra>",
            )
        )

    if zero_line:
        figure.add_hline(y=0, line={"color": AXIS, "width": 1, "dash": "dot"})

    figure.update_layout(
        xaxis=_DATE_AXIS,
        yaxis={"title": yaxis_title, "ticksuffix": "%" if percent else ""},
        showlegend=len(usable) > 1,
    )
    return figure


def rebased_chart(
    series_by_label: Mapping[str, pd.Series], start: pd.Timestamp | None = None
) -> go.Figure:
    """Index several price series to 100 at a common start date.

    Rebasing is what makes a multi-asset chart readable: plotting the S&P at
    7,600 against the Russell at 2,900 on one axis compares nothing.
    """
    rebased: dict[str, pd.Series] = {}
    for label, series in series_by_label.items():
        values = series.dropna().sort_index()
        if start is not None:
            values = values.loc[start:]
        if values.empty or values.iloc[0] <= 0:
            continue
        rebased[label] = values / values.iloc[0] * 100.0

    if not rebased:
        return empty_figure()

    figure = line_chart(rebased, yaxis_title="Indexed to 100")
    figure.add_hline(y=100, line={"color": AXIS, "width": 1, "dash": "dot"})
    return figure


def curve_chart(curves: Mapping[str, Sequence]) -> go.Figure:
    """The Treasury curve today against earlier snapshots.

    The x-axis is log-scaled in tenor. A linear axis squeezes the entire front
    end — where most of the action usually is — into the leftmost few percent
    of the chart.
    """
    if not curves:
        return empty_figure()

    figure = go.Figure()
    styles = [
        {"width": 2.4, "dash": None, "color": NAVY},
        {"width": 1.6, "dash": "dash", "color": CATEGORICAL[1]},
        {"width": 1.4, "dash": "dot", "color": CATEGORICAL[2]},
        {"width": 1.2, "dash": "longdash", "color": MUTED},
    ]

    for index, (label, points) in enumerate(curves.items()):
        if not points:
            continue
        style = styles[index % len(styles)]
        figure.add_trace(
            go.Scatter(
                x=[p.tenor_years for p in points],
                y=[p.yield_pct for p in points],
                name=label,
                mode="lines+markers" if index == 0 else "lines",
                line={"color": style["color"], "width": style["width"], "dash": style["dash"]},
                marker={"size": 6, "color": style["color"]},
                customdata=[p.label for p in points],
                hovertemplate="%{customdata}: %{y:.2f}%<extra>" + label + "</extra>",
            )
        )

    tenors = [p.tenor_years for points in curves.values() for p in points]
    labels = {p.tenor_years: p.label for points in curves.values() for p in points}
    figure.update_layout(
        xaxis={
            "type": "log",
            "title": "Maturity",
            "tickvals": sorted(set(tenors)),
            "ticktext": [labels[t] for t in sorted(set(tenors))],
            "showgrid": True,
            "gridcolor": GRID,
        },
        yaxis={"title": "Yield", "ticksuffix": "%"},
        hovermode="closest",
    )
    return figure


def small_multiples(
    series_by_title: Mapping[str, pd.Series],
    columns: int = 4,
    days: int | None = None,
    row_height: int = 110,
) -> go.Figure:
    """A grid of sparkline-style panels, one per series.

    Used for the macro dashboard: sixteen indicators as full charts would fill
    three pages, whereas a grid conveys the shape of each at a glance.
    """
    usable = {k: _trim(v, days) for k, v in series_by_title.items()}
    usable = {k: v for k, v in usable.items() if len(v) > 1}
    if not usable:
        return empty_figure()

    rows = (len(usable) + columns - 1) // columns
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=list(usable),
        vertical_spacing=max(0.06, 0.30 / rows),
        horizontal_spacing=0.05,
    )

    for index, (title, values) in enumerate(usable.items()):
        row, column = divmod(index, columns)
        rising = float(values.iloc[-1]) >= float(values.iloc[0])
        colour = POSITIVE if rising else NEGATIVE
        figure.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(),
                mode="lines",
                line={"color": colour, "width": 1.4},
                showlegend=False,
                hovertemplate=f"{title}: %{{y:.2f}}<br>%{{x|%b %Y}}<extra></extra>",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.add_trace(
            go.Scatter(
                x=[values.index[-1]],
                y=[float(values.iloc[-1])],
                mode="markers",
                marker={"size": 5, "color": colour},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row + 1,
            col=column + 1,
        )

    figure.update_xaxes(showgrid=False, showticklabels=False, showline=False, ticks="")
    figure.update_yaxes(showgrid=False, showline=False, tickfont={"size": 9})
    for annotation in figure.layout.annotations:
        annotation.font.size = 10
        annotation.font.color = MUTED

    figure.update_layout(margin={"l": 40, "r": 12, "t": 24, "b": 12})
    figure._tearsheet_height = rows * row_height  # noqa: SLF001 — consumed by the caller
    return figure


def regime_chart(history: pd.Series, bands: Sequence) -> go.Figure:
    """The regime composite through time, with its bands shaded behind it."""
    values = history.dropna()
    if values.empty:
        return empty_figure()

    figure = go.Figure()
    lower = -4.0
    for band in bands:
        upper = min(float(band.max), 4.0)
        if upper <= lower:
            continue
        shade = {
            "positive": "rgba(11,122,90,0.06)",
            "neutral": "rgba(138,143,152,0.05)",
            "negative": "rgba(179,38,30,0.06)",
        }[band.tone]
        figure.add_hrect(y0=lower, y1=upper, fillcolor=shade, line_width=0, layer="below")
        lower = upper

    figure.add_trace(
        go.Scatter(
            x=values.index,
            y=values.to_numpy(),
            mode="lines",
            line={"color": NAVY, "width": 1.8},
            name="Composite",
            hovertemplate="%{y:+.2f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line={"color": AXIS, "width": 1, "dash": "dot"})
    figure.update_layout(
        xaxis=_DATE_AXIS,
        yaxis={"title": "Composite (z)", "range": [-3, 3]},
        showlegend=False,
    )
    return figure


def contribution_chart(components: Sequence) -> go.Figure:
    """Horizontal bars showing each component's contribution to the composite.

    This is what makes the composite auditable rather than a black box: the
    reader can see precisely which input is driving the headline number.
    """
    if not components:
        return empty_figure()

    ordered = list(reversed(components))
    values = [c.contribution for c in ordered]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[c.label for c in ordered],
            orientation="h",
            marker={"color": [change_colour(-v) for v in values]},
            customdata=[[c.zscore, c.weight * 100, c.raw_value] for c in ordered],
            hovertemplate=(
                "Contribution %{x:+.3f}<br>z-score %{customdata[0]:+.2f}"
                "<br>weight %{customdata[1]:.0f}%<br>level %{customdata[2]:,.2f}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line={"color": AXIS, "width": 1})
    figure.update_layout(
        xaxis={"title": "Contribution to composite", "zeroline": False},
        yaxis={"showgrid": False, "automargin": True},
        showlegend=False,
        hovermode="closest",
    )
    return figure


def dual_axis_chart(
    left: pd.Series,
    right: pd.Series,
    left_label: str,
    right_label: str,
    days: int | None = None,
) -> go.Figure:
    """Two series on independent axes, for comparing level against level.

    Used for copper/gold against the 10-year yield: the relationship is the
    point, and the units are not comparable.
    """
    left_values, right_values = _trim(left, days), _trim(right, days)
    if left_values.empty or right_values.empty:
        return empty_figure()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=left_values.index,
            y=left_values.to_numpy(),
            name=left_label,
            line={"color": NAVY, "width": 1.8},
            hovertemplate=f"{left_label}: %{{y:.3f}}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=right_values.index,
            y=right_values.to_numpy(),
            name=right_label,
            yaxis="y2",
            line={"color": CATEGORICAL[1], "width": 1.8, "dash": "dot"},
            hovertemplate=f"{right_label}: %{{y:.2f}}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis=_DATE_AXIS,
        # Plotly 7 removed the flat `titlefont` shorthand in favour of
        # `title.font`.
        yaxis={"title": {"text": left_label, "font": {"color": NAVY}}},
        yaxis2={
            "title": {"text": right_label, "font": {"color": CATEGORICAL[1]}},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
    )
    return figure


def intraday_chart(bars_by_ticker: Mapping[str, pd.DataFrame]) -> go.Figure:
    """Normalised intraday paths for the previous session.

    Each path is indexed to its own opening print, so four instruments on very
    different price scales can share one axis.
    """
    usable = {k: v for k, v in bars_by_ticker.items() if not v.empty and "close" in v.columns}
    if not usable:
        return empty_figure()

    figure = go.Figure()
    for index, (ticker, bars) in enumerate(usable.items()):
        closes = bars["close"]
        if closes.empty or float(closes.iloc[0]) <= 0:
            continue
        normalised = (closes / float(closes.iloc[0]) - 1.0) * 100.0
        figure.add_trace(
            go.Scatter(
                x=normalised.index,
                y=normalised.to_numpy(),
                name=ticker,
                mode="lines",
                line={"color": CATEGORICAL[index % len(CATEGORICAL)], "width": 1.6},
                hovertemplate=f"{ticker}: %{{y:+.2f}}%<extra></extra>",
            )
        )

    figure.add_hline(y=0, line={"color": AXIS, "width": 1, "dash": "dot"})
    figure.update_layout(
        xaxis={"tickformat": "%H:%M", "title": "Session time (ET)", "showgrid": False},
        yaxis={"title": "From open", "ticksuffix": "%"},
    )
    return figure


def bar_chart(labels: Sequence[str], values: Sequence[float], yaxis_title: str = "") -> go.Figure:
    """A vertical bar chart coloured by sign."""
    if not labels:
        return empty_figure()

    figure = go.Figure(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker={"color": [change_colour(v) for v in values]},
            hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
        )
    )
    figure.add_hline(y=0, line={"color": AXIS, "width": 1})
    figure.update_layout(
        xaxis={"showgrid": False},
        yaxis={"title": yaxis_title, "ticksuffix": "%"},
        showlegend=False,
        hovermode="closest",
    )
    return figure


def gauge_bar(value: float, minimum: float = -2.0, maximum: float = 2.0) -> go.Figure:
    """A single horizontal marker showing where a score sits on a scale."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[value],
            y=[0],
            mode="markers",
            marker={"size": 18, "color": NAVY, "symbol": "diamond"},
            hovertemplate="%{x:+.2f}<extra></extra>",
        )
    )
    figure.add_shape(
        type="line", x0=minimum, x1=maximum, y0=0, y1=0, line={"color": NEUTRAL, "width": 3}
    )
    figure.update_layout(
        xaxis={"range": [minimum, maximum], "showgrid": False, "zeroline": True},
        yaxis={"visible": False, "range": [-1, 1]},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 10, "b": 30},
    )
    return figure
