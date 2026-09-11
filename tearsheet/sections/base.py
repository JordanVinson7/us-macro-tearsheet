"""Shared helpers for section builders.

A section builder takes the collected :class:`~tearsheet.data.bundle.DataBundle`
and returns a :class:`~tearsheet.models.SectionResult` whose ``context`` is
handed straight to that section's Jinja partial. Builders never raise: the
pipeline's :func:`~tearsheet.pipeline.build_section` wrapper converts anything
that escapes into a renderable "data unavailable" placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from tearsheet.analytics import returns as ret
from tearsheet.analytics import volatility as vol
from tearsheet.charts.template import figure_to_html
from tearsheet.config import Config, Instrument
from tearsheet.models import SourceRef

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle


@dataclass
class PerformanceRow:
    """One row of a performance table.

    Attributes:
        label: Display name.
        ticker: Yahoo symbol.
        last: Latest close.
        returns: Percentage returns keyed by window name.
        from_high: Percentage below the trailing high (negative).
        from_ma: Percentage from each moving average, keyed by window length.
        volatility: Annualised realised volatility in percent.
        zscore: Z-score of the latest daily return.
        available: False when no source could supply the instrument.
        source: Provenance, so a fallback-sourced row can be marked.
    """

    label: str
    ticker: str
    last: float | None = None
    returns: dict[str, float | None] = field(default_factory=dict)
    from_high: float | None = None
    from_ma: dict[int, float | None] = field(default_factory=dict)
    volatility: float | None = None
    zscore: float | None = None
    available: bool = True
    source: SourceRef | None = None

    @property
    def is_fallback(self) -> bool:
        """Whether this row came from a second-choice source."""
        return bool(self.source and self.source.is_fallback)


def close_series(bundle: DataBundle, ticker: str) -> pd.Series | None:
    """The close series for a ticker, or None when unavailable."""
    frame = bundle.prices.frames.get(ticker)
    if frame is None or "close" not in frame.columns or frame.empty:
        return None
    return frame["close"]


def build_performance_row(
    instrument: Instrument, bundle: DataBundle, config: Config
) -> PerformanceRow:
    """Compute every metric in the performance table for one instrument."""
    analytics = config.analytics
    prices = close_series(bundle, instrument.ticker)

    if prices is None:
        return PerformanceRow(label=instrument.name, ticker=instrument.ticker, available=False)

    suspect = bundle.prices.suspect_returns.get(instrument.ticker)
    windows = ret.window_returns(prices, analytics.return_windows, suspect)
    windows["YTD"] = ret.ytd_return(prices, bundle.session, suspect)

    return PerformanceRow(
        label=instrument.name,
        ticker=instrument.ticker,
        last=float(prices.iloc[-1]),
        returns=windows,
        from_high=ret.drawdown_from_high(prices, analytics.zscore_lookback),
        from_ma={w: ret.distance_from_ma(prices, w) for w in analytics.moving_averages},
        volatility=vol.latest_realised_volatility(
            prices,
            analytics.realised_vol_window,
            analytics.annualisation_factor,
            suspect,
        ),
        zscore=ret.return_zscore(
            prices, analytics.zscore_lookback, suspect, analytics.min_observations
        ),
        source=bundle.prices.sources.get(instrument.ticker),
    )


def build_performance_rows(
    instruments: list[Instrument], bundle: DataBundle, config: Config
) -> list[PerformanceRow]:
    """Build performance rows for a group of instruments."""
    return [build_performance_row(i, bundle, config) for i in instruments]


def render_chart(figure: go.Figure, section: str, name: str, height: int = 320) -> str:
    """Serialise a figure with a stable, section-scoped DOM id.

    The id is what the PDF step waits on to confirm every chart has drawn.
    """
    # Small multiples compute their own height from the number of panels.
    height = int(getattr(figure, "_tearsheet_height", height) or height)
    return figure_to_html(figure, f"chart-{section}-{name}", height=height)


def source_note(*sources: SourceRef | None) -> list[SourceRef]:
    """De-duplicate provenance references for a section footer.

    Returns SourceRef objects rather than strings so the footer can still tell
    whether a fallback was used; rendering to text happens in the template.
    """
    seen: list[SourceRef] = []
    for source in sources:
        if source is not None and source not in seen:
            seen.append(source)
    return seen
