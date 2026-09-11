"""Section 4 — equities.

Performance table, sector heatmap, breadth, relative-value ratios, a
year-to-date chart and the previous session's intraday panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tearsheet.analytics import returns as ret
from tearsheet.analytics import volatility as vol
from tearsheet.charts import heatmaps, timeseries
from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.sections.base import (
    build_performance_rows,
    close_series,
    render_chart,
    source_note,
)

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the equities section."""
    config = settings.config
    analytics = config.analytics
    result = SectionResult(key="equities", title="Equities")

    index_rows = build_performance_rows(config.universe.indices, bundle, config)
    sector_rows = build_performance_rows(config.universe.sectors, bundle, config)

    if not any(r.available for r in index_rows + sector_rows):
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No equity price history was available."
        return result

    charts: dict[str, str] = {}

    # --- sector heatmap ----------------------------------------------------
    windows = config.display.heatmap_windows
    available_sectors = [r for r in sector_rows if r.available]
    charts["heatmap"] = render_chart(
        heatmaps.performance_heatmap(
            [r.label for r in available_sectors],
            windows,
            [[r.returns.get(w) for w in windows] for r in available_sectors],
        ),
        "equities",
        "heatmap",
        height=max(260, 28 * len(available_sectors) + 70),
    )

    # --- breadth -----------------------------------------------------------
    sector_series = {
        i.ticker: s
        for i in config.universe.sectors
        if (s := close_series(bundle, i.ticker)) is not None
    }
    breadth = {
        window: ret.share_above_ma(sector_series, window) for window in analytics.moving_averages
    }

    # --- year to date ------------------------------------------------------
    start = pd.Timestamp(year=bundle.session.year - 1, month=12, day=31)
    ytd_series = {
        i.label: s
        for i in config.universe.indices
        if (s := close_series(bundle, i.ticker)) is not None
    }
    charts["ytd"] = render_chart(
        timeseries.rebased_chart(ytd_series, start), "equities", "ytd", height=340
    )

    # --- relative-value ratios ---------------------------------------------
    ratio_series: dict[str, pd.Series] = {}
    for ratio in config.universe.ratios:
        numerator = close_series(bundle, ratio.numerator)
        denominator = close_series(bundle, ratio.denominator)
        if numerator is None or denominator is None:
            continue
        aligned = pd.DataFrame({"n": numerator, "d": denominator}).dropna()
        if aligned.empty:
            continue
        ratio_values = aligned["n"] / aligned["d"]
        ratio_series[f"{ratio.key} — {ratio.name}"] = ratio_values / ratio_values.iloc[0] * 100.0

    charts["ratios"] = render_chart(
        timeseries.line_chart(ratio_series, yaxis_title="Indexed to 100", days=365),
        "equities",
        "ratios",
        height=320,
    )

    # --- previous-session intraday panel -----------------------------------
    intraday_stats = []
    if bundle.intraday:
        charts["intraday"] = render_chart(
            timeseries.intraday_chart(bundle.intraday), "equities", "intraday", height=320
        )
        for ticker, bars in bundle.intraday.items():
            stats = vol.intraday_statistics(bars)
            if not stats:
                continue
            intraday_stats.append(
                {
                    "ticker": ticker,
                    "vwap": stats.get("vwap"),
                    "close": stats.get("close"),
                    "session_return": stats.get("session_return"),
                    "first_hour": stats.get("first_hour_return"),
                    "last_hour": stats.get("last_hour_return"),
                    "high_time": _clock(stats.get("high_time")),
                    "low_time": _clock(stats.get("low_time")),
                    "volatility": stats.get("realised_volatility"),
                }
            )
    else:
        result.warnings.append("Intraday minute bars were unavailable.")

    result.context = {
        "index_rows": index_rows,
        "sector_rows": sector_rows,
        "return_columns": ["1D", "1W", "1M", "3M", "YTD", "1Y"],
        "moving_averages": analytics.moving_averages,
        "breadth": breadth,
        "intraday_stats": intraday_stats,
        "intraday_session": bundle.session,
        "charts": charts,
        "vol_window": analytics.realised_vol_window,
    }
    result.sources = source_note(
        *[r.source for r in index_rows + sector_rows if r.available],
        *bundle.sources.get("intraday", []),
    )
    result.notes = [
        f"Returns are simple percentage changes over {analytics.return_windows['1W']}, "
        f"{analytics.return_windows['1M']}, {analytics.return_windows['3M']} and "
        f"{analytics.return_windows['1Y']} trading days; YTD is measured from the "
        "final close of the previous calendar year.",
        f"Volatility is the annualised standard deviation of daily returns over "
        f"{analytics.realised_vol_window} sessions.",
        f"The z-score compares the latest one-day return with its trailing "
        f"{analytics.zscore_lookback}-session distribution.",
        "Prices are dividend-adjusted, so multi-month returns are total returns.",
    ]
    result.status = (
        SectionStatus.OK
        if all(r.available for r in index_rows + sector_rows) and bundle.intraday
        else SectionStatus.PARTIAL
    )
    return result


def _clock(timestamp: object) -> str | None:
    """Format a timestamp as ``HH:MM`` in the session's own timezone."""
    if timestamp is None or not hasattr(timestamp, "strftime"):
        return None
    return timestamp.strftime("%H:%M")
