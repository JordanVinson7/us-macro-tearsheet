"""Section 6 — credit spreads.

High-yield and investment-grade option-adjusted spreads, with their level,
basis-point change, percentile and z-score against available history.

The lookback is clamped to what FRED actually holds. Several ICE BofA series
carry only about three years of history there, so asking for a five-year
percentile would silently produce a three-year one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tearsheet.analytics import rates as rt
from tearsheet.analytics import volatility as vol
from tearsheet.charts import timeseries
from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.sections.base import render_chart

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the credit section."""
    config = settings.config
    result = SectionResult(key="credit", title="Credit")

    rows = []
    series_by_label = {}

    for spec in config.fred.credit:
        series = bundle.fred.get(spec.id)
        if series is None or series.empty:
            result.warnings.append(f"Credit series {spec.id} was unavailable.")
            continue

        # Percentile and z-score are only as long as the data allows.
        available = len(series)
        lookback = min(config.analytics.percentile_lookback, available)
        metadata = bundle.fred_meta.get(spec.id)
        history_years = metadata.history_years if metadata else available / 252

        rows.append(
            {
                "name": spec.name,
                "label": spec.label,
                "series_id": spec.id,
                "level": float(series.iloc[-1]),
                "change_1d": rt.change_in_bp(series, 1),
                "change_1w": rt.change_in_bp(series, 5),
                "change_1m": rt.change_in_bp(series, 21),
                "percentile": vol.level_percentile(series, lookback),
                "zscore": vol.level_zscore(series, lookback),
                "lookback_days": lookback,
                "history_years": history_years,
                "observation_date": series.index[-1].date(),
            }
        )
        series_by_label[spec.label] = series

    if not rows:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No credit spread data was available from FRED."
        return result

    charts = {
        "history": render_chart(
            timeseries.line_chart(series_by_label, yaxis_title="OAS", percent=True),
            "credit",
            "history",
            height=300,
        )
    }

    shortest = min(r["history_years"] for r in rows)
    result.context = {"rows": rows, "charts": charts, "shortest_history": shortest}
    result.sources = list(bundle.sources.get("fred", []))
    result.notes = [
        "Option-adjusted spreads over Treasuries, in percentage points; "
        "changes are in basis points.",
        f"Percentiles and z-scores use up to {config.analytics.percentile_lookback} "
        f"sessions, clamped to available history (shortest here: {shortest:.1f} years).",
    ]
    result.status = (
        SectionStatus.OK if len(rows) == len(config.fred.credit) else SectionStatus.PARTIAL
    )
    return result
