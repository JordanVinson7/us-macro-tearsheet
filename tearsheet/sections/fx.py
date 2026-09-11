"""Section 8 — FX, commodities and crypto.

Performance tables in the same format as equities, plus the copper/gold ratio
against the 10-year yield — a widely watched read on growth expectations
versus what the bond market is pricing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tearsheet.charts import timeseries
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
    """Build the FX, commodities and crypto section."""
    config = settings.config
    result = SectionResult(key="fx_commodities", title="FX, commodities & crypto")

    groups = {
        "FX": build_performance_rows(config.universe.fx, bundle, config),
        "Commodities": build_performance_rows(config.universe.commodities, bundle, config),
        "Crypto": build_performance_rows(config.universe.crypto, bundle, config),
    }
    all_rows = [row for rows in groups.values() for row in rows]

    if not any(r.available for r in all_rows):
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No FX, commodity or crypto prices were available."
        return result

    charts: dict[str, str] = {}

    # --- performance bars --------------------------------------------------
    available = [r for r in all_rows if r.available and r.returns.get("1M") is not None]
    charts["performance"] = render_chart(
        timeseries.bar_chart(
            [r.label for r in available],
            [r.returns["1M"] for r in available],
            yaxis_title="1-month return",
        ),
        "fx",
        "performance",
        height=300,
    )

    # --- copper/gold against the 10-year -----------------------------------
    copper = close_series(bundle, "HG=F")
    gold = close_series(bundle, "GC=F")
    ten_year = bundle.fred.get("DGS10")
    note = None

    if copper is not None and gold is not None and ten_year is not None:
        aligned = pd.DataFrame({"copper": copper, "gold": gold}).dropna()
        ratio = (aligned["copper"] / aligned["gold"]) * 1000.0
        charts["copper_gold"] = render_chart(
            timeseries.dual_axis_chart(
                ratio, ten_year, "Copper/gold (×1000)", "10Y yield %", days=365 * 2
            ),
            "fx",
            "copper_gold",
            height=320,
        )
    else:
        note = "The copper/gold ratio chart needs HG=F, GC=F and DGS10; one was unavailable."
        result.warnings.append(note)

    result.context = {
        "groups": groups,
        "return_columns": ["1D", "1W", "1M", "3M", "YTD", "1Y"],
        "moving_averages": config.analytics.moving_averages,
        "charts": charts,
        "copper_gold_note": note,
        "vol_window": config.analytics.realised_vol_window,
    }
    result.sources = source_note(*[r.source for r in all_rows if r.available])
    result.notes = [
        "Futures prices are continuous front-month series. Returns spanning a "
        "contract roll are suppressed rather than shown, because the price step "
        "at a roll is an artefact of the series, not a market move.",
        "The copper/gold ratio is a common proxy for growth expectations; "
        "plotting it against the 10-year yield compares it with what the bond "
        "market is pricing.",
    ]
    result.status = (
        SectionStatus.OK
        if all(r.available for r in all_rows) and "copper_gold" in charts
        else SectionStatus.PARTIAL
    )
    return result
