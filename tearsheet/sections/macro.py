"""Section 7 — the macro dashboard.

Each indicator's latest and prior reading, its change, year-on-year where
meaningful, the observation date and the next scheduled release, plus a grid
of five-year trend charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tearsheet.analytics import transforms
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
    """Build the macro dashboard."""
    config = settings.config
    result = SectionResult(key="macro", title="Macro dashboard")

    next_release = bundle.calendar.next_release_by_series if bundle.calendar else {}
    rows = []
    trend_series = {}

    for spec in config.fred.macro:
        series = bundle.fred.get(spec.id)
        if series is None or series.empty:
            rows.append({"name": spec.name, "series_id": spec.id, "available": False})
            continue

        reading = transforms.read_series(
            series, spec.transform, spec.frequency, spec.scale, spec.units, spec.yoy_units
        )
        staleness = bundle.staleness.get(spec.id)

        rows.append(
            {
                "available": reading.is_available,
                "name": spec.name,
                "series_id": spec.id,
                "latest": reading.latest,
                "prior": reading.prior,
                "change": reading.change,
                "yoy": reading.yoy,
                "yoy_units": reading.yoy_units,
                "units": spec.units,
                "transform": spec.transform,
                "observation_date": reading.observation_date,
                "next_release": next_release.get(spec.id),
                "stale": bool(staleness and staleness.is_stale),
                "proxy_for": spec.proxy_for,
            }
        )

        transformed = transforms.apply_transform(series, spec.transform, spec.frequency, spec.scale)
        if len(transformed.dropna()) > 1:
            trend_series[spec.name] = transformed

    if not any(r.get("available") for r in rows):
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No macro series were available from FRED."
        return result

    charts = {
        "trends": render_chart(
            timeseries.small_multiples(
                trend_series, columns=4, days=365 * config.display.small_multiples_years
            ),
            "macro",
            "trends",
        )
    }

    proxies = [r for r in rows if r.get("proxy_for")]
    result.context = {
        "rows": rows,
        "charts": charts,
        "trend_years": config.display.small_multiples_years,
        "has_proxies": bool(proxies),
    }
    result.sources = list(bundle.sources.get("fred", []))
    result.notes = [
        "Transforms are declared per series in config.yaml: year-on-year for "
        "price indices, a three-month average change for payrolls, and a "
        "four-week average for jobless claims.",
        "Year-on-year is measured from the underlying level. Series that are "
        "themselves rates are shown in percentage points (pp), and indices that "
        "cross zero in index points (pts), because a percentage change of a "
        "near-zero base is meaningless.",
        "FRED dates an observation at the start of its period, so a current "
        "monthly series can legitimately look six weeks old. The stale badge "
        "therefore fires on a missed scheduled release, not on age.",
    ]
    if proxies:
        result.notes.append(
            "ISM PMI data is not available on FRED. The regional Federal Reserve "
            "manufacturing surveys are shown as proxies and labelled as such."
        )

    missing = [r for r in rows if not r.get("available")]
    result.status = SectionStatus.OK if not missing else SectionStatus.PARTIAL
    return result
