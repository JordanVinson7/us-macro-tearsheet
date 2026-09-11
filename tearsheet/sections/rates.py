"""Section 5 — rates and the Fed.

The Treasury curve today against earlier snapshots, key yields in basis-point
terms, the classified curve move, spread history, and real/breakeven/policy
rates.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from tearsheet.analytics import rates as rt
from tearsheet.charts import timeseries
from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.sections.base import render_chart

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle

#: How far back each historical curve snapshot is taken, in calendar days.
_SNAPSHOT_DAYS = {"1W": 7, "1M": 30, "1Y": 365}


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the rates section."""
    config = settings.config
    result = SectionResult(key="rates", title="Rates & the Fed")

    specs = [(s.id, s.name, s.tenor_years or 0.0) for s in config.fred.treasury_curve]
    curve = rt.build_curve(bundle.fred, specs)

    if not curve:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No Treasury curve data was available from FRED."
        return result

    # --- curve chart: today against earlier snapshots ----------------------
    curves = {"Today": curve}
    for label, days in _SNAPSHOT_DAYS.items():
        snapshot = rt.build_curve(bundle.fred, specs, bundle.session - timedelta(days=days))
        if snapshot:
            curves[f"{label} ago"] = snapshot

    charts = {"curve": render_chart(timeseries.curve_chart(curves), "rates", "curve", height=340)}

    # --- key yields --------------------------------------------------------
    yield_rows = []
    for point in curve:
        series = bundle.fred[point.series_id]
        yield_rows.append(
            {
                "label": point.label,
                "series_id": point.series_id,
                "level": point.yield_pct,
                "observation_date": point.observation_date,
                "changes": {
                    window: rt.change_in_bp(series, periods=periods)
                    for window, periods in config.analytics.return_windows.items()
                },
            }
        )

    # --- curve move classification -----------------------------------------
    short, long = config.curve.short_tenor, config.curve.long_tenor
    classifications = []
    if short in bundle.fred and long in bundle.fred:
        for window in config.curve.comparison_windows:
            periods = config.analytics.return_windows.get(window, 1)
            move = rt.classify_curve_move(
                rt.change_in_bp(bundle.fred[short], periods),
                rt.change_in_bp(bundle.fred[long], periods),
                config.curve.classification_threshold_bp,
            )
            if move:
                classifications.append({"window": window, "move": move})

    # --- spreads -----------------------------------------------------------
    spread_names = {"T10Y2Y": "2s10s", "T10Y3M": "3m10y"}
    spread_series = {
        name: bundle.fred[sid] for sid, name in spread_names.items() if sid in bundle.fred
    }
    charts["spreads"] = render_chart(
        timeseries.line_chart(
            spread_series, yaxis_title="Spread", days=365 * 3, percent=True, zero_line=True
        ),
        "rates",
        "spreads",
        height=300,
    )
    spread_states = [
        {
            "name": name,
            "state": rt.inversion_state(series, config.flags.curve_inversion_tolerance_bp),
        }
        for name, series in spread_series.items()
    ]

    # --- other rates -------------------------------------------------------
    other_rows = []
    for spec in config.fred.rates:
        series = bundle.fred.get(spec.id)
        if series is None or series.empty:
            continue
        staleness = bundle.staleness.get(spec.id)
        other_rows.append(
            {
                "name": spec.name,
                "series_id": spec.id,
                "level": float(series.iloc[-1]),
                "change_1d": rt.change_in_bp(series, 1),
                "change_1w": rt.change_in_bp(series, 5),
                "observation_date": series.index[-1].date(),
                "stale": bool(staleness and staleness.is_stale),
            }
        )

    result.context = {
        "curve": curve,
        "curve_observed": curve[0].observation_date,
        "yield_rows": yield_rows,
        "return_columns": list(config.analytics.return_windows),
        "classifications": classifications,
        "spread_states": spread_states,
        "other_rows": other_rows,
        "charts": charts,
    }
    result.sources = list(bundle.sources.get("fred", []))
    result.notes = [
        "Yields are FRED constant-maturity series, which are published with a "
        "one- to two-day lag; each figure carries its observation date.",
        "Changes are in basis points. Curve snapshots use the last observation "
        "on or before the target date, since the series skip weekends.",
        "Bull/bear describes the direction of yields; steepener/flattener the "
        f"change in the {long[3:]}-minus-{short[3:]} spread. Moves below "
        f"{config.curve.classification_threshold_bp:.0f}bp are treated as unchanged.",
    ]
    result.status = SectionStatus.OK if len(curve) >= 8 else SectionStatus.PARTIAL
    return result
