"""Section 1 — header, KPI strip and the "What stood out" panel.

The flags are produced entirely by the rules in
:mod:`tearsheet.analytics.flags`, with every threshold coming from config.yaml.
Nothing on this page is written by a language model: the panel has to be
reproducible, and a reader must be able to ask why a line appeared and get an
exact answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tearsheet.analytics import flags as fl
from tearsheet.analytics import rates as rt
from tearsheet.analytics import returns as ret
from tearsheet.config import Config, Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.sections.base import close_series

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle

#: Instrument groups swept by the move, extreme and moving-average rules.
_FLAG_GROUPS = ("indices", "sectors", "commodities", "crypto", "fx", "volatility")


def _kpi(key: str, bundle: DataBundle, config: Config) -> dict | None:
    """Build one KPI tile.

    A key that names a FRED series is treated as a yield and shown with a
    basis-point change; anything else is treated as a price and shown with a
    percentage change.
    """
    series = bundle.fred.get(key)
    if series is not None and not series.empty:
        spec = config.fred.find(key)
        return {
            "key": key,
            "label": spec.label if spec else key,
            "value": float(series.iloc[-1]),
            "value_suffix": "%",
            "change": rt.change_in_bp(series, 1),
            "change_suffix": "bp",
            "change_decimals": 0,
            "observation_date": series.index[-1].date(),
        }

    prices = close_series(bundle, key)
    if prices is None:
        return None

    instrument = config.universe.find(key)
    suspect = bundle.prices.suspect_returns.get(key)
    return {
        "key": key,
        "label": instrument.label if instrument else key,
        "value": float(prices.iloc[-1]),
        "value_suffix": "",
        "change": ret.simple_return(prices, 1, suspect),
        "change_suffix": "%",
        "change_decimals": 2,
        "observation_date": prices.index[-1].date(),
    }


def _collect_flags(bundle: DataBundle, config: Config) -> list[fl.Flag]:
    """Run every flag rule over the day's data."""
    thresholds = config.flags
    analytics = config.analytics
    found: list[fl.Flag | None] = []

    groups = config.universe.instrument_groups
    for group_name in _FLAG_GROUPS:
        for instrument in groups.get(group_name, []):
            prices = close_series(bundle, instrument.ticker)
            if prices is None:
                continue
            suspect = bundle.prices.suspect_returns.get(instrument.ticker)

            found.append(
                fl.flag_return_zscore(
                    instrument.ticker,
                    instrument.name,
                    ret.return_zscore(
                        prices, analytics.zscore_lookback, suspect, analytics.min_observations
                    ),
                    ret.simple_return(prices, 1, suspect),
                    thresholds.return_zscore,
                )
            )
            found.append(
                fl.flag_extreme(
                    instrument.ticker,
                    instrument.name,
                    ret.at_new_extreme(prices, thresholds.high_low_window),
                    "52-week",
                )
            )
            for window in analytics.moving_averages:
                found.append(
                    fl.flag_ma_cross(
                        instrument.ticker,
                        instrument.name,
                        window,
                        ret.ma_cross(prices, window, thresholds.ma_cross_tolerance_pct),
                    )
                )

    # VIX term structure.
    vix, vix3m = close_series(bundle, "^VIX"), close_series(bundle, "^VIX3M")
    ratio = None
    if vix is not None and vix3m is not None and not vix.empty and not vix3m.empty:
        ratio = float(vix.iloc[-1]) / float(vix3m.iloc[-1])
    found.append(fl.flag_vix_term_structure(ratio, thresholds.vix_backwardation_ratio))

    # Curve inversions.
    for series_id, name in (("T10Y2Y", "2s10s"), ("T10Y3M", "3m10y")):
        series = bundle.fred.get(series_id)
        if series is None or series.empty:
            continue
        state = rt.inversion_state(series, thresholds.curve_inversion_tolerance_bp)
        if state:
            found.append(fl.flag_curve_transition(name, state.transition, state.spread_bp))

    # Credit spread moves.
    for spec in config.fred.credit:
        series = bundle.fred.get(spec.id)
        if series is None or series.empty:
            continue
        found.append(
            fl.flag_credit_move(
                spec.id, spec.name, rt.change_in_bp(series, 1), thresholds.credit_spread_move_bp
            )
        )

    return [flag for flag in found if flag is not None]


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the header, KPI strip and flags panel."""
    config = settings.config
    result = SectionResult(key="header", title="Key takeaways")

    kpis = [k for key in config.display.kpi_strip if (k := _kpi(key, bundle, config))]
    all_flags = _collect_flags(bundle, config)
    flags = fl.rank_flags(all_flags, config.flags.max_flags)

    if not kpis:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No headline figures were available."
        return result

    # "Data as of" is reported per source, because they genuinely differ: FRED
    # yields lag by a day or two, while prices are the prior session's close.
    as_of: dict[str, str] = {"Prices": bundle.session.strftime("%d %b %Y")}
    if bundle.fred:
        latest_fred = max(s.index[-1].date() for s in bundle.fred.values() if not s.empty)
        as_of["FRED"] = latest_fred.strftime("%d %b %Y")
    if bundle.intraday:
        as_of["Intraday"] = bundle.session.strftime("%d %b %Y")
    if bundle.quotes:
        stamps = [q.as_of for q in bundle.quotes.values() if q.as_of]
        if stamps:
            as_of["Pre-open quotes"] = max(stamps).strftime("%d %b %Y, %H:%M ET")

    result.context = {
        "kpis": kpis,
        "flags": flags,
        "flag_count": len(all_flags),
        "as_of": as_of,
        "session": bundle.session,
        "zscore_threshold": config.flags.return_zscore,
    }
    result.sources = list(bundle.sources.get("prices", [])) + list(bundle.sources.get("fred", []))
    result.notes = [
        f"Flags are generated by explicit rules, not by a language model. "
        f"A move is flagged when its one-day return exceeds "
        f"{config.flags.return_zscore:.1f} standard deviations of its trailing "
        f"{config.analytics.zscore_lookback}-session distribution.",
        "Every threshold is set in config.yaml.",
    ]
    result.status = (
        SectionStatus.OK if len(kpis) == len(config.display.kpi_strip) else SectionStatus.PARTIAL
    )
    return result
