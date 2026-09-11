"""Section 9 — cross-asset correlations and the risk regime composite.

The composite is a *descriptive* summary of conditions against their own
two-year history. It is explicitly not a trading signal, and is labelled that
way wherever it appears.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tearsheet.analytics import correlations as corr
from tearsheet.analytics import regime as rg
from tearsheet.analytics import volatility as vol
from tearsheet.charts import heatmaps, timeseries
from tearsheet.config import Config, Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.sections.base import close_series, render_chart

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle

#: Instruments whose volatility percentile is worth tabulating.
_VOLATILITY_UNIVERSE = ["^GSPC", "^NDX", "^RUT", "TLT", "GC=F", "CL=F", "BTC-USD"]


def build_regime_inputs(bundle: DataBundle, config: Config) -> dict[str, pd.Series]:
    """Derive the level series each regime component is scored against.

    Four components are raw series; three are derived here. A component whose
    inputs are missing is simply absent from the mapping, and
    :func:`~tearsheet.analytics.regime.compute_regime` re-normalises the
    remaining weights rather than scoring it as zero.
    """
    inputs: dict[str, pd.Series] = {}

    vix = close_series(bundle, "^VIX")
    if vix is not None:
        inputs["vix_level"] = vix

    vix3m = close_series(bundle, "^VIX3M")
    if vix is not None and vix3m is not None:
        aligned = pd.DataFrame({"near": vix, "far": vix3m}).dropna()
        if not aligned.empty:
            inputs["vix_term"] = aligned["near"] / aligned["far"]

    hy = bundle.fred.get("BAMLH0A0HYM2")
    if hy is not None and not hy.empty:
        inputs["hy_spread"] = hy

    spx = close_series(bundle, "^GSPC")
    if spx is not None and len(spx) > 200:
        inputs["spx_vs_200dma"] = (spx / spx.rolling(200).mean() - 1.0) * 100.0

    spy, tlt = close_series(bundle, "SPY"), close_series(bundle, "TLT")
    if spy is not None and tlt is not None:
        correlation = corr.price_correlation(spy, tlt, config.analytics.correlation_window)
        if not correlation.empty:
            inputs["stock_bond_corr"] = correlation

    dxy = close_series(bundle, "DX-Y.NYB")
    if dxy is not None and len(dxy) > 21:
        inputs["dxy_momentum"] = dxy.pct_change(21) * 100.0

    nfci = bundle.fred.get("NFCI")
    if nfci is not None and not nfci.empty:
        inputs["nfci"] = nfci

    return inputs


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the cross-asset and regime section."""
    config = settings.config
    analytics = config.analytics
    result = SectionResult(key="cross_asset", title="Cross-asset & regime")
    charts: dict[str, str] = {}

    # --- correlation matrix ------------------------------------------------
    members: dict[str, pd.Series] = {}
    spread_labels: set[str] = set()
    for member in config.cross_asset.members:
        series = (
            bundle.fred.get(member.key) if member.is_spread else close_series(bundle, member.key)
        )
        if series is None or series.empty:
            continue
        members[member.label] = series
        if member.is_spread:
            spread_labels.add(member.label)

    matrix = corr.correlation_matrix(
        corr.to_changes(members, spread_labels),
        analytics.correlation_window,
        analytics.min_observations,
    )
    charts["correlations"] = render_chart(
        heatmaps.correlation_heatmap(matrix, analytics.correlation_window),
        "crossasset",
        "correlations",
        height=max(320, 34 * len(matrix) + 90),
    )

    # --- stock-bond correlation -------------------------------------------
    spy, tlt = close_series(bundle, "SPY"), close_series(bundle, "TLT")
    stock_bond = (
        corr.price_correlation(spy, tlt, analytics.correlation_window)
        if spy is not None and tlt is not None
        else pd.Series(dtype="float64")
    )
    charts["stock_bond"] = render_chart(
        timeseries.line_chart(
            {"SPY / TLT": stock_bond}, yaxis_title="Correlation", zero_line=True, fill=True
        ),
        "crossasset",
        "stock_bond",
        height=280,
    )

    # --- VIX term structure ------------------------------------------------
    vix, vix3m = close_series(bundle, "^VIX"), close_series(bundle, "^VIX3M")
    term_ratio = None
    term_note = None
    if vix is not None and vix3m is not None:
        aligned = pd.DataFrame({"near": vix, "far": vix3m}).dropna()
        ratio = aligned["near"] / aligned["far"]
        term_ratio = float(ratio.iloc[-1]) if not ratio.empty else None
        charts["vix_term"] = render_chart(
            timeseries.line_chart({"VIX / VIX3M": ratio}, yaxis_title="Ratio", days=365 * 2),
            "crossasset",
            "vix_term",
            height=280,
        )
    else:
        term_note = (
            "Yahoo Finance stopped updating ^VIX3M, so the VIX term structure "
            "is unavailable and its regime component has been dropped."
        )
        result.warnings.append(term_note)

    # --- volatility percentiles -------------------------------------------
    volatility_rows = []
    for ticker in _VOLATILITY_UNIVERSE:
        series = close_series(bundle, ticker)
        if series is None:
            continue
        suspect = bundle.prices.suspect_returns.get(ticker)
        volatility_rows.append(
            {
                "ticker": ticker,
                "volatility": vol.latest_realised_volatility(
                    series, analytics.realised_vol_window, analytics.annualisation_factor, suspect
                ),
                "percentile": vol.volatility_percentile(
                    series,
                    analytics.realised_vol_window,
                    analytics.percentile_lookback,
                    analytics.annualisation_factor,
                    suspect,
                ),
            }
        )

    # --- regime composite --------------------------------------------------
    regime_result = rg.compute_regime(
        build_regime_inputs(bundle, config),
        config.regime.components,
        config.regime.bands,
        config.regime.lookback,
    )
    if regime_result is not None:
        charts["regime"] = render_chart(
            timeseries.regime_chart(regime_result.history, config.regime.bands),
            "crossasset",
            "regime",
            height=280,
        )
        charts["contributions"] = render_chart(
            timeseries.contribution_chart(regime_result.components),
            "crossasset",
            "contributions",
            height=max(220, 34 * len(regime_result.components) + 80),
        )
    else:
        result.warnings.append("The risk regime composite had no usable components.")

    result.context = {
        "matrix": matrix,
        "correlation_window": analytics.correlation_window,
        "stock_bond_latest": float(stock_bond.iloc[-1]) if not stock_bond.empty else None,
        "term_ratio": term_ratio,
        "term_note": term_note,
        "volatility_rows": volatility_rows,
        "regime": regime_result,
        "regime_total_components": len(config.regime.components),
        "charts": charts,
    }
    result.sources = list(bundle.sources.get("prices", [])) + list(bundle.sources.get("fred", []))
    result.notes = [
        f"Correlations use {analytics.correlation_window} sessions of daily "
        "changes. Credit spreads are differenced in basis points rather than "
        "percentage-changed, which would be meaningless near zero.",
        f"Volatility percentiles rank the current {analytics.realised_vol_window}-day "
        f"realised volatility within its own trailing {analytics.percentile_lookback} sessions.",
        "The regime composite z-scores each component against its own trailing "
        f"{config.regime.lookback} sessions, applies the configured sign and weight, "
        "and re-normalises over whichever components have data.",
        "The composite is a descriptive summary of conditions relative to recent "
        "history. It is not a forecast, a recommendation, or a trading signal.",
    ]
    result.status = (
        SectionStatus.OK
        if regime_result is not None and regime_result.is_complete and not matrix.empty
        else SectionStatus.PARTIAL
    )
    return result
