"""The ordered fallback chain for daily price data.

Yahoo is the primary source and is usually fine, but GitHub Actions runners
share IP addresses with a lot of other traffic and Yahoo throttles them. When
that happens the run degrades rather than fails:

1. **yfinance** — full OHLCV history for everything.
2. **Polygon grouped daily** — one call returns the previous session's OHLC for
   every US stock and ETF. This recovers *levels* but not history, so affected
   sections render as PARTIAL: a price, but no returns or volatility.
3. **FRED levels** — a small hand-mapped set (S&P 500, VIX, WTI, EURUSD,
   USDJPY, the dollar index). Slower to update than Yahoo but, unlike grouped
   daily, it carries full history, so it can keep the charts alive.

Whatever supplies a ticker is recorded on the :class:`~tearsheet.models.PriceData`
and reported in the page footer, so a reader can always tell when a number came
from a second-choice source.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from tearsheet.config import Config
from tearsheet.data.fred import FredClient
from tearsheet.data.polygon import POLYGON_GROUPED_SOURCE, PolygonClient
from tearsheet.data.yfinance import OHLCV_COLUMNS, download_daily
from tearsheet.models import PriceData, SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.fallbacks")

FRED_LEVEL_SOURCE = SourceRef(name="FRED", detail="series levels", is_fallback=True)

#: Grouped daily only covers US-listed stocks and ETFs. Index, futures and FX
#: symbols carry Yahoo-specific decorations and are never available there.
_YAHOO_ONLY_MARKERS = ("^", "=F", "=X", "-USD", ".NYB")


def _is_us_equity_symbol(ticker: str) -> bool:
    """Whether Polygon's US equity universe could plausibly contain this symbol."""
    return not any(marker in ticker for marker in _YAHOO_ONLY_MARKERS)


def _frame_from_close(close: pd.Series) -> pd.DataFrame:
    """Build an OHLCV-shaped frame from a close-only series.

    Sources that provide only a closing level still have to satisfy consumers
    that expect OHLCV columns, so open/high/low mirror the close and volume is
    left as NaN rather than being fabricated as zero.
    """
    frame = pd.DataFrame(index=pd.to_datetime(close.index).tz_localize(None).normalize())
    for column in OHLCV_COLUMNS:
        frame[column] = float("nan") if column == "volume" else close.to_numpy()
    return frame.sort_index()


def _polygon_grouped_fallback(
    missing: list[str],
    session: date,
    polygon: PolygonClient | None,
    report: RunReport,
) -> PriceData:
    """Recover last-known levels for US equities in a single Polygon call."""
    recovered = PriceData(as_of=session)
    candidates = {t for t in missing if _is_us_equity_symbol(t)}
    if not candidates or polygon is None:
        return recovered

    try:
        bars = polygon.fetch_grouped_daily(session, tickers=candidates)
    except Exception as exc:  # noqa: BLE001 — this is itself a fallback path
        report.warn(f"Polygon grouped-daily fallback failed: {exc}")
        return recovered

    index = pd.DatetimeIndex([pd.Timestamp(session)])
    for ticker, bar in bars.items():
        recovered.frames[ticker] = pd.DataFrame(
            {column: [bar.get(column, float("nan"))] for column in OHLCV_COLUMNS},
            index=index,
        )
        recovered.sources[ticker] = POLYGON_GROUPED_SOURCE

    if recovered.frames:
        report.warn(
            f"Recovered {len(recovered.frames)} ticker(s) from Polygon grouped daily; "
            "only the latest level is available, so returns and volatility are omitted."
        )
    return recovered


def _fred_level_fallback(
    missing: list[str],
    config: Config,
    fred: FredClient | None,
    session: date,
    report: RunReport,
) -> PriceData:
    """Recover a mapped subset of levels from FRED, with full history."""
    recovered = PriceData(as_of=session)
    mapping = {t: sid for t, sid in config.fred.fallback_levels.items() if t in set(missing)}
    if not mapping or fred is None:
        return recovered

    start = session - timedelta(days=config.sources.yfinance.history_days)
    for ticker, series_id in mapping.items():
        try:
            values = fred.fetch_observations(series_id, start=start, end=session)
        except Exception as exc:  # noqa: BLE001 — last link in the chain
            report.warn(f"FRED fallback for {ticker} ({series_id}) failed: {exc}")
            continue
        if values.empty:
            continue
        recovered.frames[ticker] = _frame_from_close(values)
        recovered.sources[ticker] = SourceRef(
            name="FRED", detail=f"{series_id} (fallback for {ticker})", is_fallback=True
        )

    if recovered.frames:
        report.warn(
            f"Recovered {len(recovered.frames)} ticker(s) from FRED levels; "
            "these update more slowly than Yahoo."
        )
    return recovered


def resolve_prices(
    tickers: list[str],
    config: Config,
    report: RunReport,
    *,
    as_of: date,
    polygon: PolygonClient | None = None,
    fred: FredClient | None = None,
) -> PriceData:
    """Fetch daily price history, walking the fallback chain as needed.

    Args:
        tickers: Yahoo symbols to resolve.
        config: Full configuration; the chain order comes from
            ``sources.yfinance.fallback_chain``.
        report: Run report for warnings and call accounting.
        as_of: The last completed session.
        polygon: Client for the grouped-daily fallback, if available.
        fred: Client for the FRED level fallback, if available.

    Returns:
        Price data with per-ticker provenance. Tickers no source could supply
        remain listed in ``missing``.
    """
    data = download_daily(tickers, config.sources.yfinance, report, as_of=as_of)

    handlers = {
        "polygon_grouped_daily": lambda missing: _polygon_grouped_fallback(
            missing, as_of, polygon, report
        ),
        "fred_levels": lambda missing: _fred_level_fallback(missing, config, fred, as_of, report),
    }

    for step in config.sources.yfinance.fallback_chain:
        if not data.missing:
            break
        handler = handlers.get(step)
        if handler is None:
            report.warn(f"Unknown fallback step {step!r} in config; skipping.")
            continue
        logger.info("Trying fallback %r for %d ticker(s)", step, len(data.missing))
        data.merge(handler(data.missing))

    if data.missing:
        report.warn(
            f"No source could supply: {', '.join(data.missing)}. "
            "Affected sections will show 'data unavailable'."
        )
    return data
