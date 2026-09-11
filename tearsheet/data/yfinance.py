"""yfinance client for daily bars and pre-open quotes.

Yahoo is the primary source for everything priced daily. Two practical
problems shape this module:

* **Cloud IPs get rate-limited.** GitHub Actions runners share addresses with
  a lot of other traffic, so requests are batched through ``yf.download``
  (one HTTP round trip per batch rather than per ticker) and retried with
  exponential backoff. When Yahoo still refuses, the caller falls back through
  :mod:`tearsheet.data.fallbacks`.
* **A pre-open run must not see a partial bar.** Yahoo happily returns a
  half-formed bar for a session in progress. Everything here is trimmed to the
  last completed session, so an afternoon local run produces the same numbers
  as a real 08:00 ET run.

``fetch_quotes`` is the documented exception to the project's "intraday comes
from Polygon" rule: Polygon's free tier has no snapshot endpoint, so live
pre-open levels for futures, FX and crypto come from Yahoo instead.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from tearsheet.config import YFinanceSourceConfig
from tearsheet.models import PriceData, Quote, SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.yfinance")

YF_SOURCE = SourceRef(name="yfinance", detail="Yahoo Finance daily bars")
YF_QUOTE_SOURCE = SourceRef(
    name="yfinance",
    detail="Yahoo quotes (Polygon's free tier has no snapshot endpoint)",
)

#: Canonical OHLCV column names used throughout the project.
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class YahooUnavailableError(RuntimeError):
    """Raised when Yahoo returns nothing usable for an entire batch."""


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Split ``items`` into chunks of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalise(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Turn a ``yf.download`` result into one tidy OHLCV frame per ticker.

    yfinance returns a two-level column index of (field, ticker) for a
    multi-ticker request and a flat index for a single ticker, so both shapes
    are handled here rather than at every call site.
    """
    frames: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return frames

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(1):
                    continue
                frame = raw.xs(ticker, axis=1, level=1)
            else:
                frame = raw
        except KeyError:
            continue

        frame = frame.rename(columns=str.lower)
        available = [c for c in OHLCV_COLUMNS if c in frame.columns]
        if "close" not in available:
            continue

        frame = frame[available].astype("float64")
        frame = frame.dropna(subset=["close"])
        if frame.empty:
            continue

        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frames[ticker] = frame.sort_index()

    return frames


def download_daily(
    tickers: list[str],
    config: YFinanceSourceConfig,
    report: RunReport,
    *,
    as_of: date,
    start: date | None = None,
) -> PriceData:
    """Download daily OHLCV history for many tickers.

    Args:
        tickers: Yahoo symbols.
        config: Batching and retry settings.
        report: Run report for call accounting.
        as_of: Last completed session; history is trimmed to this date so a
            partially-formed current bar can never reach the analytics.
        start: Earliest date to request. Defaults to ``history_days`` back.

    Returns:
        Populated price data, with anything Yahoo could not supply listed in
        :attr:`~tearsheet.models.PriceData.missing`.
    """
    start = start or as_of - timedelta(days=config.history_days)
    data = PriceData(as_of=as_of)

    for batch in _batched(tickers, config.batch_size):
        try:
            frames = _download_batch(batch, config, report, start=start, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 — the fallback chain handles this
            report.warn(f"yfinance batch of {len(batch)} tickers failed: {exc}")
            frames = {}

        for ticker, frame in frames.items():
            data.frames[ticker] = frame
            data.sources[ticker] = YF_SOURCE

    data.missing = [t for t in tickers if t not in data.frames]
    if data.missing:
        report.warn(f"yfinance returned no data for: {', '.join(data.missing)}")

    logger.info("yfinance: %d/%d tickers, history to %s", len(data.frames), len(tickers), as_of)
    return data


def _download_batch(
    batch: list[str],
    config: YFinanceSourceConfig,
    report: RunReport,
    *,
    start: date,
    as_of: date,
) -> dict[str, pd.DataFrame]:
    """Download one batch, retrying on transient Yahoo failures."""
    retryer = Retrying(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=2, min=2, max=45),
        retry=retry_if_exception_type((YahooUnavailableError, OSError)),
        reraise=True,
        before_sleep=lambda state: logger.warning(
            "yfinance attempt %d failed; backing off", state.attempt_number
        ),
    )

    def attempt() -> dict[str, pd.DataFrame]:
        report.record_api_call("yfinance")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                batch,
                start=start.isoformat(),
                # yfinance treats `end` as exclusive.
                end=(as_of + timedelta(days=1)).isoformat(),
                interval="1d",
                # Adjusted closes make multi-month total returns correct for
                # dividend-paying ETFs such as the sector SPDRs.
                auto_adjust=True,
                group_by="column",
                progress=False,
                threads=False,
                # repair=True would use yfinance's bad-tick correction, but it
                # pulls in scipy and scikit-learn (~120MB) for every CI run.
                # tearsheet.data.quality applies its own sanity checks instead.
            )
        frames = _normalise(raw, batch)
        if not frames:
            raise YahooUnavailableError(f"no data for batch: {', '.join(batch)}")
        # Guard against a partial bar for a session still in progress.
        cutoff = pd.Timestamp(as_of)
        return {t: f.loc[:cutoff] for t, f in frames.items()}

    return retryer(attempt)


def fetch_quotes(
    tickers: list[str],
    report: RunReport,
    timezone: str = "America/New_York",
) -> dict[str, Quote]:
    """Fetch live-ish quotes for the overnight panel.

    This is the project's one documented departure from "intraday data comes
    from Polygon": the free Polygon tier exposes no snapshot endpoint, so
    pre-open futures, FX and crypto levels come from Yahoo.

    Args:
        tickers: Yahoo symbols.
        report: Run report for call accounting.
        timezone: Timezone for the quote timestamp.

    Returns:
        Quotes keyed by ticker. Symbols that fail are simply absent.
    """
    quotes: dict[str, Quote] = {}
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)

    for ticker in tickers:
        try:
            report.record_api_call("yfinance")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = yf.Ticker(ticker).fast_info
                price = info.get("lastPrice")
                previous = info.get("previousClose")
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the panel
            logger.debug("Quote for %s failed: %s", ticker, exc)
            continue

        if price is None or not float(price) > 0:
            continue

        quotes[ticker] = Quote(
            ticker=ticker,
            price=float(price),
            previous_close=float(previous) if previous else None,
            as_of=now,
            source=YF_QUOTE_SOURCE,
        )

    logger.info("yfinance quotes: %d/%d symbols", len(quotes), len(tickers))
    return quotes
