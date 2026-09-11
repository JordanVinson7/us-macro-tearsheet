"""Polygon (rebranded "Massive" in late 2025) REST client.

Only three endpoints are used, all of them available on the free tier:

* ``/v2/aggs/ticker/{t}/range/1/minute/...`` — previous-session minute bars,
  one call per ticker, for the intraday panel.
* ``/v2/aggs/grouped/locale/us/market/stocks/{date}`` — one call returning the
  prior session's OHLC for every US stock and ETF. This is the first fallback
  when Yahoo rate-limits the runner.
* ``/v2/reference/news`` — headlines.

The free tier allows **5 requests per rolling 60 seconds** and has no snapshot
endpoint, so the run budget is deliberately small (see ``call_budget`` in
config.yaml) and pre-open quotes come from Yahoo instead. Every request passes
through a :class:`~tearsheet.data.base.RateLimiter`, which blocks pre-emptively
rather than absorbing a 429 and a penalty backoff.

The base URL is configurable because the Massive rebrand may eventually retire
the ``api.polygon.io`` host; the API surface and keys are unchanged.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from tearsheet.config import PolygonSourceConfig
from tearsheet.data.base import ApiClient, DataSourceError, RateLimiter
from tearsheet.models import NewsItem, SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.polygon")

POLYGON_SOURCE = SourceRef(name="Polygon", detail="minute aggregates")
POLYGON_GROUPED_SOURCE = SourceRef(name="Polygon", detail="grouped daily bars", is_fallback=True)
POLYGON_NEWS_SOURCE = SourceRef(name="Polygon", detail="news endpoint")

#: US regular trading hours, used to strip extended-hours minute bars.
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class PolygonClient:
    """Rate-limited client for the handful of Polygon endpoints used here."""

    def __init__(self, config: PolygonSourceConfig, api_key: str, report: RunReport) -> None:
        """Initialise the client.

        Args:
            config: Polygon settings, including the rate limit and call budget.
            api_key: The Polygon/Massive API key.
            report: Run report for call accounting.
        """
        self._api_key = api_key
        self._client = ApiClient(
            name="polygon",
            base_url=config.base_url,
            report=report,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            rate_limiter=RateLimiter(config.rate_limit.max_calls, config.rate_limit.per_seconds),
            call_budget=config.call_budget,
        )
        self.report = report

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._client.close()

    @property
    def seconds_rate_limited(self) -> float:
        """Total time this run spent blocked by the rate limiter."""
        return self._client.wait_seconds

    def _get(self, path: str, **params: object) -> dict:
        """Issue a GET with the API key attached."""
        return self._client.get(path, {**params, "apiKey": self._api_key})

    # -- aggregates ----------------------------------------------------------

    def fetch_minute_bars(
        self,
        ticker: str,
        session: date,
        timezone: str = "America/New_York",
        *,
        regular_hours_only: bool = True,
    ) -> pd.DataFrame:
        """Fetch one session of minute bars.

        Args:
            ticker: US stock or ETF symbol.
            session: The trading session to fetch.
            timezone: Timezone the returned index is expressed in.
            regular_hours_only: Drop pre- and post-market bars. Polygon returns
                extended hours by default, which would distort VWAP and the
                first- and last-hour return statistics.

        Returns:
            A frame indexed by ET timestamp with ``open/high/low/close/volume``
            and ``vwap`` columns. Empty if Polygon returned nothing.
        """
        payload = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{session.isoformat()}/{session.isoformat()}",
            adjusted="true",
            sort="asc",
            limit=50000,
        )
        results = payload.get("results") or []
        if not results:
            logger.warning("Polygon returned no minute bars for %s on %s", ticker, session)
            return pd.DataFrame()

        frame = pd.DataFrame(results).rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trades",
            }
        )
        frame.index = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(timezone)
        frame = frame.drop(columns=[c for c in ("t", "trades") if c in frame.columns])

        if regular_hours_only:
            times = frame.index.time
            mask = (times >= REGULAR_OPEN) & (times < REGULAR_CLOSE)
            frame = frame[mask]

        logger.debug("Polygon: %d minute bars for %s on %s", len(frame), ticker, session)
        return frame.sort_index()

    def fetch_grouped_daily(
        self, session: date, tickers: set[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Fetch one session of daily bars for every US stock and ETF.

        A single call covers the entire market, which is what makes this a
        viable fallback when Yahoo refuses a cloud IP.

        Args:
            session: The trading session to fetch.
            tickers: Optional filter applied client-side.

        Returns:
            OHLCV keyed by ticker.
        """
        payload = self._get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{session.isoformat()}",
            adjusted="true",
        )
        results = payload.get("results") or []
        bars: dict[str, dict[str, float]] = {}
        for row in results:
            symbol = row.get("T")
            if not symbol or (tickers is not None and symbol not in tickers):
                continue
            bars[symbol] = {
                "open": float(row["o"]),
                "high": float(row["h"]),
                "low": float(row["l"]),
                "close": float(row["c"]),
                "volume": float(row.get("v", 0.0)),
            }
        logger.info("Polygon grouped daily for %s: %d symbols matched", session, len(bars))
        return bars

    # -- news ----------------------------------------------------------------

    def fetch_news(self, limit: int = 20, timezone: str = "America/New_York") -> list[NewsItem]:
        """Fetch recent market headlines.

        Args:
            limit: Maximum headlines to request.
            timezone: Timezone for the returned timestamps.

        Returns:
            Headlines, newest first. An empty list if the endpoint is
            unavailable on the current plan.
        """
        try:
            payload = self._get(
                "/v2/reference/news", limit=limit, order="desc", sort="published_utc"
            )
        except DataSourceError as exc:
            self.report.warn(f"Polygon news unavailable: {exc}")
            return []

        items: list[NewsItem] = []
        for row in payload.get("results") or []:
            title = (row.get("title") or "").strip()
            if not title:
                continue
            items.append(
                NewsItem(
                    title=title,
                    publisher=(row.get("publisher") or {}).get("name", "Unknown"),
                    url=row.get("article_url", ""),
                    published_at=_parse_utc(row.get("published_utc"), timezone),
                    tickers=tuple(row.get("tickers") or ())[:6],
                    source=POLYGON_NEWS_SOURCE,
                )
            )
        return items


def _parse_utc(value: str | None, timezone: str) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp into the report timezone."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ZoneInfo(timezone))
    except ValueError:
        return None
