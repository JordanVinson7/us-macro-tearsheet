"""Notable earnings for the report date, via yfinance.

This is its own module rather than part of :mod:`tearsheet.data.calendar`
because it has a different provider, a different failure mode and its own call
budget — economic releases come from FRED in a handful of cheap calls, whereas
earnings cost one slow Yahoo request per name.

**How "notable" is decided.** The brief defines notable as a market cap above
$10bn. Fetching a live market cap means a ``Ticker.info`` call per name, which
is the slowest and most rate-limit-prone call in yfinance — on a 60-name list
it would dominate the runtime and invite throttling. So the watchlist in
config.yaml *is* the filter: it is curated to names already above the
threshold, and only dates and EPS estimates are fetched. Same output, a
fraction of the cost.

**Why two passes.** ``Ticker.calendar`` is the cheap call but returns only a
date, so it cannot tell before-open from after-close. ``get_earnings_dates``
returns the timestamp but costs roughly twice as much. The cheap call is run
across the whole watchlist to find who reports today, then the expensive call
is made only for those few names.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Literal

import yfinance as yf

from tearsheet.config import EarningsConfig
from tearsheet.models import SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.earnings")

EARNINGS_SOURCE = SourceRef(name="yfinance", detail="earnings calendar")

#: Concurrency for the watchlist sweep. Kept low deliberately: Yahoo throttles
#: aggressively, and a partial earnings list is a far better outcome than a
#: rate-limit cascade that also breaks the price download.
MAX_WORKERS = 5

Session = Literal["bmo", "amc", "unknown"]

#: Reports timestamped before the opening bell are before-open; at or after the
#: closing bell, after-close.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    """One company reporting on the report date.

    Attributes:
        ticker: Yahoo symbol.
        session: Before open, after close, or unknown.
        eps_estimate: Consensus EPS estimate, where Yahoo provides one.
        when: Precise reporting timestamp in ET, when known.
    """

    ticker: str
    session: Session
    eps_estimate: float | None
    when: datetime | None

    @property
    def time_label(self) -> str:
        """Display label for the reporting time."""
        if self.when is None:
            return "—"
        return self.when.strftime("%H:%M ET")


@dataclass
class EarningsCalendar:
    """Notable earnings for the report date.

    Attributes:
        before_open: Companies reporting before the opening bell.
        after_close: Companies reporting after the closing bell.
        unscheduled: Companies reporting today at an unknown time.
        checked: How many watchlist names were successfully queried.
        failed: How many failed, usually through Yahoo throttling.
        source: Provenance for the footer.
    """

    before_open: list[EarningsEvent] = field(default_factory=list)
    after_close: list[EarningsEvent] = field(default_factory=list)
    unscheduled: list[EarningsEvent] = field(default_factory=list)
    checked: int = 0
    failed: int = 0
    source: SourceRef = EARNINGS_SOURCE

    @property
    def total(self) -> int:
        """How many companies report today."""
        return len(self.before_open) + len(self.after_close) + len(self.unscheduled)

    @property
    def is_complete(self) -> bool:
        """Whether every watchlist name was successfully checked."""
        return self.failed == 0


def _next_earnings_date(ticker: str) -> tuple[date | None, float | None]:
    """Cheap first pass: the next earnings date and EPS estimate for one name."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calendar = yf.Ticker(ticker).calendar

    if not isinstance(calendar, dict):
        return None, None

    dates = calendar.get("Earnings Date") or []
    if isinstance(dates, date):
        dates = [dates]
    if not dates:
        return None, None

    estimate = calendar.get("Earnings Average")
    return dates[0], float(estimate) if estimate is not None else None


def _reporting_time(ticker: str, on: date) -> datetime | None:
    """Expensive second pass: the precise reporting timestamp, if published."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = yf.Ticker(ticker).get_earnings_dates(limit=8)

    if frame is None or frame.empty:
        return None
    for timestamp in frame.index:
        if timestamp.date() == on:
            return timestamp.to_pydatetime()
    return None


def _classify(when: datetime | None) -> Session:
    """Split a reporting timestamp into before-open or after-close."""
    if when is None:
        return "unknown"
    clock = when.time()
    if clock < MARKET_OPEN:
        return "bmo"
    if clock >= MARKET_CLOSE:
        return "amc"
    return "unknown"


def fetch_earnings(config: EarningsConfig, run_date: date, report: RunReport) -> EarningsCalendar:
    """Find watchlist companies reporting on the report date.

    Args:
        config: Earnings settings, including the watchlist and call budget.
        run_date: The report date.
        report: Run report for call accounting and warnings.

    Returns:
        The populated calendar. Individual failures are counted rather than
        raised, so partial Yahoo throttling still yields a usable list.
    """
    calendar = EarningsCalendar()
    if not config.enabled:
        return calendar

    watchlist = config.watchlist[: config.max_calls]
    if len(config.watchlist) > config.max_calls:
        report.warn(f"Earnings watchlist truncated to {config.max_calls} names by max_calls.")

    reporting_today: list[tuple[str, float | None]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_next_earnings_date, t): t for t in watchlist}
        for future in as_completed(futures):
            ticker = futures[future]
            report.record_api_call("yfinance")
            try:
                next_date, estimate = future.result()
            except Exception as exc:  # noqa: BLE001 — one bad name must not stop the sweep
                calendar.failed += 1
                logger.debug("Earnings lookup for %s failed: %s", ticker, exc)
                continue
            calendar.checked += 1
            if next_date == run_date:
                reporting_today.append((ticker, estimate))

    for ticker, estimate in sorted(reporting_today):
        report.record_api_call("yfinance")
        try:
            when = _reporting_time(ticker, run_date)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Earnings time for %s failed: %s", ticker, exc)
            when = None

        event = EarningsEvent(
            ticker=ticker, session=_classify(when), eps_estimate=estimate, when=when
        )
        if event.session == "bmo":
            calendar.before_open.append(event)
        elif event.session == "amc":
            calendar.after_close.append(event)
        else:
            calendar.unscheduled.append(event)

    if calendar.failed:
        report.warn(
            f"Earnings: {calendar.failed}/{len(watchlist)} watchlist names could not be "
            "checked (likely Yahoo throttling); the list may be incomplete."
        )
    logger.info(
        "Earnings: %d reporting today (%d before open, %d after close) from %d checked",
        calendar.total,
        len(calendar.before_open),
        len(calendar.after_close),
        calendar.checked,
    )
    return calendar
