"""Section 3 — overnight and futures.

Equity futures against the prior cash close, plus overnight moves in Treasury
futures, the dollar, oil, gold and bitcoin.

Live pre-open levels come from Yahoo Finance rather than Polygon. This is the
project's one deliberate departure from "intraday data comes from Polygon":
the free Polygon tier exposes no snapshot endpoint, so there is no way to get
a pre-open quote from it. The exception is recorded in the section footer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.sections.base import close_series

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle

#: Non-equity instruments shown in the overnight panel, in display order.
_OVERNIGHT_EXTRAS = ["ZN=F", "DX-Y.NYB", "CL=F", "GC=F", "BTC-USD"]


def _row(
    ticker: str, label: str, bundle: DataBundle, cash_ticker: str | None = None
) -> dict | None:
    """Build one overnight row, or None when no quote is available."""
    quote = bundle.quotes.get(ticker)
    if quote is None:
        return None

    prior_close = quote.previous_close
    basis = "prior close"

    # For equity futures the meaningful comparison is against the cash index's
    # closing print, which is what "futures are up 0.4%" normally means.
    if cash_ticker:
        cash = close_series(bundle, cash_ticker)
        if cash is not None and not cash.empty:
            prior_close = float(cash.iloc[-1])
            basis = f"{cash_ticker} close"

    change_pct = None
    if prior_close and prior_close > 0:
        change_pct = (quote.price / prior_close - 1.0) * 100.0

    return {
        "ticker": ticker,
        "label": label,
        "price": quote.price,
        "prior_close": prior_close,
        "basis": basis,
        "change_pct": change_pct,
        "as_of": quote.as_of,
    }


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the overnight and futures section."""
    config = settings.config
    result = SectionResult(key="overnight", title="Overnight & futures")

    if not bundle.quotes:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No pre-open quotes were available."
        return result

    futures_rows = [
        row
        for instrument in config.universe.futures
        if instrument.cash
        and (row := _row(instrument.ticker, instrument.name, bundle, instrument.cash))
    ]

    other_rows = []
    for ticker in _OVERNIGHT_EXTRAS:
        instrument = config.universe.find(ticker)
        label = instrument.name if instrument else ticker
        if row := _row(ticker, label, bundle):
            other_rows.append(row)

    if not futures_rows and not other_rows:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No overnight quotes could be matched to instruments."
        return result

    timestamps = [r["as_of"] for r in futures_rows + other_rows if r["as_of"]]
    result.context = {
        "futures_rows": futures_rows,
        "other_rows": other_rows,
        "quoted_at": max(timestamps) if timestamps else None,
        "cash_session": bundle.session,
    }
    result.sources = list(bundle.sources.get("overnight", []))
    result.notes = [
        "Equity futures are shown against the prior cash index close, which is "
        "what a pre-open futures move is normally quoted against.",
        "Pre-open quotes come from Yahoo Finance. Polygon's free tier has no "
        "snapshot endpoint, so this is a documented exception to the project's "
        "rule that intraday data comes from Polygon.",
        "Quotes are indicative and may be delayed.",
    ]
    result.status = (
        SectionStatus.OK
        if len(futures_rows) >= 4 and len(other_rows) >= 4
        else SectionStatus.PARTIAL
    )
    return result
