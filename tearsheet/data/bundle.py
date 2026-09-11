"""Assembly of every input the tear-sheet needs into one :class:`DataBundle`.

This module is the seam between "talking to the internet" and "computing
things". Sections never call a data client directly; they read from the bundle.
That keeps the analytics pure and testable, and it is what allows ``--mock`` to
substitute a recorded bundle for a live one with no other code changes.

Every acquisition step is individually guarded. A failure in one domain — say
Polygon's minute bars — leaves that field empty and records a warning; it never
prevents the other domains from being collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tearsheet.config import Settings
from tearsheet.data.calendar import CALENDAR_SOURCE, CalendarEntry, EconomicCalendar, build_calendar
from tearsheet.data.earnings import (
    EARNINGS_SOURCE,
    EarningsCalendar,
    EarningsEvent,
    fetch_earnings,
)
from tearsheet.data.fallbacks import resolve_prices
from tearsheet.data.fixtures import FixtureStore
from tearsheet.data.fred import FRED_SOURCE, FredClient, FredSeriesMeta
from tearsheet.data.news import collect_headlines
from tearsheet.data.polygon import POLYGON_SOURCE, PolygonClient
from tearsheet.data.quality import apply_price_checks, compute_staleness
from tearsheet.data.sessions import last_completed_session
from tearsheet.data.yfinance import fetch_quotes
from tearsheet.models import NewsItem, PriceData, Quote, SourceRef, Staleness
from tearsheet.observability import RunReport, get_logger

if TYPE_CHECKING:
    from tearsheet.models import RunOptions

logger = get_logger("data.bundle")


@dataclass
class DataBundle:
    """Every input needed to build the tear-sheet.

    Attributes:
        run_date: The report date.
        session: The last completed trading session the data describes.
        prices: Daily OHLCV history with per-ticker provenance.
        quotes: Pre-open quotes for the overnight panel.
        fred: FRED observations keyed by series ID.
        fred_meta: Validated FRED metadata keyed by series ID.
        fred_invalid: Configured series IDs FRED did not recognise.
        intraday: Previous-session minute bars keyed by ticker.
        calendar: Economic releases and the FOMC schedule.
        earnings: Notable earnings for the report date.
        headlines: Deduplicated market headlines.
        staleness: Staleness verdicts keyed by series ID.
        sources: Provenance by domain, aggregated into the footer.
    """

    run_date: date
    session: date
    prices: PriceData = field(default_factory=PriceData)
    quotes: dict[str, Quote] = field(default_factory=dict)
    fred: dict[str, pd.Series] = field(default_factory=dict)
    fred_meta: dict[str, FredSeriesMeta] = field(default_factory=dict)
    fred_invalid: list[str] = field(default_factory=list)
    intraday: dict[str, pd.DataFrame] = field(default_factory=dict)
    calendar: EconomicCalendar | None = None
    earnings: EarningsCalendar | None = None
    headlines: list[NewsItem] = field(default_factory=list)
    staleness: dict[str, Staleness] = field(default_factory=dict)
    sources: dict[str, list[SourceRef]] = field(default_factory=dict)

    def add_source(self, domain: str, source: SourceRef) -> None:
        """Record which source supplied a domain of data."""
        refs = self.sources.setdefault(domain, [])
        if source not in refs:
            refs.append(source)

    def series(self, series_id: str) -> pd.Series | None:
        """A FRED series by ID, or None when unavailable."""
        return self.fred.get(series_id)

    def has(self, *series_ids: str) -> bool:
        """Whether every named FRED series is present and non-empty."""
        return all(sid in self.fred and not self.fred[sid].empty for sid in series_ids)


def refresh_staleness(bundle: DataBundle, settings: Settings, report: RunReport) -> None:
    """Recompute staleness for every FRED series and warn about the stale ones.

    Run after the calendar so the exact "has the scheduled release passed?"
    test can be used wherever a next-release date is known.
    """
    next_release = bundle.calendar.next_release_by_series if bundle.calendar else {}

    for spec in settings.config.fred.all_series():
        values = bundle.fred.get(spec.id)
        observed = values.index[-1].date() if values is not None and not values.empty else None
        bundle.staleness[spec.id] = compute_staleness(
            observed,
            spec.frequency,
            bundle.run_date,
            settings.config.quality,
            next_release.get(spec.id),
        )

    stale = sorted(sid for sid, s in bundle.staleness.items() if s.is_stale)
    if stale:
        report.warn(f"{len(stale)} FRED series are stale: {', '.join(stale)}")


def _overnight_tickers(settings: Settings) -> list[str]:
    """Symbols that need a live pre-open quote."""
    universe = settings.config.universe
    groups = (universe.futures, universe.fx, universe.commodities, universe.crypto)
    return list(dict.fromkeys(i.ticker for group in groups for i in group))


def collect(settings: Settings, options: RunOptions, report: RunReport) -> DataBundle:
    """Acquire everything the tear-sheet needs.

    Args:
        settings: Configuration and credentials.
        options: Resolved run options; ``--mock`` replays fixtures instead.
        report: Run report for warnings, timings and call accounting.

    Returns:
        The assembled bundle. Domains that could not be collected are left
        empty rather than raising.
    """
    if options.is_mock:
        return load_from_fixtures(settings, options, report)
    return collect_live(settings, options, report)


def collect_live(settings: Settings, options: RunOptions, report: RunReport) -> DataBundle:
    """Acquire data from the live APIs."""
    config = settings.config
    session = last_completed_session(options.run_date, config.schedule.exchange_calendar)
    bundle = DataBundle(run_date=options.run_date, session=session)

    fred = _make_fred(settings, report)
    polygon = _make_polygon(settings, report)

    _collect_fred(bundle, settings, fred, report)
    _collect_prices(bundle, settings, fred, polygon, report)
    _collect_quotes(bundle, settings, report)
    _collect_intraday(bundle, settings, polygon, report)
    _collect_calendar(bundle, settings, fred, report)
    _collect_earnings(bundle, settings, report)
    _collect_headlines(bundle, settings, polygon, report)

    # Deliberately last: staleness is far more accurate once the calendar has
    # supplied each series' next scheduled release date.
    refresh_staleness(bundle, settings, report)

    if fred is not None:
        fred.close()
    if polygon is not None:
        report.timings["polygon:rate_limited"] = polygon.seconds_rate_limited
        polygon.close()

    return bundle


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _make_fred(settings: Settings, report: RunReport) -> FredClient | None:
    """Build a FRED client, or warn and return None when the key is absent."""
    if not settings.secrets.has("fred_api_key"):
        report.warn("FRED_API_KEY is not set; rates, credit and macro will be unavailable.")
        return None
    return FredClient(settings.config.sources.fred, settings.secrets.reveal("fred_api_key"), report)


def _make_polygon(settings: Settings, report: RunReport) -> PolygonClient | None:
    """Build a Polygon client, or warn and return None when the key is absent."""
    if not settings.secrets.has("polygon_api_key"):
        report.warn("POLYGON_API_KEY is not set; the intraday panel will be unavailable.")
        return None
    return PolygonClient(
        settings.config.sources.polygon, settings.secrets.reveal("polygon_api_key"), report
    )


# ---------------------------------------------------------------------------
# Individually guarded acquisition steps
# ---------------------------------------------------------------------------


def _collect_fred(
    bundle: DataBundle, settings: Settings, fred: FredClient | None, report: RunReport
) -> None:
    """Validate and fetch every configured FRED series."""
    if fred is None:
        return
    config = settings.config
    specs = config.fred.all_series()

    with report.timed("fetch:fred"):
        try:
            if config.sources.fred.validate_series_on_startup:
                bundle.fred_meta, bundle.fred_invalid = fred.validate_series(specs)
                specs = [s for s in specs if s.id not in bundle.fred_invalid]

            start = bundle.session - timedelta(days=365 * config.display.small_multiples_years)
            bundle.fred = fred.fetch_many(specs, start, bundle.fred_meta or None)
        except Exception as exc:  # noqa: BLE001 — isolated domain
            report.warn(f"FRED collection failed: {exc}")
            return

    bundle.add_source("fred", FRED_SOURCE)


def _collect_prices(
    bundle: DataBundle,
    settings: Settings,
    fred: FredClient | None,
    polygon: PolygonClient | None,
    report: RunReport,
) -> None:
    """Resolve daily price history through the fallback chain."""
    with report.timed("fetch:prices"):
        try:
            prices = resolve_prices(
                settings.config.universe.all_yf_tickers(),
                settings.config,
                report,
                as_of=bundle.session,
                polygon=polygon,
                fred=fred,
            )
            bundle.prices = apply_price_checks(prices, settings.config, report)
        except Exception as exc:  # noqa: BLE001 — isolated domain
            report.warn(f"Price collection failed: {exc}")
            return

    for source in dict.fromkeys(bundle.prices.sources.values()):
        bundle.add_source("prices", source)


def _collect_quotes(bundle: DataBundle, settings: Settings, report: RunReport) -> None:
    """Fetch pre-open quotes for the overnight panel."""
    with report.timed("fetch:quotes"):
        try:
            bundle.quotes = fetch_quotes(
                _overnight_tickers(settings), report, settings.config.meta.timezone
            )
        except Exception as exc:  # noqa: BLE001 — isolated domain
            report.warn(f"Overnight quotes failed: {exc}")
            return

    for quote in bundle.quotes.values():
        bundle.add_source("overnight", quote.source)


def _collect_intraday(
    bundle: DataBundle, settings: Settings, polygon: PolygonClient | None, report: RunReport
) -> None:
    """Fetch previous-session minute bars from Polygon."""
    if polygon is None:
        return
    with report.timed("fetch:intraday"):
        for ticker in settings.config.universe.intraday.tickers:
            try:
                bars = polygon.fetch_minute_bars(
                    ticker, bundle.session, settings.config.meta.timezone
                )
            except Exception as exc:  # noqa: BLE001 — one ticker must not stop the panel
                report.warn(f"Polygon minute bars for {ticker} failed: {exc}")
                continue
            if not bars.empty:
                bundle.intraday[ticker] = bars

    if bundle.intraday:
        bundle.add_source("intraday", POLYGON_SOURCE)


def _collect_calendar(
    bundle: DataBundle, settings: Settings, fred: FredClient | None, report: RunReport
) -> None:
    """Build the economic release calendar."""
    if fred is None:
        return
    with report.timed("fetch:calendar"):
        try:
            bundle.calendar = build_calendar(
                fred,
                settings.config,
                bundle.run_date,
                report,
                series_ids=[s.id for s in settings.config.fred.macro],
            )
        except Exception as exc:  # noqa: BLE001 — isolated domain
            report.warn(f"Calendar build failed: {exc}")
            return

    for source in bundle.calendar.sources:
        bundle.add_source("calendar", source)


def _collect_earnings(bundle: DataBundle, settings: Settings, report: RunReport) -> None:
    """Find notable companies reporting today."""
    with report.timed("fetch:earnings"):
        try:
            bundle.earnings = fetch_earnings(settings.config.earnings, bundle.run_date, report)
        except Exception as exc:  # noqa: BLE001 — isolated domain
            report.warn(f"Earnings lookup failed: {exc}")
            return

    if bundle.earnings is not None:
        bundle.add_source("earnings", bundle.earnings.source)


def _collect_headlines(
    bundle: DataBundle, settings: Settings, polygon: PolygonClient | None, report: RunReport
) -> None:
    """Gather and deduplicate market headlines."""
    with report.timed("fetch:headlines"):
        try:
            bundle.headlines = collect_headlines(
                settings.config.sources.news,
                report,
                polygon=polygon if settings.config.sources.polygon.news_enabled else None,
                timezone=settings.config.meta.timezone,
            )
        except Exception as exc:  # noqa: BLE001 — isolated domain
            report.warn(f"Headline collection failed: {exc}")
            return

    for item in bundle.headlines:
        if item.source is not None:
            bundle.add_source("headlines", item.source)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def save_fixtures(bundle: DataBundle, root: Path) -> dict[str, int]:
    """Record a live bundle so ``--mock`` can replay it offline.

    Args:
        bundle: A bundle collected from the live APIs.
        root: The fixtures directory.

    Returns:
        Counts of what was written, for the run summary.
    """
    store = FixtureStore(root)
    store.save_frames("prices", bundle.prices.frames)
    store.save_series("fred", bundle.fred)
    store.save_frames("intraday", bundle.intraday)

    store.save_json(
        "meta",
        {
            "run_date": bundle.run_date,
            "session": bundle.session,
            "price_sources": {
                t: [s.name, s.detail, s.is_fallback] for t, s in bundle.prices.sources.items()
            },
            "missing": bundle.prices.missing,
            "suspect_returns": bundle.prices.suspect_returns,
            "fred_invalid": bundle.fred_invalid,
        },
    )
    store.save_json(
        "quotes",
        [
            {
                "ticker": q.ticker,
                "price": q.price,
                "previous_close": q.previous_close,
                "as_of": q.as_of,
            }
            for q in bundle.quotes.values()
        ],
    )
    if bundle.calendar is not None:
        store.save_json(
            "calendar",
            {
                "today": [_entry_to_dict(e) for e in bundle.calendar.today],
                "upcoming": [_entry_to_dict(e) for e in bundle.calendar.upcoming],
                "days_to_fomc": bundle.calendar.days_to_fomc,
                "next_release_by_series": bundle.calendar.next_release_by_series,
                "unmatched": bundle.calendar.unmatched,
            },
        )
    if bundle.earnings is not None:
        store.save_json(
            "earnings",
            {
                "checked": bundle.earnings.checked,
                "failed": bundle.earnings.failed,
                "events": [
                    {
                        "ticker": e.ticker,
                        "session": e.session,
                        "eps_estimate": e.eps_estimate,
                        "when": e.when,
                    }
                    for group in (
                        bundle.earnings.before_open,
                        bundle.earnings.after_close,
                        bundle.earnings.unscheduled,
                    )
                    for e in group
                ],
            },
        )

    store.save_json(
        "headlines",
        [
            {
                "title": h.title,
                "publisher": h.publisher,
                "url": h.url,
                "published_at": h.published_at,
                "tickers": list(h.tickers),
            }
            for h in bundle.headlines
        ],
    )

    counts = {
        "prices": len(bundle.prices.frames),
        "fred": len(bundle.fred),
        "intraday": len(bundle.intraday),
        "quotes": len(bundle.quotes),
        "headlines": len(bundle.headlines),
    }
    store.write_manifest(bundle.run_date, bundle.session, counts)
    logger.info("Recorded fixtures to %s: %s", root, counts)
    return counts


def load_from_fixtures(settings: Settings, options: RunOptions, report: RunReport) -> DataBundle:
    """Rebuild a bundle from recorded fixtures, making no network calls."""
    root = settings.project_root / "fixtures"
    store = FixtureStore(root)

    if not store.exists:
        report.warn(
            f"No fixtures found in {root}. Record them first with: python -m tearsheet run --record"
        )
        return DataBundle(
            run_date=options.run_date,
            session=last_completed_session(options.run_date),
        )

    manifest = store.manifest()
    bundle = DataBundle(
        run_date=date.fromisoformat(manifest["run_date"]),
        session=date.fromisoformat(manifest["session"]),
    )

    meta = store.load_json("meta", {}) or {}
    bundle.prices = PriceData(as_of=bundle.session, missing=meta.get("missing", []))
    bundle.prices.frames = store.load_frames("prices")
    for ticker, (name, detail, is_fallback) in (meta.get("price_sources") or {}).items():
        bundle.prices.sources[ticker] = SourceRef(name, detail, is_fallback)
    bundle.prices.suspect_returns = {
        ticker: [date.fromisoformat(d) for d in dates]
        for ticker, dates in (meta.get("suspect_returns") or {}).items()
    }

    bundle.fred = store.load_series("fred")
    bundle.fred_invalid = meta.get("fred_invalid", [])
    bundle.intraday = store.load_frames("intraday")

    bundle.quotes = {
        row["ticker"]: Quote(
            ticker=row["ticker"],
            price=row["price"],
            previous_close=row.get("previous_close"),
            as_of=None,
            source=SourceRef("fixtures", "recorded quote"),
        )
        for row in store.load_json("quotes", []) or []
    }
    bundle.headlines = [
        NewsItem(
            title=row["title"],
            publisher=row["publisher"],
            url=row["url"],
            published_at=None,
            tickers=tuple(row.get("tickers", ())),
            source=SourceRef("fixtures", "recorded headline"),
        )
        for row in store.load_json("headlines", []) or []
    ]

    _load_calendar_fixture(bundle, store, settings)
    _load_earnings_fixture(bundle, store)
    refresh_staleness(bundle, settings, report)

    # Domain provenance is not itself recorded in the fixtures, so it is
    # reconstructed from what was actually loaded. The footer then reports the
    # same sources a live run would, marked as replayed.
    if bundle.fred:
        bundle.add_source("fred", FRED_SOURCE)
    if bundle.intraday:
        bundle.add_source("intraday", POLYGON_SOURCE)
    if bundle.calendar:
        bundle.add_source("calendar", CALENDAR_SOURCE)
    if bundle.earnings:
        bundle.add_source("earnings", EARNINGS_SOURCE)
    for quote in bundle.quotes.values():
        bundle.add_source("overnight", quote.source)
    for source in dict.fromkeys(bundle.prices.sources.values()):
        bundle.add_source("prices", source)

    bundle.add_source("fixtures", SourceRef("fixtures", f"recorded {manifest.get('recorded_at')}"))
    logger.info(
        "Loaded fixtures recorded %s: %s",
        manifest.get("recorded_at"),
        manifest.get("counts"),
    )
    return bundle


def _entry_to_dict(entry: CalendarEntry) -> dict:
    """Serialise one calendar entry for the fixture store."""
    return {
        "name": entry.name,
        "release_id": entry.release_id,
        "date": entry.date,
        "typical_time_et": entry.typical_time_et,
        "is_today": entry.is_today,
    }


def _entry_from_dict(payload: dict) -> CalendarEntry:
    """Rebuild a calendar entry from the fixture store."""
    return CalendarEntry(
        name=payload["name"],
        release_id=payload["release_id"],
        date=date.fromisoformat(payload["date"]),
        typical_time_et=payload.get("typical_time_et"),
        is_today=payload["is_today"],
    )


def _load_calendar_fixture(bundle: DataBundle, store: FixtureStore, settings: Settings) -> None:
    """Rebuild the economic calendar from fixtures, if one was recorded."""
    payload = store.load_json("calendar")
    if not payload:
        return
    bundle.calendar = EconomicCalendar(
        run_date=bundle.run_date,
        today=[_entry_from_dict(e) for e in payload.get("today", [])],
        upcoming=[_entry_from_dict(e) for e in payload.get("upcoming", [])],
        next_fomc=settings.config.calendar.next_fomc(bundle.run_date),
        days_to_fomc=payload.get("days_to_fomc"),
        next_release_by_series={
            k: date.fromisoformat(v)
            for k, v in (payload.get("next_release_by_series") or {}).items()
        },
        unmatched=payload.get("unmatched", []),
    )


def _load_earnings_fixture(bundle: DataBundle, store: FixtureStore) -> None:
    """Rebuild the earnings calendar from fixtures, if one was recorded."""
    payload = store.load_json("earnings")
    if not payload:
        return
    calendar = EarningsCalendar(checked=payload.get("checked", 0), failed=payload.get("failed", 0))
    for row in payload.get("events", []):
        event = EarningsEvent(
            ticker=row["ticker"],
            session=row["session"],
            eps_estimate=row.get("eps_estimate"),
            when=datetime.fromisoformat(row["when"]) if row.get("when") else None,
        )
        {"bmo": calendar.before_open, "amc": calendar.after_close}.get(
            event.session, calendar.unscheduled
        ).append(event)
    bundle.earnings = calendar
