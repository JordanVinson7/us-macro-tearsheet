"""Economic release calendar and FOMC schedule.

FRED is the only source here. Two details matter:

* ``include_release_dates_with_no_data=true`` is essential. Without it FRED
  returns dates only for releases that have *already published*, which is the
  exact opposite of what a forward-looking calendar needs.
* **Release IDs are resolved by name**, never hardcoded, so the whitelist in
  config.yaml stays human-readable and cannot silently point at the wrong
  release after a FRED reorganisation.

FRED does not publish release *times*, so the times shown come from a small
mapping of typical ET times in config.yaml and are labelled as typical
wherever they appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from tearsheet.config import CalendarConfig, Config, FomcMeeting
from tearsheet.data.fred import FredClient, ReleaseDate
from tearsheet.models import SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.calendar")

CALENDAR_SOURCE = SourceRef(name="FRED", detail="releases/dates endpoint")


@dataclass(frozen=True, slots=True)
class CalendarEntry:
    """One scheduled economic release.

    Attributes:
        name: FRED's official release name.
        release_id: FRED release ID.
        date: Scheduled publication date.
        typical_time_et: Usual publication time in ET, or None if unknown.
            This is a typical time, not a guaranteed one.
        is_today: Whether the release falls on the report date.
    """

    name: str
    release_id: int
    date: date
    typical_time_et: str | None
    is_today: bool

    @property
    def weekday(self) -> str:
        """Short weekday label, e.g. ``"Mon"``."""
        return self.date.strftime("%a")


@dataclass
class EconomicCalendar:
    """Everything the calendar section needs.

    Attributes:
        run_date: The report date.
        today: Whitelisted releases scheduled for today.
        upcoming: Whitelisted releases for the rest of the lookahead window.
        next_fomc: The next FOMC decision, if one is configured.
        days_to_fomc: Calendar days until that decision.
        next_release_by_series: FRED series ID to its next scheduled release
            date, used by the macro dashboard.
        unmatched: Whitelist entries FRED did not recognise.
        sources: Provenance for the footer.
    """

    run_date: date
    today: list[CalendarEntry] = field(default_factory=list)
    upcoming: list[CalendarEntry] = field(default_factory=list)
    next_fomc: FomcMeeting | None = None
    days_to_fomc: int | None = None
    next_release_by_series: dict[str, date] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=lambda: [CALENDAR_SOURCE])


def _resolve_whitelist(
    releases: dict[str, int], whitelist: list[str], report: RunReport
) -> tuple[dict[int, str], list[str]]:
    """Map whitelisted release names to IDs, reporting any that do not exist."""
    resolved: dict[int, str] = {}
    unmatched: list[str] = []
    for name in whitelist:
        release_id = releases.get(name)
        if release_id is None:
            unmatched.append(name)
            report.warn(f"Calendar whitelist entry {name!r} does not match any FRED release name.")
            continue
        resolved[release_id] = name
    return resolved, unmatched


def build_calendar(
    fred: FredClient,
    config: Config,
    run_date: date,
    report: RunReport,
    series_ids: list[str] | None = None,
) -> EconomicCalendar:
    """Assemble the economic calendar.

    Args:
        fred: A FRED client.
        config: Full configuration.
        run_date: The report date.
        report: Run report for warnings.
        series_ids: Macro series to resolve next-release dates for.

    Returns:
        The populated calendar. On failure the calendar comes back empty rather
        than raising, so the section degrades on its own.
    """
    calendar_config = config.calendar
    horizon = max(calendar_config.lookahead_days, calendar_config.next_release_horizon_days)
    calendar = EconomicCalendar(run_date=run_date)

    calendar.next_fomc = calendar_config.next_fomc(run_date)
    if calendar.next_fomc:
        calendar.days_to_fomc = (calendar.next_fomc.date - run_date).days

    try:
        releases = fred.fetch_releases()
    except Exception as exc:  # noqa: BLE001 — the section degrades on its own
        report.warn(f"Could not fetch FRED releases: {exc}")
        return calendar

    resolved, calendar.unmatched = _resolve_whitelist(
        releases, calendar_config.release_whitelist, report
    )

    # Resolve which release publishes each macro series first, so that every
    # release we care about — whitelisted or not — is fetched in one pass.
    series_release_ids: dict[str, int] = {}
    for series_id in series_ids or []:
        lookup = fred.fetch_series_release(series_id)
        if lookup is not None:
            series_release_ids[series_id] = lookup[0]

    wanted = set(resolved) | set(series_release_ids.values())
    dates_by_release: dict[int, list[date]] = {}
    all_dates: list[ReleaseDate] = []
    window_end = run_date + timedelta(days=horizon)

    for release_id in sorted(wanted):
        entries = fred.fetch_release_dates(release_id, run_date, window_end)
        all_dates.extend(entries)
        dates_by_release[release_id] = sorted(e.date for e in entries)

    calendar.today, calendar.upcoming = _split_entries(
        all_dates, resolved, calendar_config, run_date
    )

    calendar.next_release_by_series = {
        series_id: dates_by_release[release_id][0]
        for series_id, release_id in series_release_ids.items()
        if dates_by_release.get(release_id)
    }

    logger.info(
        "Calendar: %d release(s) today, %d upcoming, next FOMC %s",
        len(calendar.today),
        len(calendar.upcoming),
        calendar.next_fomc.date if calendar.next_fomc else "unknown",
    )
    return calendar


def _split_entries(
    all_dates: list[ReleaseDate],
    resolved: dict[int, str],
    calendar_config: CalendarConfig,
    run_date: date,
) -> tuple[list[CalendarEntry], list[CalendarEntry]]:
    """Split whitelisted release dates into today's and the rest of the window."""
    lookahead_end = run_date + timedelta(days=calendar_config.lookahead_days)
    times = calendar_config.typical_times_et

    entries = [
        CalendarEntry(
            name=resolved[item.release_id],
            release_id=item.release_id,
            date=item.date,
            typical_time_et=times.get(resolved[item.release_id]),
            is_today=item.date == run_date,
        )
        for item in all_dates
        if item.release_id in resolved and run_date <= item.date <= lookahead_end
    ]
    entries.sort(key=lambda e: (e.date, e.typical_time_et or "99:99", e.name))

    return (
        [e for e in entries if e.is_today],
        [e for e in entries if not e.is_today],
    )
