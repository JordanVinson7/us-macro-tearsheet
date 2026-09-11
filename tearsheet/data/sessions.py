"""US equity trading-session helpers.

A pre-open report is always *about* the last completed session, never about a
partially-formed bar for today. Centralising that here means the same rule is
used by the data layer, the analytics and the Phase 6 schedule guard, and that
a local afternoon run behaves exactly like a real 08:00 ET run.
"""

from __future__ import annotations

import functools
from datetime import date, datetime

import exchange_calendars as xcals

from tearsheet.observability import get_logger

logger = get_logger("data.sessions")


@functools.lru_cache(maxsize=4)
def get_calendar(name: str = "XNYS") -> xcals.ExchangeCalendar:
    """Return an exchange calendar, cached because construction is expensive."""
    return xcals.get_calendar(name)


def is_session(day: date, calendar_name: str = "XNYS") -> bool:
    """Whether ``day`` is a trading session (not a weekend or market holiday)."""
    return get_calendar(calendar_name).is_session(day.isoformat())


def previous_session(day: date, calendar_name: str = "XNYS") -> date:
    """The last trading session strictly before ``day``."""
    calendar = get_calendar(calendar_name)
    return calendar.previous_session(day.isoformat()).date()


def last_completed_session(
    as_of: datetime | date,
    calendar_name: str = "XNYS",
) -> date:
    """The most recent session whose closing bell has already rung.

    Args:
        as_of: The moment to evaluate from. A timezone-aware datetime is
            compared against the actual close; a plain date is treated as
            "before today's open", which is the pre-open case.
        calendar_name: Exchange calendar identifier.

    Returns:
        The date of the last completed trading session.
    """
    calendar = get_calendar(calendar_name)

    if isinstance(as_of, datetime):
        day = as_of.date()
        if calendar.is_session(day.isoformat()):
            close = calendar.session_close(day.isoformat())
            if as_of >= close.to_pydatetime():
                return day
        return previous_session(day, calendar_name)

    # A bare date means "as at the start of that day", so today never counts.
    return previous_session(as_of, calendar_name)


def sessions_between(start: date, end: date, calendar_name: str = "XNYS") -> list[date]:
    """Every trading session in the inclusive range ``start``..``end``."""
    calendar = get_calendar(calendar_name)
    return [ts.date() for ts in calendar.sessions_in_range(start.isoformat(), end.isoformat())]


def session_n_ago(reference: date, n: int, calendar_name: str = "XNYS") -> date:
    """The session ``n`` trading days before ``reference``."""
    calendar = get_calendar(calendar_name)
    session = calendar.date_to_session(reference.isoformat(), direction="previous")
    index = calendar.sessions.get_loc(session)
    return calendar.sessions[max(0, index - n)].date()
