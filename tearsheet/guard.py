"""The schedule guard: does this invocation publish a report?

GitHub's cron runs in UTC and ignores daylight saving, so the workflow fires
twice — 12:00 and 13:00 UTC — and lets this guard decide which firing counts.
In EDT those are 08:00 and 09:00 ET; in EST, 07:00 and 08:00. Both land inside
the configured window, so **idempotency is what stops two sheets being
published**: the second firing sees today's archive file already committed and
stands down.

Scheduled jobs on GitHub can also start minutes or tens of minutes late under
load, which is why the window is hours wide rather than a target time.

The guard is a pure function of (clock, calendar, filesystem) so its whole
decision table is unit-testable without a scheduler.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tearsheet.config import Settings
from tearsheet.data.sessions import is_session
from tearsheet.models import GuardDecision, RunOptions
from tearsheet.observability import get_logger

logger = get_logger("guard")


def report_path(settings: Settings, options: RunOptions, run_date: date) -> Path:
    """Where the archived sheet for a date would live."""
    return options.output_dir / settings.config.output.archive_dir / f"{run_date.isoformat()}.html"


def already_published(settings: Settings, options: RunOptions, run_date: date) -> bool:
    """Whether a sheet for this date has already been written."""
    return report_path(settings, options, run_date).exists()


def evaluate(settings: Settings, options: RunOptions, now: datetime | None = None) -> GuardDecision:
    """Decide whether this invocation should produce a report.

    Checks, in order: an explicit override, the trading calendar, the publish
    window, and finally whether today's sheet already exists.

    Args:
        settings: Configuration, supplying the window and calendar.
        options: Run options; ``force`` bypasses every check.
        now: The moment to evaluate. Defaults to the current time in the
            configured timezone. Injectable so the decision table can be tested.

    Returns:
        The decision, with a reason that is logged either way.
    """
    schedule = settings.config.schedule
    timezone = ZoneInfo(settings.config.meta.timezone)
    now_et = now.astimezone(timezone) if now else datetime.now(timezone)
    today = now_et.date()

    if options.force:
        return GuardDecision(True, "--force was given; every check bypassed.", now_et)

    if schedule.weekdays_only and today.weekday() >= 5:
        return GuardDecision(False, f"{today:%A} is a weekend; markets are closed.", now_et)

    if schedule.skip_market_holidays and not is_session(today, schedule.exchange_calendar):
        return GuardDecision(
            False,
            f"{today:%d %b %Y} is not a {schedule.exchange_calendar} trading session.",
            now_et,
        )

    clock = now_et.time()
    if not (schedule.window_start <= clock <= schedule.window_end):
        return GuardDecision(
            False,
            f"{clock:%H:%M} ET is outside the publish window "
            f"({schedule.window_start:%H:%M}–{schedule.window_end:%H:%M} ET).",
            now_et,
        )

    if already_published(settings, options, today):
        return GuardDecision(
            False,
            f"A sheet for {today:%d %b %Y} already exists at "
            f"{report_path(settings, options, today)}; nothing to do.",
            now_et,
        )

    return GuardDecision(
        True,
        f"{clock:%H:%M} ET is inside the publish window and no sheet exists for today.",
        now_et,
    )
