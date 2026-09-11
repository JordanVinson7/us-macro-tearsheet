"""Section 2 — today's calendar.

Economic releases for today and the rest of the week, the next FOMC decision,
and notable earnings split into before-open and after-close.

FRED does not publish release *times*, so the times shown come from a mapping
of typical ET times in config.yaml and are labelled as typical throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build today's calendar section."""
    config = settings.config
    result = SectionResult(key="calendar", title="Today's calendar")

    calendar = bundle.calendar
    earnings = bundle.earnings

    if calendar is None and earnings is None:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "Neither the economic calendar nor the earnings calendar was available."
        return result

    releases_today = calendar.today if calendar else []
    upcoming = calendar.upcoming if calendar else []

    result.context = {
        "today": [
            {
                "name": entry.name,
                "time": entry.typical_time_et,
                "date": entry.date,
            }
            for entry in releases_today
        ],
        "upcoming": [
            {
                "name": entry.name,
                "time": entry.typical_time_et,
                "date": entry.date,
                "weekday": entry.weekday,
            }
            for entry in upcoming
        ],
        "fomc": calendar.next_fomc if calendar else None,
        "days_to_fomc": calendar.days_to_fomc if calendar else None,
        "fomc_note": config.calendar.fomc_note,
        "earnings": earnings,
        "earnings_threshold_bn": config.earnings.market_cap_threshold_usd / 1e9,
        "lookahead_days": config.calendar.lookahead_days,
    }
    result.sources = list(bundle.sources.get("calendar", [])) + list(
        bundle.sources.get("earnings", [])
    )
    result.notes = [
        "Release dates come from FRED, filtered to a whitelist of major US "
        "releases. Release IDs are resolved by name rather than hardcoded.",
        "FRED does not publish release times. Times shown are typical ET "
        "publication times and are not guaranteed.",
        "FOMC dates are maintained in config.yaml from federalreserve.gov and "
        "must be updated each year.",
        f"Notable earnings are drawn from a curated watchlist of companies above "
        f"roughly ${config.earnings.market_cap_threshold_usd / 1e9:.0f}bn market "
        "capitalisation.",
    ]

    if calendar and calendar.unmatched:
        result.warnings.append(
            f"Calendar whitelist entries not recognised by FRED: {', '.join(calendar.unmatched)}"
        )
    if earnings and not earnings.is_complete:
        result.warnings.append(
            f"{earnings.failed} earnings lookups failed; the list may be incomplete."
        )

    degraded = (
        calendar is None or earnings is None or bool(calendar.unmatched) or not earnings.is_complete
    )
    result.status = SectionStatus.PARTIAL if degraded else SectionStatus.OK
    return result
