"""Jinja filters for number and date formatting.

All number formatting lives here rather than in the templates so that the
conventions are applied once and consistently:

* Changes always carry an explicit ``+`` or ``−`` sign, so colour is never the
  only signal that a number is positive or negative. That matters for readers
  with colour vision deficiency and for greyscale printing.
* A missing value renders as a muted "n/a" rather than as ``0`` or ``None``.
  The analytics return ``None`` when there is not enough history, and that
  distinction has to survive all the way to the page.
* A true minus sign (U+2212) is used rather than a hyphen, which is what makes
  negative numbers line up in a tabular-numeral column.
"""

from __future__ import annotations

from datetime import date, datetime

MINUS = "−"
NA = '<span class="na">n/a</span>'


def _signed(value: float, decimals: int) -> str:
    """Format with an explicit sign and a typographic minus."""
    text = f"{abs(value):,.{decimals}f}"
    return f"+{text}" if value >= 0 else f"{MINUS}{text}"


def number(value: float | None, decimals: int = 2) -> str:
    """A plain number with thousands separators."""
    if value is None:
        return NA
    return f"{value:,.{decimals}f}".replace("-", MINUS)


def signed(value: float | None, decimals: int = 2) -> str:
    """A signed number, without a unit suffix."""
    if value is None:
        return NA
    return _signed(value, decimals)


def percent(value: float | None, decimals: int = 2) -> str:
    """A signed percentage."""
    if value is None:
        return NA
    return f"{_signed(value, decimals)}%"


def basis_points(value: float | None, decimals: int = 0) -> str:
    """A signed basis-point change."""
    if value is None:
        return NA
    return f"{_signed(value, decimals)}bp"


def level(value: float | None, decimals: int = 2) -> str:
    """An unsigned level, e.g. a yield or a spread."""
    if value is None:
        return NA
    return f"{value:,.{decimals}f}".replace("-", MINUS)


def tone(value: float | None) -> str:
    """CSS class for a change: ``pos``, ``neg`` or ``flat``."""
    if value is None:
        return "flat"
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "flat"


def day(value: date | datetime | None, fmt: str = "%d %b %Y") -> str:
    """Format a date, or return an em dash."""
    if value is None:
        return "—"
    return value.strftime(fmt)


def ordinal(value: float | None) -> str:
    """A percentile rendered as ``73rd``."""
    if value is None:
        return NA
    rounded = int(round(value))
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix}"


FILTERS = {
    "number": number,
    "signed": signed,
    "percent": percent,
    "bp": basis_points,
    "level": level,
    "tone": tone,
    "day": day,
    "ordinal": ordinal,
}
"""Filter name to callable, registered on the Jinja environment."""
