"""Transforms that turn a raw FRED series into its display value.

The transform for each series is declared in config.yaml, so adding an
indicator never requires new code — only a new row. The vocabulary is
deliberately small:

``level``
    The series as published.
``diff``
    Change from the previous observation.
``mom_pct``
    Month-on-month percentage change.
``yoy_pct``
    Year-on-year percentage change, the standard way to read price indices.
``diff_3m_avg``
    Three-month average of the monthly change — how payrolls are normally
    discussed, because a single month is mostly noise.
``avg_4w``
    Four-week moving average, the standard smoothing for jobless claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

#: Observations per year for each FRED reporting frequency.
_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}


def _periods_per_year(frequency: str) -> int:
    """Observations per year for a reporting cadence."""
    return _PERIODS_PER_YEAR.get(frequency, 12)


def apply_transform(
    series: pd.Series, transform: str, frequency: str = "monthly", scale: float = 1.0
) -> pd.Series:
    """Convert a raw series into its display form.

    Args:
        series: Raw observations from FRED.
        transform: One of the transform names documented above.
        frequency: Reporting cadence, used to size year-on-year windows.
        scale: Multiplier applied to level-like results, e.g. 0.001 to display
            thousands as units.

    Returns:
        The transformed series.

    Raises:
        ValueError: If the transform name is not recognised.
    """
    values = series.dropna().sort_index()
    if values.empty:
        return values

    periods = _periods_per_year(frequency)

    if transform == "level":
        return values * scale
    if transform == "diff":
        return values.diff() * scale
    if transform == "mom_pct":
        return values.pct_change() * 100.0
    if transform == "yoy_pct":
        return values.pct_change(periods) * 100.0
    if transform == "diff_3m_avg":
        return values.diff().rolling(3, min_periods=3).mean() * scale
    if transform == "avg_4w":
        return values.rolling(4, min_periods=4).mean() * scale

    raise ValueError(f"Unknown transform: {transform!r}")


def yoy_convention(units: str, series: pd.Series, override: str = "auto") -> str:
    """Decide how a series' year-on-year change should be expressed.

    Percentage change is only meaningful for a strictly positive series. Two
    families break it:

    * **Rates** — unemployment moving 4.3% to 4.1% is a fall of 0.2 percentage
      points. Calling it "-4.65%" is arithmetically true and financially
      meaningless.
    * **Series that cross zero** — a diffusion index going from 0.7 to 47.4 is
      a rise of 46.7 points, not "+6671%". Dividing by a near-zero base makes
      the number explode and tells the reader nothing.

    Inference covers the common cases, but it can only see the history that
    happened to be fetched: a diffusion index that stayed positive over the
    window would be misread as percentage-changeable. An explicit ``override``
    from config.yaml therefore always wins.

    Args:
        units: The unit label.
        series: The raw history, inspected for non-positive values.
        override: ``"auto"`` to infer, or an explicit ``"%"``/``"pp"``/``"pts"``/
            ``"none"``.

    Returns:
        ``"pp"`` for percentage points, ``"pts"`` for index points, ``"%"`` for
        a percentage change, or ``"none"`` to omit year-on-year entirely.
    """
    if override != "auto":
        return override
    if units.strip().startswith("%"):
        return "pp"
    values = series.dropna()
    if values.empty or bool((values <= 0).any()):
        return "pts"
    return "%"


@dataclass(frozen=True, slots=True)
class SeriesReading:
    """A macro series reduced to what the dashboard displays.

    Attributes:
        latest: Most recent transformed value.
        prior: The observation before it.
        change: Latest minus prior.
        yoy: Year-on-year change of the underlying level, where meaningful.
        yoy_units: ``"%"`` for a percentage change, ``"pp"`` for percentage
            points, ``"pts"`` for index points. Displayed alongside the figure
            so the conventions can never be confused.
        observation_date: Date of the latest observation.
    """

    latest: float | None
    prior: float | None
    change: float | None
    yoy: float | None
    observation_date: date | None
    yoy_units: str = "%"

    @property
    def is_available(self) -> bool:
        """Whether there is a value to display."""
        return self.latest is not None


def read_series(
    series: pd.Series,
    transform: str = "level",
    frequency: str = "monthly",
    scale: float = 1.0,
    units: str = "",
    yoy_units: str = "auto",
) -> SeriesReading:
    """Reduce a raw series to the latest value, prior value, change and YoY.

    Year-on-year is computed from the *underlying level*, not the transformed
    value: the YoY change of a YoY rate is a second derivative and is not what
    the macro dashboard is asking for. Series that are themselves rates are
    differenced into percentage points rather than percentage-changed.

    Args:
        series: Raw FRED observations.
        transform: Display transform to apply.
        frequency: Reporting cadence.
        scale: Multiplier for level-like results.
        units: Unit label, used to infer the year-on-year convention.
        yoy_units: Explicit override for that convention.

    Returns:
        The reading, with ``yoy_units`` stating which convention was used.
    """
    raw = series.dropna().sort_index()
    if raw.empty:
        return SeriesReading(None, None, None, None, None)

    transformed = apply_transform(raw, transform, frequency, scale).dropna()
    if transformed.empty:
        return SeriesReading(None, None, None, None, raw.index[-1].date())

    latest = float(transformed.iloc[-1])
    prior = float(transformed.iloc[-2]) if len(transformed) > 1 else None

    yoy = None
    convention = yoy_convention(units, raw, yoy_units)
    periods = _periods_per_year(frequency)
    if (
        convention != "none"
        and transform in {"level", "diff_3m_avg", "avg_4w"}
        and len(raw) > periods
    ):
        current, base = float(raw.iloc[-1]), float(raw.iloc[-(periods + 1)])
        yoy = (current / base - 1.0) * 100.0 if convention == "%" else (current - base) * scale

    return SeriesReading(
        latest=latest,
        prior=prior,
        change=None if prior is None else latest - prior,
        yoy=yoy,
        observation_date=transformed.index[-1].date(),
        yoy_units=convention,
    )
