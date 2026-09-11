"""Treasury curve construction and curve-move classification.

FRED publishes constant-maturity yields in percent, so a move of 0.05 is five
basis points. Everything user-facing is expressed in basis points, and the
conversion happens once, here.

The bull/bear and steepener/flattener vocabulary is the standard market one:

* **Bull** — yields fell (prices rose). **Bear** — yields rose.
* **Steepener** — the long end rose relative to the short end (2s10s widened).
  **Flattener** — the opposite.
* **Twist** — the two ends moved in opposite directions, which fits neither
  "bull" nor "bear" and is named explicitly rather than forced into one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

#: FRED quotes yields in percent; one percentage point is 100 basis points.
BASIS_POINTS_PER_PERCENT = 100.0

Direction = Literal["bull", "bear", "twist", "unchanged"]
Shape = Literal["steepener", "flattener", "parallel"]


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One point on the Treasury curve.

    Attributes:
        series_id: FRED series ID, e.g. ``DGS10``.
        label: Display label, e.g. ``10Y``.
        tenor_years: Maturity in years, used as the x-axis position.
        yield_pct: The yield in percent.
        observation_date: The date the yield was observed. Shown next to every
            figure because FRED's daily yields lag by a day or two.
    """

    series_id: str
    label: str
    tenor_years: float
    yield_pct: float
    observation_date: date


def value_as_of(series: pd.Series, target: date | None = None) -> tuple[float, date] | None:
    """The last observation at or before a target date.

    Treasury series skip weekends and holidays, so an exact-date lookup fails
    roughly three days in seven. Taking the most recent prior observation is
    both correct and what a reader means by "the 10-year a month ago".
    """
    values = series.dropna().sort_index()
    if values.empty:
        return None
    if target is not None:
        values = values.loc[: pd.Timestamp(target)]
        if values.empty:
            return None
    return float(values.iloc[-1]), values.index[-1].date()


def build_curve(
    series_by_id: dict[str, pd.Series],
    specs: list[tuple[str, str, float]],
    target: date | None = None,
) -> list[CurvePoint]:
    """Assemble a curve from individual tenor series.

    Args:
        series_by_id: Yield series keyed by FRED ID.
        specs: ``(series_id, label, tenor_years)`` triples, in tenor order.
        target: Build the curve as it stood on this date. None means latest.

    Returns:
        Curve points, ascending by tenor. Tenors with no data are omitted
        rather than interpolated — a gap is honest, an invented point is not.
    """
    points: list[CurvePoint] = []
    for series_id, label, tenor_years in specs:
        series = series_by_id.get(series_id)
        if series is None:
            continue
        reading = value_as_of(series, target)
        if reading is None:
            continue
        value, observed = reading
        points.append(
            CurvePoint(
                series_id=series_id,
                label=label,
                tenor_years=tenor_years,
                yield_pct=value,
                observation_date=observed,
            )
        )
    return sorted(points, key=lambda p: p.tenor_years)


def change_in_bp(
    series: pd.Series, periods: int = 1, reference: date | None = None
) -> float | None:
    """Change in basis points over a number of observations or since a date.

    Args:
        series: A yield or spread series, quoted in percent.
        periods: Observations to look back when ``reference`` is not given.
        reference: Compare against the last observation at or before this date.

    Returns:
        The change in basis points, or None without enough history.
    """
    values = series.dropna().sort_index()
    if values.empty:
        return None

    latest = float(values.iloc[-1])

    if reference is not None:
        prior = value_as_of(values, reference)
        if prior is None:
            return None
        base = prior[0]
    else:
        if len(values) <= periods:
            return None
        base = float(values.iloc[-(periods + 1)])

    return float((latest - base) * BASIS_POINTS_PER_PERCENT)


@dataclass(frozen=True, slots=True)
class CurveMove:
    """A classified curve move.

    Attributes:
        direction: Bull, bear, twist, or unchanged.
        shape: Steepener, flattener or parallel.
        short_bp: Change in the short tenor, in basis points.
        long_bp: Change in the long tenor, in basis points.
        spread_bp: Change in the long-minus-short spread.
        label: Display label, e.g. ``"Bear flattener"``.
    """

    direction: Direction
    shape: Shape
    short_bp: float
    long_bp: float
    spread_bp: float
    label: str


def classify_curve_move(
    short_bp: float | None,
    long_bp: float | None,
    threshold_bp: float = 2.0,
) -> CurveMove | None:
    """Classify a curve move from the two tenor changes.

    Args:
        short_bp: Change in the short tenor (typically 2Y), in basis points.
        long_bp: Change in the long tenor (typically 10Y), in basis points.
        threshold_bp: Moves smaller than this count as unchanged, which stops
            a one-basis-point drift being announced as a "bear steepener".

    Returns:
        The classification, or None if either input is missing.
    """
    if short_bp is None or long_bp is None:
        return None

    spread_bp = long_bp - short_bp
    short_moved = abs(short_bp) >= threshold_bp
    long_moved = abs(long_bp) >= threshold_bp

    if not short_moved and not long_moved:
        return CurveMove("unchanged", "parallel", short_bp, long_bp, spread_bp, "Little changed")

    if short_bp < 0 and long_bp < 0:
        direction: Direction = "bull"
    elif short_bp > 0 and long_bp > 0:
        direction = "bear"
    else:
        direction = "twist"

    if abs(spread_bp) < threshold_bp:
        shape: Shape = "parallel"
    elif spread_bp > 0:
        shape = "steepener"
    else:
        shape = "flattener"

    label = (
        f"{direction.capitalize()} {shape}"
        if shape != "parallel"
        else f"{direction.capitalize()} parallel shift"
    )
    return CurveMove(direction, shape, short_bp, long_bp, spread_bp, label)


@dataclass(frozen=True, slots=True)
class InversionState:
    """Whether a curve spread is inverted, and whether that just changed.

    Attributes:
        spread_bp: The current spread in basis points.
        is_inverted: Whether the spread is below zero by more than the tolerance.
        transition: ``"inverted"`` or ``"un-inverted"`` if the state changed
            within the lookback, otherwise None.
    """

    spread_bp: float
    is_inverted: bool
    transition: str | None


def inversion_state(
    spread: pd.Series, tolerance_bp: float = 2.0, lookback: int = 5
) -> InversionState | None:
    """Assess a curve spread for inversion and recent transitions.

    Args:
        spread: A spread series in percent, e.g. FRED ``T10Y2Y``.
        tolerance_bp: Dead zone around zero, so a spread hovering at zero does
            not flip state every day.
        lookback: How many observations back to look for a transition.

    Returns:
        The state, or None if the series is empty.
    """
    values = spread.dropna().sort_index()
    if values.empty:
        return None

    current_bp = float(values.iloc[-1]) * BASIS_POINTS_PER_PERCENT
    is_inverted = current_bp < -tolerance_bp

    transition = None
    if len(values) > 1:
        window = values.iloc[-(lookback + 1) : -1] * BASIS_POINTS_PER_PERCENT
        was_inverted = bool((window < -tolerance_bp).any())
        was_positive = bool((window > tolerance_bp).any())
        if is_inverted and was_positive:
            transition = "inverted"
        elif not is_inverted and was_inverted:
            transition = "un-inverted"

    return InversionState(current_bp, is_inverted, transition)
