"""Return, moving-average and z-score calculations.

Every function here is pure: it takes a pandas Series and primitives, and
returns a value or a Series. No configuration objects, no I/O, no logging.
That is what makes the whole module testable on synthetic data in
milliseconds, and it is why the thresholds live in config.yaml and are passed
in by the caller rather than read here.

Two conventions apply throughout:

* **Insufficient data returns ``None``, never a wrong number.** A 1-year return
  computed from three months of history is worse than no number at all, so the
  caller gets ``None`` and the page renders "n/a".
* **Suspect dates poison any window containing them.** A continuous futures
  contract roll makes the return *across* that date meaningless (see
  :mod:`tearsheet.data.quality`), so a window spanning one returns ``None``
  rather than a plausible-looking artefact.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd


def _clean(series: pd.Series) -> pd.Series:
    """Drop missing observations and sort chronologically."""
    return series.dropna().sort_index()


def _suspect_in_window(series: pd.Series, periods: int, suspect: Iterable[date] | None) -> bool:
    """Whether any suspect date falls inside the trailing ``periods`` window."""
    if not suspect:
        return False
    window_start = series.index[-(periods + 1)].date()
    window_end = series.index[-1].date()
    return any(window_start < d <= window_end for d in suspect)


def simple_return(
    series: pd.Series,
    periods: int,
    suspect: Iterable[date] | None = None,
) -> float | None:
    """Percentage return over a trailing number of observations.

    Args:
        series: Price history, ascending.
        periods: How many observations to look back.
        suspect: Dates whose returns are known to be artefacts.

    Returns:
        The return in percent, or None if there is not enough history or the
        window spans a suspect date.
    """
    values = _clean(series)
    if len(values) <= periods or periods < 1:
        return None
    if _suspect_in_window(values, periods, suspect):
        return None

    start, end = values.iloc[-(periods + 1)], values.iloc[-1]
    if start <= 0:
        return None
    return float((end / start - 1.0) * 100.0)


def window_returns(
    series: pd.Series,
    windows: dict[str, int],
    suspect: Iterable[date] | None = None,
) -> dict[str, float | None]:
    """Returns over several named windows, e.g. ``{"1D": 1, "1M": 21}``."""
    return {name: simple_return(series, periods, suspect) for name, periods in windows.items()}


def ytd_return(
    series: pd.Series,
    as_of: date | None = None,
    suspect: Iterable[date] | None = None,
) -> float | None:
    """Return since the final close of the previous calendar year.

    The previous year's last close is the correct base, not the first close of
    January: using January's open would silently discard the turn-of-year move.
    """
    values = _clean(series)
    if values.empty:
        return None

    end_date = as_of or values.index[-1].date()
    prior_year_end = pd.Timestamp(date(end_date.year - 1, 12, 31))

    base_candidates = values.loc[:prior_year_end]
    if base_candidates.empty:
        return None
    base = base_candidates.iloc[-1]
    if base <= 0:
        return None

    current = values.loc[: pd.Timestamp(end_date)]
    if current.empty:
        return None

    if suspect and any(base_candidates.index[-1].date() < d <= end_date for d in suspect):
        return None

    return float((current.iloc[-1] / base - 1.0) * 100.0)


def daily_returns(
    series: pd.Series,
    suspect: Iterable[date] | None = None,
) -> pd.Series:
    """Daily percentage returns, with suspect observations removed.

    Args:
        series: Price history.
        suspect: Dates whose returns are artefacts; these are dropped so they
            cannot distort volatility, correlation or z-score distributions.

    Returns:
        Daily returns in percent.
    """
    values = _clean(series)
    returns = values.pct_change().dropna() * 100.0
    if suspect:
        excluded = {pd.Timestamp(d) for d in suspect}
        returns = returns[~returns.index.isin(excluded)]
    return returns


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing window."""
    return _clean(series).rolling(window, min_periods=window).mean()


def distance_from_ma(series: pd.Series, window: int, min_observations: int = 0) -> float | None:
    """Percentage distance of the latest price from its moving average.

    Positive means the price is above the average.
    """
    values = _clean(series)
    if len(values) < max(window, min_observations):
        return None
    average = values.iloc[-window:].mean()
    if average <= 0:
        return None
    return float((values.iloc[-1] / average - 1.0) * 100.0)


def is_above_ma(series: pd.Series, window: int) -> bool | None:
    """Whether the latest price sits above its moving average."""
    distance = distance_from_ma(series, window)
    return None if distance is None else distance > 0


def share_above_ma(series_by_key: dict[str, pd.Series], window: int) -> float | None:
    """Share of a group trading above its moving average, in percent.

    Used for sector breadth. Members without enough history are excluded from
    both the numerator and the denominator rather than counted as False.
    """
    verdicts = [is_above_ma(s, window) for s in series_by_key.values()]
    usable = [v for v in verdicts if v is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable) * 100.0)


def rolling_extreme(series: pd.Series, window: int) -> tuple[float, float] | None:
    """The highest and lowest close over a trailing window."""
    values = _clean(series)
    if len(values) < 2:
        return None
    tail = values.iloc[-window:]
    return float(tail.max()), float(tail.min())


def drawdown_from_high(series: pd.Series, window: int, min_observations: int = 0) -> float | None:
    """Percentage below the trailing high. Returned as a negative number."""
    values = _clean(series)
    if len(values) < max(2, min_observations):
        return None
    high = values.iloc[-window:].max()
    if high <= 0:
        return None
    return float((values.iloc[-1] / high - 1.0) * 100.0)


def at_new_extreme(series: pd.Series, window: int) -> str | None:
    """Whether the latest close is a new trailing high or low.

    Returns:
        ``"high"``, ``"low"``, or None. Requires a full window of history, so a
        short series cannot produce a spurious "52-week high" on week three.
    """
    values = _clean(series)
    if len(values) < window:
        return None
    tail = values.iloc[-window:]
    latest = values.iloc[-1]
    if latest >= tail.max():
        return "high"
    if latest <= tail.min():
        return "low"
    return None


def ma_cross(series: pd.Series, window: int, tolerance_pct: float) -> str | None:
    """Detect a moving-average cross on the latest observation.

    A cross is only reported when the price closed on the other side of the
    average yesterday, and has cleared it by more than ``tolerance_pct`` today.
    The tolerance stops a price oscillating around its average from producing a
    "cross" flag every single day.

    Returns:
        ``"above"`` for an upward cross, ``"below"`` for a downward one, else None.
    """
    values = _clean(series)
    if len(values) < window + 1:
        return None

    average = values.rolling(window, min_periods=window).mean()
    today, yesterday = values.iloc[-1], values.iloc[-2]
    ma_today, ma_yesterday = average.iloc[-1], average.iloc[-2]
    if pd.isna(ma_today) or pd.isna(ma_yesterday) or ma_today <= 0:
        return None

    clearance = abs(today / ma_today - 1.0) * 100.0
    if clearance < tolerance_pct:
        return None

    if yesterday <= ma_yesterday and today > ma_today:
        return "above"
    if yesterday >= ma_yesterday and today < ma_today:
        return "below"
    return None


def zscore(value: float, distribution: pd.Series | np.ndarray) -> float | None:
    """Z-score of a value against a distribution.

    Returns None when the distribution is too small or has no dispersion —
    dividing by a zero standard deviation would produce infinity.
    """
    values = pd.Series(distribution).dropna()
    if len(values) < 2:
        return None
    deviation = float(values.std(ddof=1))
    if deviation <= 0 or not np.isfinite(deviation):
        return None
    return float((value - float(values.mean())) / deviation)


def return_zscore(
    series: pd.Series,
    lookback: int,
    suspect: Iterable[date] | None = None,
    min_observations: int = 30,
) -> float | None:
    """Z-score of the latest daily return against its trailing distribution.

    Args:
        series: Price history.
        lookback: How many trailing daily returns form the distribution.
        suspect: Dates excluded from both the value and the distribution.
        min_observations: Below this many returns, None is returned.

    Returns:
        The z-score, or None if there is not enough clean history.
    """
    returns = daily_returns(series, suspect)
    if len(returns) < max(2, min_observations):
        return None
    window = returns.iloc[-lookback:]
    return zscore(float(returns.iloc[-1]), window)


def percentile_rank(value: float, distribution: pd.Series | np.ndarray) -> float | None:
    """Percentile of a value within a distribution, from 0 to 100.

    Uses the "share of observations at or below" definition, which is the one
    readers expect: a 95th-percentile reading means only 5% of history was
    higher.
    """
    values = pd.Series(distribution).dropna()
    if values.empty:
        return None
    return float((values <= value).sum() / len(values) * 100.0)
