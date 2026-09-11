"""Realised volatility and volatility percentiles.

"Realised volatility" here means the annualised standard deviation of daily
simple returns over a trailing window — the standard descriptive measure, not
a forecast. The annualisation factor is passed in rather than hardcoded,
because 252 is right for equities but wrong for crypto, which trades every day.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

from tearsheet.analytics.returns import daily_returns, percentile_rank


def annualise(daily_std: float, periods_per_year: int) -> float:
    """Scale a daily standard deviation to an annual one."""
    return float(daily_std * np.sqrt(periods_per_year))


def realised_volatility(
    series: pd.Series,
    window: int,
    periods_per_year: int = 252,
    suspect: Iterable[date] | None = None,
) -> pd.Series:
    """Rolling annualised realised volatility, in percent.

    Args:
        series: Price history.
        window: Trailing window in observations.
        periods_per_year: Annualisation factor.
        suspect: Dates excluded from the return series.

    Returns:
        Annualised volatility in percent, indexed like the input.
    """
    returns = daily_returns(series, suspect)
    rolling_std = returns.rolling(window, min_periods=window).std(ddof=1)
    return rolling_std * np.sqrt(periods_per_year)


def latest_realised_volatility(
    series: pd.Series,
    window: int,
    periods_per_year: int = 252,
    suspect: Iterable[date] | None = None,
) -> float | None:
    """Most recent annualised realised volatility, or None without a full window."""
    values = realised_volatility(series, window, periods_per_year, suspect).dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def volatility_percentile(
    series: pd.Series,
    window: int,
    lookback: int,
    periods_per_year: int = 252,
    suspect: Iterable[date] | None = None,
) -> float | None:
    """Percentile of current realised volatility within its trailing history.

    Args:
        series: Price history.
        window: Volatility estimation window.
        lookback: How much volatility history forms the distribution.
        periods_per_year: Annualisation factor.
        suspect: Dates excluded from the return series.

    Returns:
        Percentile from 0 to 100, or None without enough history.
    """
    history = realised_volatility(series, window, periods_per_year, suspect).dropna()
    if len(history) < 2:
        return None
    return percentile_rank(float(history.iloc[-1]), history.iloc[-lookback:])


def level_percentile(series: pd.Series, lookback: int, min_observations: int = 2) -> float | None:
    """Percentile of the latest level within its own trailing history.

    Used for credit spreads and the VIX, where the level itself — rather than
    its volatility — is the quantity being ranked.
    """
    values = series.dropna()
    if len(values) < max(2, min_observations):
        return None
    return percentile_rank(float(values.iloc[-1]), values.iloc[-lookback:])


def level_zscore(series: pd.Series, lookback: int, min_observations: int = 2) -> float | None:
    """Z-score of the latest level against its trailing history."""
    from tearsheet.analytics.returns import zscore

    values = series.dropna()
    if len(values) < max(2, min_observations):
        return None
    return zscore(float(values.iloc[-1]), values.iloc[-lookback:])


def intraday_statistics(bars: pd.DataFrame) -> dict[str, float | None]:
    """Summary statistics for one session of minute bars.

    Args:
        bars: Minute bars with ``open``, ``high``, ``low``, ``close``,
            ``volume`` and ``vwap`` columns, indexed by timestamp.

    Returns:
        VWAP, first- and last-hour returns, the times of the session high and
        low, and intraday realised volatility annualised from minute returns.
    """
    if bars.empty or "close" not in bars.columns:
        return {}

    closes = bars["close"]
    volume = bars["volume"] if "volume" in bars.columns else None

    vwap = None
    if volume is not None and float(volume.sum()) > 0 and "vwap" in bars.columns:
        vwap = float((bars["vwap"] * volume).sum() / volume.sum())

    def _window_return(window: pd.Series) -> float | None:
        if len(window) < 2 or window.iloc[0] <= 0:
            return None
        return float((window.iloc[-1] / window.iloc[0] - 1.0) * 100.0)

    minute_returns = closes.pct_change().dropna()
    # 390 minutes per session, 252 sessions per year.
    annualised = (
        float(minute_returns.std(ddof=1) * np.sqrt(390 * 252) * 100.0)
        if len(minute_returns) > 2
        else None
    )

    high_idx = bars["high"].idxmax() if "high" in bars.columns else closes.idxmax()
    low_idx = bars["low"].idxmin() if "low" in bars.columns else closes.idxmin()

    return {
        "vwap": vwap,
        "open": float(bars["open"].iloc[0]) if "open" in bars.columns else None,
        "close": float(closes.iloc[-1]),
        "session_return": _window_return(closes),
        "first_hour_return": _window_return(closes.iloc[:60]),
        "last_hour_return": _window_return(closes.iloc[-60:]),
        "high_time": high_idx,
        "low_time": low_idx,
        "realised_volatility": annualised,
    }
