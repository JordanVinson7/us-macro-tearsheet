"""Cross-asset correlations.

One subtlety drives the design: **credit spreads are levels, not prices.** The
percentage change of a spread that moves from 2.70% to 2.80% is +3.7%, which is
meaningless and scales absurdly when the spread is near zero. The economically
meaningful quantity is the *change* in the spread, in basis points. So members
flagged as spreads are differenced while prices are percentage-changed, and the
two are correlated together.
"""

from __future__ import annotations

import pandas as pd


def to_changes(
    series_by_key: dict[str, pd.Series],
    spread_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Convert a mix of price and spread series into comparable daily changes.

    Args:
        series_by_key: Series keyed by display key.
        spread_keys: Keys to difference rather than percentage-change.

    Returns:
        A frame of daily changes, inner-joined on date so every correlation is
        computed over dates where both members actually traded.
    """
    spread_keys = spread_keys or set()
    changes: dict[str, pd.Series] = {}

    for key, series in series_by_key.items():
        values = series.dropna().sort_index()
        if len(values) < 2:
            continue
        if key in spread_keys:
            changes[key] = values.diff().dropna()
        else:
            changes[key] = values.pct_change().dropna() * 100.0

    if not changes:
        return pd.DataFrame()

    frame = pd.DataFrame(changes)
    # Normalising to dates drops intraday timestamp mismatches between sources.
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame


def correlation_matrix(
    changes: pd.DataFrame, window: int, min_observations: int = 30
) -> pd.DataFrame:
    """Correlation matrix of the most recent ``window`` observations.

    Rows where any member is missing are dropped, so every pair is measured
    over the same dates and the matrix stays internally consistent.
    """
    if changes.empty:
        return pd.DataFrame()

    recent = changes.dropna().iloc[-window:]
    if len(recent) < min_observations:
        return pd.DataFrame()
    return recent.corr()


def rolling_correlation(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """Rolling correlation between two series of changes."""
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if aligned.empty:
        return pd.Series(dtype="float64")
    return aligned["left"].rolling(window, min_periods=window).corr(aligned["right"]).dropna()


def latest_correlation(left: pd.Series, right: pd.Series, window: int) -> float | None:
    """Most recent rolling correlation, or None without a full window."""
    values = rolling_correlation(left, right, window)
    if values.empty:
        return None
    return float(values.iloc[-1])


def price_correlation(left_prices: pd.Series, right_prices: pd.Series, window: int) -> pd.Series:
    """Rolling correlation of daily returns between two price series.

    This is the stock-bond correlation measure: a rolling window of SPY and TLT
    daily *returns*, not of their price levels, which would merely reflect
    shared trends.
    """
    changes = to_changes({"left": left_prices, "right": right_prices})
    if changes.empty or changes.shape[1] < 2:
        return pd.Series(dtype="float64")
    return rolling_correlation(changes["left"], changes["right"], window)
