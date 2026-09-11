"""Tests for cross-asset correlation handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tearsheet.analytics.correlations import (
    correlation_matrix,
    latest_correlation,
    price_correlation,
    rolling_correlation,
    to_changes,
)


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)))


def test_prices_are_percentage_changed():
    changes = to_changes({"SPX": series([100.0, 110.0])})

    assert changes["SPX"].iloc[-1] == pytest.approx(10.0)


def test_spreads_are_differenced_not_percentage_changed():
    """2.70% -> 2.80% is +10bp, not +3.7%, which would scale absurdly near zero."""
    changes = to_changes({"HY": series([2.70, 2.80])}, spread_keys={"HY"})

    assert changes["HY"].iloc[-1] == pytest.approx(0.10)


def test_mixed_prices_and_spreads_are_joined_on_date():
    changes = to_changes(
        {"SPX": series([100.0, 110.0, 121.0]), "HY": series([2.7, 2.8, 2.9])},
        spread_keys={"HY"},
    )

    assert list(changes.columns) == ["SPX", "HY"]
    assert len(changes) == 2


def test_a_series_too_short_to_difference_is_dropped():
    assert to_changes({"X": series([1.0])}).empty


def test_perfectly_correlated_series_give_a_correlation_of_one():
    base = np.linspace(1.0, 2.0, 60) + np.sin(np.arange(60))
    changes = pd.DataFrame(
        {"A": base, "B": base * 3.0}, index=pd.bdate_range("2026-01-01", periods=60)
    )

    matrix = correlation_matrix(changes, window=60, min_observations=30)

    assert matrix.loc["A", "B"] == pytest.approx(1.0)


def test_inversely_correlated_series_give_minus_one():
    base = np.sin(np.arange(60))
    changes = pd.DataFrame({"A": base, "B": -base}, index=pd.bdate_range("2026-01-01", periods=60))

    assert correlation_matrix(changes, 60, 30).loc["A", "B"] == pytest.approx(-1.0)


def test_too_few_observations_yields_an_empty_matrix():
    changes = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [2.0, 1.0]}, index=pd.bdate_range("2026-01-01", periods=2)
    )

    assert correlation_matrix(changes, window=60, min_observations=30).empty


def test_empty_input_yields_an_empty_matrix():
    assert correlation_matrix(pd.DataFrame(), window=60).empty


def test_rolling_correlation_needs_a_full_window():
    left = series(list(np.sin(np.arange(40))))
    right = series(list(np.cos(np.arange(40))))

    assert rolling_correlation(left, right, window=60).empty
    assert not rolling_correlation(left, right, window=20).empty


def test_latest_correlation_returns_the_final_value():
    left = series(list(np.sin(np.arange(40))))
    right = series(list(np.sin(np.arange(40))))

    assert latest_correlation(left, right, window=20) == pytest.approx(1.0)


def test_stock_bond_correlation_uses_returns_not_levels():
    """Two rising price series are not correlated if their returns are opposed."""
    rng = np.random.default_rng(7)
    shocks = rng.normal(0, 0.01, 80)
    spy = pd.Series(100 * np.cumprod(1 + shocks), index=pd.bdate_range("2026-01-01", periods=80))
    tlt = pd.Series(100 * np.cumprod(1 - shocks), index=spy.index)

    result = price_correlation(spy, tlt, window=60)

    assert float(result.iloc[-1]) < -0.95
