"""Tests for return and moving-average analytics, on synthetic data."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tearsheet.analytics.returns import (
    at_new_extreme,
    daily_returns,
    distance_from_ma,
    drawdown_from_high,
    ma_cross,
    percentile_rank,
    return_zscore,
    share_above_ma,
    simple_return,
    window_returns,
    ytd_return,
    zscore,
)


def prices(values: list[float], start: str = "2026-01-01") -> pd.Series:
    """A price series on consecutive business days."""
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


# -- simple returns ----------------------------------------------------------


def test_simple_return_is_percentage_change_over_the_window():
    assert simple_return(prices([100.0, 105.0, 110.0]), 2) == pytest.approx(10.0)


def test_one_day_return_uses_the_previous_close():
    assert simple_return(prices([100.0, 105.0, 110.0]), 1) == pytest.approx(4.7619, abs=1e-4)


def test_insufficient_history_returns_none_not_a_wrong_number():
    """A 1-year return from three days of data is worse than no number."""
    assert simple_return(prices([100.0, 101.0, 102.0]), 252) is None


def test_negative_return_is_signed_correctly():
    assert simple_return(prices([100.0, 80.0]), 1) == pytest.approx(-20.0)


def test_window_returns_maps_every_named_window():
    series = prices([100.0 + i for i in range(30)])

    result = window_returns(series, {"1D": 1, "1W": 5, "1Y": 252})

    assert result["1D"] is not None
    assert result["1W"] is not None
    assert result["1Y"] is None


# -- suspect returns (futures contract rolls) --------------------------------


def test_a_window_spanning_a_contract_roll_returns_none():
    """A roll makes the return across it an artefact, not a market move."""
    series = prices([115.0, 114.0, 78.0, 77.0])
    roll_date = series.index[2].date()  # the 114 -> 78 step happens on this date

    assert simple_return(series, 2, suspect=[roll_date]) is None
    assert simple_return(series, 3, suspect=[roll_date]) is None


def test_a_window_clear_of_the_roll_is_still_computed():
    """Only windows containing the bad return are poisoned, not every window."""
    series = prices([115.0, 114.0, 78.0, 77.0])
    roll_date = series.index[2].date()

    # The final day's return (78 -> 77) sits entirely after the roll.
    assert simple_return(series, 1, suspect=[roll_date]) == pytest.approx(-1.282, abs=1e-3)


def test_daily_returns_drop_suspect_observations():
    series = prices([115.0, 114.0, 78.0, 77.0])
    roll_date = series.index[2].date()

    clean = daily_returns(series, suspect=[roll_date])

    assert len(clean) == 2
    assert clean.min() > -5.0, "the -31% roll artefact must be gone"


# -- year to date ------------------------------------------------------------


def test_ytd_uses_the_previous_year_final_close_as_the_base():
    """Using January's first close would discard the turn-of-year move."""
    index = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"])
    series = pd.Series([90.0, 100.0, 105.0, 110.0], index=index)

    assert ytd_return(series, date(2026, 1, 5)) == pytest.approx(10.0)


def test_ytd_returns_none_without_previous_year_data():
    assert ytd_return(prices([100.0, 110.0], start="2026-01-02")) is None


# -- moving averages ---------------------------------------------------------


def test_distance_from_ma_is_signed_percentage():
    series = prices([10.0] * 9 + [11.0])

    # Mean of the last 10 is 10.1; the latest is 11.
    assert distance_from_ma(series, 10) == pytest.approx(8.9109, abs=1e-4)


def test_distance_from_ma_needs_a_full_window():
    assert distance_from_ma(prices([10.0, 11.0]), 50) is None


def test_share_above_ma_ignores_members_without_enough_history():
    group = {
        "A": prices([1.0] * 9 + [2.0]),  # above
        "B": prices([2.0] * 9 + [1.0]),  # below
        "C": prices([1.0, 2.0]),  # too short to judge
    }

    assert share_above_ma(group, 10) == pytest.approx(50.0)


def test_share_above_ma_is_none_when_nothing_is_measurable():
    assert share_above_ma({"A": prices([1.0, 2.0])}, 50) is None


# -- extremes ----------------------------------------------------------------


def test_new_high_and_low_are_detected():
    assert at_new_extreme(prices([1.0, 2.0, 3.0]), 3) == "high"
    assert at_new_extreme(prices([3.0, 2.0, 1.0]), 3) == "low"
    assert at_new_extreme(prices([1.0, 3.0, 2.0]), 3) is None


def test_a_short_series_cannot_report_a_52_week_high():
    """Three weeks of data must not produce a '52-week high' flag."""
    assert at_new_extreme(prices([1.0, 2.0, 3.0]), 252) is None


def test_drawdown_from_high_is_negative():
    assert drawdown_from_high(prices([100.0, 120.0, 90.0]), 252) == pytest.approx(-25.0)


# -- moving-average crosses --------------------------------------------------


def test_upward_ma_cross_is_detected():
    series = prices([10.0] * 10 + [20.0])

    assert ma_cross(series, 10, tolerance_pct=0.15) == "above"


def test_downward_ma_cross_is_detected():
    series = prices([20.0] * 10 + [10.0])

    assert ma_cross(series, 10, tolerance_pct=0.15) == "below"


def test_a_price_hugging_its_average_does_not_cross_every_day():
    """Without a tolerance this oscillation would flag a cross daily."""
    series = prices([10.0] * 10 + [10.005])

    assert ma_cross(series, 10, tolerance_pct=0.15) is None


# -- z-scores and percentiles ------------------------------------------------


def test_zscore_against_a_known_distribution():
    distribution = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # mean 3, sd 1.5811

    assert zscore(5.0, distribution) == pytest.approx(1.2649, abs=1e-4)
    assert zscore(3.0, distribution) == pytest.approx(0.0)


def test_zscore_of_a_flat_distribution_is_none_not_infinity():
    assert zscore(5.0, pd.Series([2.0, 2.0, 2.0])) is None


def test_zscore_needs_at_least_two_observations():
    assert zscore(5.0, pd.Series([2.0])) is None


def test_return_zscore_flags_an_outlier_day():
    quiet = [100.0 * (1.001**i) for i in range(60)]
    series = prices([*quiet, quiet[-1] * 1.10])  # a 10% day after 0.1% days

    result = return_zscore(series, lookback=252, min_observations=30)

    assert result is not None
    assert result > 3.0


def test_return_zscore_needs_minimum_observations():
    assert return_zscore(prices([100.0, 101.0, 102.0]), 252, min_observations=30) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, 10.0), (5.0, 50.0), (10.0, 100.0)],
)
def test_percentile_rank(value, expected):
    distribution = pd.Series([float(i) for i in range(1, 11)])

    assert percentile_rank(value, distribution) == pytest.approx(expected)


def test_percentile_rank_of_empty_distribution_is_none():
    assert percentile_rank(1.0, pd.Series(dtype="float64")) is None
