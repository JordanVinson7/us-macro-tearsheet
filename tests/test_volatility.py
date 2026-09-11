"""Tests for realised volatility and volatility percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tearsheet.analytics.volatility import (
    annualise,
    intraday_statistics,
    latest_realised_volatility,
    level_percentile,
    level_zscore,
    realised_volatility,
    volatility_percentile,
)


def prices(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)))


def test_annualisation_scales_by_root_time():
    assert annualise(1.0, 252) == pytest.approx(np.sqrt(252))
    assert annualise(1.0, 365) == pytest.approx(np.sqrt(365))


def test_constant_growth_has_zero_realised_volatility():
    """Identical daily returns mean no dispersion, so volatility is zero."""
    series = prices([100.0 * (1.01**i) for i in range(30)])

    assert latest_realised_volatility(series, window=20) == pytest.approx(0.0, abs=1e-9)


def test_realised_volatility_matches_a_hand_computed_case():
    series = prices([100.0, 110.0, 99.0, 108.9])  # returns +10%, -10%, +10%

    result = latest_realised_volatility(series, window=3, periods_per_year=252)

    expected = pd.Series([10.0, -10.0, 10.0]).std(ddof=1) * np.sqrt(252)
    assert result == pytest.approx(float(expected), abs=1e-6)


def test_volatility_needs_a_full_window():
    assert latest_realised_volatility(prices([100.0, 101.0]), window=20) is None


def test_a_contract_roll_does_not_inflate_volatility():
    """The whole point of tracking suspect dates: one roll would dominate."""
    quiet = [100.0 * (1.001**i) for i in range(40)]
    series = prices([*quiet, quiet[-1] * 0.6, quiet[-1] * 0.6 * 1.001])
    roll_date = series.index[40].date()

    contaminated = latest_realised_volatility(series, window=20)
    cleaned = latest_realised_volatility(series, window=20, suspect=[roll_date])

    assert contaminated > 50.0
    assert cleaned < 1.0


def test_volatility_percentile_is_high_after_a_volatility_spike():
    calm = [100.0 * (1.0005**i) for i in range(200)]
    noisy = [calm[-1] * (1.05 if i % 2 else 0.95) for i in range(30)]
    series = prices([*calm, *noisy])

    result = volatility_percentile(series, window=20, lookback=252)

    assert result is not None
    assert result > 90.0


def test_level_percentile_and_zscore_on_a_known_distribution():
    series = pd.Series(
        [float(i) for i in range(1, 11)], index=pd.bdate_range("2026-01-01", periods=10)
    )

    assert level_percentile(series, lookback=10) == pytest.approx(100.0)
    assert level_zscore(series, lookback=10) == pytest.approx(1.4863, abs=1e-4)


def test_intraday_statistics_on_a_synthetic_session():
    index = pd.date_range("2026-09-10 09:30", periods=120, freq="1min", tz="America/New_York")
    closes = np.linspace(100.0, 110.0, 120)
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.full(120, 1000.0),
            "vwap": closes,
        },
        index=index,
    )

    stats = intraday_statistics(bars)

    assert stats["session_return"] == pytest.approx(10.0)
    # Bar 59 of a 120-bar linear ramp from 100 to 110 closes at 104.958.
    assert stats["first_hour_return"] == pytest.approx(4.9580, abs=1e-3)
    assert stats["high_time"] == index[-1]
    assert stats["low_time"] == index[0]
    assert stats["vwap"] == pytest.approx(float(closes.mean()))


def test_intraday_statistics_of_an_empty_session_is_empty():
    assert intraday_statistics(pd.DataFrame()) == {}


def test_realised_volatility_series_is_indexed_like_its_input():
    series = prices([100.0 + i for i in range(40)])

    result = realised_volatility(series, window=20)

    assert result.dropna().index[-1] == series.index[-1]
