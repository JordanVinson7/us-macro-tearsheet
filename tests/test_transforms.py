"""Tests for FRED series transforms and the macro dashboard reading."""

from __future__ import annotations

import pandas as pd
import pytest

from tearsheet.analytics.transforms import apply_transform, read_series


def monthly(values: list[float], start: str = "2025-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def weekly(values: list[float], start: str = "2026-01-03") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="W-SAT"))


def test_level_transform_applies_only_the_scale():
    """ICSA is published in persons; scale 0.001 displays it in thousands."""
    result = apply_transform(monthly([206_000.0]), "level", scale=0.001)

    assert result.iloc[-1] == pytest.approx(206.0)


def test_month_on_month_percentage():
    assert apply_transform(monthly([100.0, 102.0]), "mom_pct").iloc[-1] == pytest.approx(2.0)


def test_year_on_year_uses_twelve_monthly_periods():
    series = monthly([100.0 + i for i in range(13)])  # 100 -> 112

    result = apply_transform(series, "yoy_pct", frequency="monthly")

    assert result.iloc[-1] == pytest.approx(12.0)


def test_year_on_year_uses_four_periods_for_quarterly_data():
    series = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 110.0],
        index=pd.date_range("2025-01-01", periods=5, freq="QS"),
    )

    result = apply_transform(series, "yoy_pct", frequency="quarterly")

    assert result.iloc[-1] == pytest.approx(10.0)


def test_three_month_average_change_smooths_payrolls():
    """A single payrolls month is mostly noise; the 3m average is the read."""
    series = monthly([100.0, 130.0, 160.0, 190.0])  # monthly changes of 30 each

    result = apply_transform(series, "diff_3m_avg")

    assert result.iloc[-1] == pytest.approx(30.0)


def test_four_week_average_smooths_claims():
    series = weekly([200.0, 210.0, 190.0, 200.0])

    result = apply_transform(series, "avg_4w")

    assert result.iloc[-1] == pytest.approx(200.0)


def test_diff_transform():
    assert apply_transform(monthly([100.0, 97.0]), "diff").iloc[-1] == pytest.approx(-3.0)


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_transform(monthly([1.0, 2.0]), "wishful_thinking")


def test_empty_series_transforms_to_empty():
    assert apply_transform(pd.Series(dtype="float64"), "yoy_pct").empty


# -- read_series -------------------------------------------------------------


def test_read_series_reports_latest_prior_and_change():
    reading = read_series(monthly([4.0, 4.1, 4.3]), transform="level")

    assert reading.latest == pytest.approx(4.3)
    assert reading.prior == pytest.approx(4.1)
    assert reading.change == pytest.approx(0.2, abs=1e-9)
    assert reading.is_available


def test_read_series_computes_yoy_from_the_underlying_level():
    """The YoY of a YoY rate is a second derivative, which nobody wants."""
    reading = read_series(monthly([100.0 + i for i in range(13)]), transform="level")

    assert reading.yoy == pytest.approx(12.0)


def test_read_series_omits_yoy_for_a_rate_transform():
    reading = read_series(monthly([100.0 + i for i in range(14)]), transform="yoy_pct")

    assert reading.yoy is None
    assert reading.latest is not None


def test_read_series_of_an_empty_series_is_unavailable():
    reading = read_series(pd.Series(dtype="float64"))

    assert not reading.is_available
    assert reading.observation_date is None


def test_read_series_records_the_observation_date():
    series = monthly([1.0, 2.0])

    assert read_series(series).observation_date == series.index[-1].date()


def test_a_rate_reports_yoy_in_percentage_points():
    """Unemployment 4.3% -> 4.1% is -0.2pp, not '-4.65%'."""
    series = monthly([4.3] + [4.2] * 11 + [4.1])

    reading = read_series(series, transform="level", units="%")

    assert reading.yoy == pytest.approx(-0.2, abs=1e-9)
    assert reading.yoy_units == "pp"


def test_a_level_series_reports_yoy_as_a_percentage():
    series = monthly([100.0] * 12 + [110.0])

    reading = read_series(series, transform="level", units="k SAAR")

    assert reading.yoy == pytest.approx(10.0)
    assert reading.yoy_units == "%"


@pytest.mark.parametrize(
    ("units", "values", "expected"),
    [
        ("%", [4.0, 4.1], "pp"),  # a rate
        ("% q/q", [1.5, 2.1], "pp"),  # a growth rate
        ("index", [50.0, 55.0], "%"),  # strictly positive index
        ("$bn", [100.0, 110.0], "%"),
        ("diffusion", [-5.0, 20.0], "pts"),  # crosses zero
        ("index", [-0.5, -0.6], "pts"),  # NFCI is negative
    ],
)
def test_yoy_convention_is_inferred(units, values, expected):
    from tearsheet.analytics.transforms import yoy_convention

    assert yoy_convention(units, pd.Series(values)) == expected


def test_a_diffusion_index_crossing_zero_reports_points_not_percent():
    """0.7 -> 47.4 is +46.7 points, not '+6671%'."""
    series = monthly([0.7] + [-2.0] * 11 + [47.4])

    reading = read_series(series, transform="level", units="diffusion")

    assert reading.yoy == pytest.approx(46.7)
    assert reading.yoy_units == "pts"


def test_an_explicit_override_beats_inference():
    """Inference only sees the fetched window, so config always wins."""
    series = monthly([0.7] + [10.0] * 11 + [47.4])  # never goes negative

    inferred = read_series(series, transform="level", units="diffusion")
    overridden = read_series(series, transform="level", units="diffusion", yoy_units="pts")

    assert inferred.yoy_units == "%"
    assert overridden.yoy == pytest.approx(46.7)


def test_yoy_can_be_suppressed_entirely():
    reading = read_series(monthly([1.0] * 13), transform="level", yoy_units="none")

    assert reading.yoy is None
