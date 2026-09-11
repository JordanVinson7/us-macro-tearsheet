"""Tests for curve construction and curve-move classification."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tearsheet.analytics.rates import (
    build_curve,
    change_in_bp,
    classify_curve_move,
    inversion_state,
    value_as_of,
)


def yields(values: list[float], start: str = "2026-09-01") -> pd.Series:
    """A yield series in percent, on consecutive business days."""
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


# -- lookups -----------------------------------------------------------------


def test_value_as_of_falls_back_to_the_most_recent_prior_observation():
    """Treasury series skip weekends, so exact-date lookups must not fail."""
    series = pd.Series([4.0, 4.1], index=pd.to_datetime(["2026-09-04", "2026-09-08"]))

    value, observed = value_as_of(series, date(2026, 9, 6))  # a Sunday

    assert value == pytest.approx(4.0)
    assert observed == date(2026, 9, 4)


def test_value_as_of_returns_none_before_the_series_starts():
    assert value_as_of(yields([4.0]), date(2020, 1, 1)) is None


def test_build_curve_sorts_by_tenor_and_skips_missing_series():
    series = {"DGS2": yields([4.43]), "DGS10": yields([4.83])}
    specs = [("DGS10", "10Y", 10.0), ("DGS2", "2Y", 2.0), ("DGS30", "30Y", 30.0)]

    curve = build_curve(series, specs)

    assert [p.label for p in curve] == ["2Y", "10Y"], "missing tenors are omitted, not guessed"
    assert curve[0].yield_pct == pytest.approx(4.43)
    assert curve[0].observation_date == series["DGS2"].index[-1].date()


# -- basis point changes -----------------------------------------------------


def test_change_in_bp_converts_percent_to_basis_points():
    assert change_in_bp(yields([4.00, 4.05]), periods=1) == pytest.approx(5.0)


def test_change_in_bp_handles_a_fall():
    assert change_in_bp(yields([4.10, 4.00]), periods=1) == pytest.approx(-10.0)


def test_change_in_bp_against_a_reference_date():
    series = yields([4.00, 4.10, 4.25])

    assert change_in_bp(series, reference=series.index[0].date()) == pytest.approx(25.0)


def test_change_in_bp_without_enough_history_is_none():
    assert change_in_bp(yields([4.0]), periods=5) is None


# -- curve classification ----------------------------------------------------


@pytest.mark.parametrize(
    ("short_bp", "long_bp", "expected"),
    [
        (-10.0, -5.0, "Bull steepener"),  # yields fell, long end fell less
        (-5.0, -10.0, "Bull flattener"),  # yields fell, long end fell more
        (10.0, 15.0, "Bear steepener"),  # yields rose, long end rose more
        (15.0, 10.0, "Bear flattener"),  # yields rose, short end rose more
        (-10.0, 10.0, "Twist steepener"),  # opposite directions
        (10.0, -10.0, "Twist flattener"),
    ],
)
def test_curve_move_classification(short_bp, long_bp, expected):
    move = classify_curve_move(short_bp, long_bp, threshold_bp=2.0)

    assert move.label == expected


def test_a_tiny_drift_is_reported_as_little_changed():
    """One basis point must not be announced as a bear steepener."""
    move = classify_curve_move(1.0, 1.5, threshold_bp=2.0)

    assert move.direction == "unchanged"
    assert move.label == "Little changed"


def test_a_parallel_shift_is_named_as_such():
    move = classify_curve_move(10.0, 10.0, threshold_bp=2.0)

    assert move.shape == "parallel"
    assert move.label == "Bear parallel shift"


def test_spread_change_is_long_minus_short():
    move = classify_curve_move(5.0, 12.0, threshold_bp=2.0)

    assert move.spread_bp == pytest.approx(7.0)


def test_classification_needs_both_inputs():
    assert classify_curve_move(None, 5.0, 2.0) is None
    assert classify_curve_move(5.0, None, 2.0) is None


# -- inversion ---------------------------------------------------------------


def test_a_negative_spread_is_inverted():
    state = inversion_state(yields([0.10, -0.15]), tolerance_bp=2.0)

    assert state.is_inverted
    assert state.spread_bp == pytest.approx(-15.0)


def test_crossing_into_inversion_is_reported_as_a_transition():
    state = inversion_state(yields([0.20, 0.10, 0.05, -0.10]), tolerance_bp=2.0)

    assert state.transition == "inverted"


def test_crossing_back_out_is_reported_as_un_inverted():
    state = inversion_state(yields([-0.20, -0.10, -0.05, 0.10]), tolerance_bp=2.0)

    assert not state.is_inverted
    assert state.transition == "un-inverted"


def test_a_spread_sitting_at_zero_does_not_flip_state_daily():
    """The tolerance is what stops a flat curve generating a flag every day."""
    state = inversion_state(yields([0.005, -0.005, 0.005, -0.01]), tolerance_bp=2.0)

    assert not state.is_inverted
    assert state.transition is None


def test_a_steadily_inverted_curve_reports_no_transition():
    state = inversion_state(yields([-0.30, -0.28, -0.25, -0.26]), tolerance_bp=2.0)

    assert state.is_inverted
    assert state.transition is None
