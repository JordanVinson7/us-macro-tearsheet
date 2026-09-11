"""Tests for the rule-based 'What stood out' panel."""

from __future__ import annotations

from tearsheet.analytics.flags import (
    Flag,
    flag_credit_move,
    flag_curve_transition,
    flag_extreme,
    flag_ma_cross,
    flag_return_zscore,
    flag_vix_term_structure,
    rank_flags,
)


def test_a_large_move_is_flagged():
    flag = flag_return_zscore("^GSPC", "S&P 500", zscore=2.6, return_pct=-2.4, threshold=2.0)

    assert flag is not None
    assert "S&P 500" in flag.text
    assert "fell" in flag.text
    assert "\u22122.40%" in flag.text, "a typographic minus, matching the tables"


def test_an_ordinary_move_is_not_flagged():
    assert flag_return_zscore("^GSPC", "S&P 500", zscore=0.8, return_pct=0.4, threshold=2.0) is None


def test_a_missing_zscore_is_not_flagged():
    assert flag_return_zscore("^GSPC", "S&P 500", None, 1.0, 2.0) is None


def test_an_extreme_move_is_escalated_to_high_severity():
    modest = flag_return_zscore("X", "X", 2.1, 1.0, 2.0)
    extreme = flag_return_zscore("X", "X", 3.5, 5.0, 2.0)

    assert modest.severity == "notable"
    assert extreme.severity == "high"


def test_changes_always_carry_an_explicit_sign():
    """Colour must never be the only signal that a move was positive."""
    flag = flag_return_zscore("X", "X", 2.5, 3.1, 2.0)

    assert "+3.10%" in flag.text


def test_new_high_is_flagged():
    flag = flag_extreme("^GSPC", "S&P 500", "high", "52-week")

    assert "new 52-week high" in flag.text


def test_no_extreme_means_no_flag():
    assert flag_extreme("^GSPC", "S&P 500", None, "52-week") is None


def test_moving_average_cross_is_flagged():
    flag = flag_ma_cross("XLK", "Technology", 200, "below")

    assert "crossed below its 200-day moving average" in flag.text


def test_backwardation_is_flagged_as_high_severity():
    flag = flag_vix_term_structure(ratio=1.05, threshold=1.0)

    assert flag.severity == "high"
    assert "backwardation" in flag.text


def test_contango_is_not_flagged():
    assert flag_vix_term_structure(ratio=0.87, threshold=1.0) is None


def test_an_unavailable_vix_ratio_is_not_flagged():
    """When ^VIX3M is missing the rule must stay silent, not guess."""
    assert flag_vix_term_structure(None, threshold=1.0) is None


def test_curve_inversion_is_flagged():
    flag = flag_curve_transition("2s10s", "inverted", -12.0)

    assert "has inverted" in flag.text
    assert "-12bp" in flag.text


def test_a_stable_curve_is_not_flagged():
    assert flag_curve_transition("2s10s", None, 39.0) is None


def test_a_large_credit_move_is_flagged_with_direction():
    widened = flag_credit_move("HY", "High yield OAS", 18.0, threshold_bp=10.0)
    tightened = flag_credit_move("HY", "High yield OAS", -18.0, threshold_bp=10.0)

    assert "widened 18bp" in widened.text
    assert "tightened 18bp" in tightened.text


def test_a_small_credit_move_is_not_flagged():
    assert flag_credit_move("HY", "High yield OAS", 4.0, threshold_bp=10.0) is None


# -- ranking -----------------------------------------------------------------


def test_flags_are_ranked_by_severity_then_magnitude():
    flags = [
        Flag("move", "notable small", "notable", value=2.1),
        Flag("credit", "high big", "high", value=40.0),
        Flag("move", "notable big", "notable", value=3.0),
        Flag("trend", "info", "info", value=1.0),
    ]

    ranked = rank_flags(flags, max_flags=4)

    assert [f.text for f in ranked] == ["high big", "notable big", "notable small", "info"]


def test_the_panel_is_trimmed_to_its_maximum():
    flags = [Flag("move", f"flag {i}", "notable", value=float(i)) for i in range(20)]

    assert len(rank_flags(flags, max_flags=12)) == 12


def test_a_quiet_day_produces_no_flags():
    assert rank_flags([], max_flags=12) == []
