"""Tests for the risk regime composite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tearsheet.analytics.regime import (
    band_for,
    compute_regime,
    normalised_weights,
    rolling_zscore,
)


@dataclass(frozen=True)
class Spec:
    """Minimal stand-in for a configured regime component."""

    weight: float
    sign: int
    label: str


@dataclass(frozen=True)
class Band:
    """Minimal stand-in for a configured band."""

    max: float
    label: str
    tone: str


BANDS = [
    Band(-1.0, "Risk-on", "positive"),
    Band(-0.3, "Mildly risk-on", "positive"),
    Band(0.3, "Neutral", "neutral"),
    Band(1.0, "Mildly risk-off", "negative"),
    Band(99.0, "Risk-off", "negative"),
]


def noisy_series(periods: int = 300, seed: int = 1, final: float | None = None) -> pd.Series:
    """A deterministic series with enough dispersion to z-score."""
    rng = np.random.default_rng(seed)
    values = 20.0 + rng.normal(0, 2.0, periods)
    if final is not None:
        values[-1] = final
    return pd.Series(values, index=pd.bdate_range("2025-01-01", periods=periods))


# -- rolling z-score ---------------------------------------------------------


def test_rolling_zscore_does_not_look_ahead():
    """Each historical point must be scored only against data available then."""
    series = noisy_series(200)
    before = rolling_zscore(series, lookback=100)

    extended = pd.concat(
        [series, pd.Series([999.0], index=[series.index[-1] + pd.Timedelta(days=1)])]
    )
    after = rolling_zscore(extended, lookback=100)

    # check_freq=False: concat drops the index freq attribute, values are the point.
    pd.testing.assert_series_equal(before, after.iloc[: len(before)], check_freq=False)


def test_rolling_zscore_of_an_extreme_reading_is_large():
    series = noisy_series(300, final=40.0)  # ~10 standard deviations above

    assert float(rolling_zscore(series, lookback=252).iloc[-1]) > 5.0


def test_rolling_zscore_of_a_flat_series_is_empty_not_infinite():
    flat = pd.Series([5.0] * 300, index=pd.bdate_range("2025-01-01", periods=300))

    assert rolling_zscore(flat, lookback=252).empty


# -- weights -----------------------------------------------------------------


def test_weights_normalise_to_one():
    components = {"a": Spec(0.2, 1, "A"), "b": Spec(0.3, 1, "B")}

    weights = normalised_weights(components, {"a", "b"})

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["b"] == pytest.approx(0.6)


def test_a_missing_component_is_dropped_and_the_rest_reweighted():
    """Scoring a missing input as zero would drag the composite to neutral."""
    components = {"a": Spec(0.5, 1, "A"), "b": Spec(0.3, 1, "B"), "c": Spec(0.2, 1, "C")}

    weights = normalised_weights(components, {"a", "b"})

    assert set(weights) == {"a", "b"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.625)


def test_no_available_components_raises():
    with pytest.raises(ValueError, match="no usable data|No regime components"):
        normalised_weights({"a": Spec(1.0, 1, "A")}, set())


# -- bands -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-2.0, "Risk-on"),
        (-0.5, "Mildly risk-on"),
        (0.0, "Neutral"),
        (0.5, "Mildly risk-off"),
        (3.0, "Risk-off"),
    ],
)
def test_band_lookup(score, expected):
    assert band_for(score, BANDS)[0] == expected


# -- composite ---------------------------------------------------------------


def test_score_is_the_sum_of_contributions():
    inputs = {"vix": noisy_series(300, seed=1), "hy": noisy_series(300, seed=2)}
    components = {"vix": Spec(0.6, 1, "VIX"), "hy": Spec(0.4, 1, "HY")}

    result = compute_regime(inputs, components, BANDS, lookback=252)

    assert result.score == pytest.approx(sum(c.contribution for c in result.components))


def test_a_high_reading_with_a_positive_sign_pushes_the_score_risk_off():
    inputs = {"vix": noisy_series(300, final=40.0)}
    components = {"vix": Spec(1.0, 1, "VIX")}

    result = compute_regime(inputs, components, BANDS, lookback=252)

    assert result.score > 1.0
    assert result.label == "Risk-off"


def test_the_sign_inverts_the_contribution():
    """SPX above its 200dma is risk-ON, so its sign is negative."""
    inputs = {"spx": noisy_series(300, final=40.0)}

    positive = compute_regime(inputs, {"spx": Spec(1.0, 1, "SPX")}, BANDS, lookback=252)
    negative = compute_regime(inputs, {"spx": Spec(1.0, -1, "SPX")}, BANDS, lookback=252)

    assert positive.score == pytest.approx(-negative.score)


def test_a_missing_component_is_reported_not_hidden():
    inputs = {"vix": noisy_series(300)}
    components = {"vix": Spec(0.5, 1, "VIX"), "vix_term": Spec(0.5, 1, "VIX3M")}

    result = compute_regime(inputs, components, BANDS, lookback=252)

    assert result.missing == ["vix_term"]
    assert not result.is_complete
    assert result.components[0].weight == pytest.approx(1.0)


def test_no_usable_inputs_returns_none():
    components = {"vix": Spec(1.0, 1, "VIX")}

    assert compute_regime({}, components, BANDS, lookback=252) is None


def test_components_are_ordered_by_absolute_contribution():
    inputs = {
        "big": noisy_series(300, seed=3, final=45.0),
        "small": noisy_series(300, seed=4),
    }
    components = {"big": Spec(0.5, 1, "Big"), "small": Spec(0.5, 1, "Small")}

    result = compute_regime(inputs, components, BANDS, lookback=252)

    assert result.components[0].key == "big"


def test_history_is_produced_for_the_chart():
    inputs = {"vix": noisy_series(300, seed=5), "hy": noisy_series(300, seed=6)}
    components = {"vix": Spec(0.5, 1, "VIX"), "hy": Spec(0.5, 1, "HY")}

    result = compute_regime(inputs, components, BANDS, lookback=252)

    assert not result.history.empty
    assert float(result.history.iloc[-1]) == pytest.approx(result.score)


def test_mixed_frequency_components_are_forward_filled():
    """NFCI is weekly and the VIX daily; the chart must not collapse to weekly."""
    daily = noisy_series(300, seed=11)
    weekly = daily.resample("W-FRI").last()
    components = {"vix": Spec(0.5, 1, "VIX"), "nfci": Spec(0.5, 1, "NFCI")}

    result = compute_regime({"vix": daily, "nfci": weekly}, components, BANDS, lookback=252)

    assert len(result.history) > 40, "history must stay daily, not drop to weekly"
    assert float(result.history.iloc[-1]) == pytest.approx(result.score)
