"""The risk regime composite.

**This is a descriptive summary of current conditions, not a trading signal.**
It answers "how do today's conditions compare with the last two years?" and
nothing more. It is labelled as such everywhere it appears.

Construction: each component is z-scored against its own trailing history,
multiplied by a sign (+1 where a higher reading means more risk-off, −1 where
it means less), and combined with configured weights.

Two design points are worth stating explicitly:

* **Missing components are dropped and the remaining weights re-normalised.**
  The alternative — treating a missing input as zero — would silently pull the
  composite toward neutral and misrepresent it as a full reading. When
  ``^VIX3M`` is unavailable, the composite is honestly a six-component index,
  and says so.
* **Components are z-scored, not raw.** The VIX trades around 15–30 and the HY
  spread around 3–8; combining them raw would make the composite a VIX proxy
  with decoration.

The module depends on structural protocols rather than the config classes, so
it can be tested with plain objects and has no import-time coupling to
config.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class ComponentSpec(Protocol):
    """The weighting attributes a regime component must expose."""

    weight: float
    sign: int
    label: str


class BandSpec(Protocol):
    """A labelled upper bound on the composite score."""

    max: float
    label: str
    tone: str


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """One component's contribution to the composite.

    Attributes:
        key: Component key from config.
        label: Display label.
        weight: Re-normalised weight actually applied.
        sign: +1 if a higher reading means more risk-off.
        raw_value: The latest untransformed reading.
        zscore: The z-score of that reading against its trailing history.
        contribution: ``zscore * sign * weight`` — its share of the composite.
    """

    key: str
    label: str
    weight: float
    sign: int
    raw_value: float
    zscore: float
    contribution: float


@dataclass(frozen=True, slots=True)
class RegimeResult:
    """The composite and everything needed to explain it.

    Attributes:
        score: The weighted composite. Positive means risk-off.
        label: The band the score falls into, e.g. "Mildly risk-off".
        tone: Display tone for that band.
        components: Per-component contributions, largest absolute first.
        missing: Component keys that had no usable data.
        history: The composite through time, for the history chart.
    """

    score: float
    label: str
    tone: str
    components: list[ComponentResult]
    missing: list[str]
    history: pd.Series

    @property
    def is_complete(self) -> bool:
        """Whether every configured component contributed."""
        return not self.missing


def rolling_zscore(series: pd.Series, lookback: int, min_periods: int | None = None) -> pd.Series:
    """Z-score of each observation against its own trailing window.

    Using a rolling window rather than the full-sample mean and standard
    deviation keeps the history chart honest: each historical point is scored
    only against data available at that time, not against the future.
    """
    values = series.dropna().sort_index()
    minimum = min_periods or max(20, lookback // 4)
    mean = values.rolling(lookback, min_periods=minimum).mean()
    deviation = values.rolling(lookback, min_periods=minimum).std(ddof=1)
    scored = (values - mean) / deviation.replace(0.0, pd.NA)
    return scored.dropna()


def normalised_weights(
    components: Mapping[str, ComponentSpec], available: set[str]
) -> dict[str, float]:
    """Re-normalise weights over the components that have data.

    Raises:
        ValueError: If no component is available.
    """
    usable = {k: components[k] for k in available if k in components}
    if not usable:
        raise ValueError("No regime components have usable data.")
    total = sum(spec.weight for spec in usable.values())
    return {k: spec.weight / total for k, spec in usable.items()}


def band_for(score: float, bands: Sequence[BandSpec]) -> tuple[str, str]:
    """Return the ``(label, tone)`` of the band a score falls into."""
    for band in bands:
        if score <= band.max:
            return band.label, band.tone
    return bands[-1].label, bands[-1].tone


def compute_regime(
    inputs: Mapping[str, pd.Series],
    components: Mapping[str, ComponentSpec],
    bands: Sequence[BandSpec],
    lookback: int = 504,
) -> RegimeResult | None:
    """Build the risk regime composite.

    Args:
        inputs: Level series keyed by component key. A key that is absent, or
            whose series is too short to z-score, is treated as missing.
        components: Weight, sign and label per component key.
        bands: Ordered bands mapping a score to a label.
        lookback: Trailing window for the z-scores, in observations.

    Returns:
        The composite result, or None if no component had usable data.
    """
    scored: dict[str, pd.Series] = {}
    raw_latest: dict[str, float] = {}

    for key in components:
        series = inputs.get(key)
        if series is None:
            continue
        z = rolling_zscore(series, lookback)
        if z.empty:
            continue
        scored[key] = z
        raw_latest[key] = float(series.dropna().iloc[-1])

    missing = sorted(set(components) - set(scored))
    if not scored:
        return None

    weights = normalised_weights(components, set(scored))

    results: list[ComponentResult] = []
    score = 0.0
    for key, weight in weights.items():
        spec = components[key]
        latest_z = float(scored[key].iloc[-1])
        contribution = latest_z * spec.sign * weight
        score += contribution
        results.append(
            ComponentResult(
                key=key,
                label=spec.label,
                weight=weight,
                sign=spec.sign,
                raw_value=raw_latest[key],
                zscore=latest_z,
                contribution=contribution,
            )
        )

    results.sort(key=lambda r: abs(r.contribution), reverse=True)
    label, tone = band_for(score, bands)

    # Components arrive at different frequencies: the VIX is daily, the NFCI
    # weekly. An inner join would collapse the history to the slowest series'
    # dates and leave the chart disagreeing with the headline score. Instead
    # the union index is forward-filled — the standard treatment for a weekly
    # indicator — and only the leading warm-up rows are dropped.
    frame = pd.DataFrame(scored).sort_index().ffill().dropna()
    history = pd.Series(dtype="float64")
    if not frame.empty:
        signed = frame.mul(pd.Series({k: components[k].sign * weights[k] for k in frame.columns}))
        history = signed.sum(axis=1)

    return RegimeResult(
        score=score,
        label=label,
        tone=tone,
        components=results,
        missing=missing,
        history=history,
    )
