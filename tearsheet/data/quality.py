"""Sanity and staleness checks applied to everything the data layer returns.

Bad data is worse than missing data: a mis-scaled price silently poisons
returns, volatility, z-scores and the regime composite all at once. These
checks run once, at the boundary, so the analytics layer can assume its inputs
are plausible.

Every function here is pure and takes its thresholds from config, which makes
them straightforward to unit-test on synthetic series.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import pandas as pd

from tearsheet.config import Config, Frequency, QualityConfig
from tearsheet.models import PriceData, Staleness
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.quality")

#: Instrument categories with their own implausible-move threshold.
CRYPTO_SUFFIX = "-USD"
VOLATILITY_TICKERS = frozenset({"^VIX", "^VIX3M", "^VVIX"})


def move_category(ticker: str) -> str:
    """Classify a ticker for the purposes of the implausible-move check.

    Crypto trades 24/7 and moves further than equities; volatility indices
    routinely move 20%+ in a day. Applying one blanket threshold would either
    let bad equity prints through or reject perfectly real VIX spikes.
    """
    if ticker in VOLATILITY_TICKERS:
        return "volatility"
    if ticker.endswith(CRYPTO_SUFFIX):
        return "crypto"
    return "default"


def max_plausible_move(ticker: str, config: QualityConfig) -> float:
    """The largest daily move, in percent, accepted for this ticker."""
    thresholds = config.max_plausible_daily_move_pct
    return thresholds.get(move_category(ticker), thresholds["default"])


class PriceCheck(NamedTuple):
    """Outcome of validating one price frame.

    Attributes:
        frame: The cleaned history. Empty means the ticker is unusable.
        problems: Human-readable issues, surfaced as run warnings.
        suspect_dates: Dates whose one-day return is implausible.
    """

    frame: pd.DataFrame
    problems: list[str]
    suspect_dates: list[date]


def check_price_frame(ticker: str, frame: pd.DataFrame, config: QualityConfig) -> PriceCheck:
    """Validate and clean one OHLCV frame.

    Args:
        ticker: The symbol, used for messages and threshold selection.
        frame: OHLCV history indexed by date.
        config: Quality thresholds.

    Returns:
        The check outcome. An empty frame means the ticker failed validation
        outright and should be treated as unavailable.
    """
    problems: list[str] = []

    if frame.empty or "close" not in frame.columns:
        return PriceCheck(frame.iloc[0:0], [f"{ticker}: no close prices"], [])

    cleaned = frame.copy()

    if config.reject_non_positive_prices:
        bad = cleaned["close"] <= 0
        if bad.any():
            problems.append(f"{ticker}: dropped {int(bad.sum())} non-positive prices")
            cleaned = cleaned[~bad]

    cleaned = cleaned.dropna(subset=["close"])
    if cleaned.empty:
        problems.append(f"{ticker}: every observation was missing or invalid")
        return PriceCheck(cleaned, problems, [])

    if not pd.notna(cleaned["close"].iloc[-1]):
        problems.append(f"{ticker}: latest close is NaN")
        return PriceCheck(cleaned.iloc[0:0], problems, [])

    # Implausible single-day moves are a bad tick, an unapplied split, or — on
    # continuous futures — a contract roll, where the series steps to a new
    # expiry and stays there. The price level is still correct either way, so
    # the observation is kept and only the RETURN across that date is marked
    # suspect. Deleting the row would corrupt the level; keeping the return
    # would corrupt every volatility and z-score computed from it.
    threshold = max_plausible_move(ticker, config)
    moves = cleaned["close"].pct_change().abs() * 100.0
    extreme = moves[moves > threshold]
    suspect_dates = [ts.date() for ts in extreme.index]
    if not extreme.empty:
        worst_date = extreme.idxmax()
        problems.append(
            f"{ticker}: {len(extreme)} implausible move(s), largest {extreme.max():.1f}% on "
            f"{worst_date:%Y-%m-%d} (threshold {threshold:.0f}%); "
            "levels kept, those returns marked suspect"
        )

    return PriceCheck(cleaned, problems, suspect_dates)


def price_age_days(frame: pd.DataFrame, as_of: date) -> int:
    """Calendar days between a frame's last bar and the reference session."""
    return (as_of - frame.index[-1].date()).days


def apply_price_checks(data: PriceData, config: Config, report: RunReport) -> PriceData:
    """Run :func:`check_price_frame` across a whole :class:`PriceData`.

    Tickers that fail outright — including those whose most recent bar is too
    old to be presented as a current level — are moved into ``missing`` so the
    fallback chain or the affected section can react to them.
    """
    checked = PriceData(as_of=data.as_of, missing=list(data.missing))
    max_age = config.quality.max_price_age_days

    for ticker, frame in data.frames.items():
        cleaned, problems, suspect = check_price_frame(ticker, frame, config.quality)
        for problem in problems:
            report.warn(f"Data quality — {problem}")
        if cleaned.empty:
            checked.missing.append(ticker)
            continue

        if data.as_of is not None:
            age = price_age_days(cleaned, data.as_of)
            if age > max_age:
                report.warn(
                    f"Data quality — {ticker}: most recent bar is "
                    f"{cleaned.index[-1]:%Y-%m-%d}, {age} days before the session "
                    f"({data.as_of}); dropped rather than shown as current."
                )
                checked.missing.append(ticker)
                continue

        checked.frames[ticker] = cleaned
        checked.sources[ticker] = data.sources[ticker]
        if suspect:
            checked.suspect_returns[ticker] = suspect

    return checked


def compute_staleness(
    observation_date: date | None,
    frequency: Frequency,
    run_date: date,
    config: QualityConfig,
    next_release: date | None = None,
) -> Staleness:
    """Decide whether a series is later than it should be.

    Where the next scheduled release date is known this is an exact test: the
    series is stale only once that release has come and gone without new data.
    Age alone is a poor proxy for low-frequency series, because FRED dates an
    observation at the start of its period — August CPI is dated 2026-08-01 but
    is not published until mid-September, so a perfectly current series looks
    six weeks old.

    Args:
        observation_date: Date of the latest observation.
        frequency: The configured update cadence.
        run_date: The report date.
        config: Quality thresholds, providing the fallback grace period.
        next_release: The next scheduled release date, when known.

    Returns:
        The staleness verdict, which drives the visible "stale" badge.
    """
    if observation_date is None:
        return Staleness(None, frequency, None, is_stale=True)

    age_days = (run_date - observation_date).days
    if next_release is not None:
        is_stale = next_release < run_date
    else:
        is_stale = age_days > config.grace_days(frequency)

    return Staleness(
        observation_date=observation_date,
        expected_frequency=frequency,
        age_days=age_days,
        is_stale=is_stale,
    )
