"""Tests for the data layer: rate limiting, quality checks, sessions, fixtures.

Everything here runs on synthetic data with no network access, so the suite is
fast and deterministic and can run in CI without credentials.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tearsheet.data.base import RateLimiter
from tearsheet.data.fixtures import FixtureStore, safe_name
from tearsheet.data.news import deduplicate
from tearsheet.data.quality import (
    check_price_frame,
    compute_staleness,
    max_plausible_move,
    move_category,
)
from tearsheet.data.sessions import is_session, last_completed_session, session_n_ago
from tearsheet.models import NewsItem, SourceRef

ET = ZoneInfo("America/New_York")


def make_frame(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV frame from a list of closes."""
    index = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000.0] * len(closes),
        },
        index=index,
    )


# -- rate limiter ------------------------------------------------------------


class FakeClock:
    """A controllable stand-in for ``time.monotonic``/``time.sleep``."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("tearsheet.data.base.time.monotonic", fake.monotonic)
    monkeypatch.setattr("tearsheet.data.base.time.sleep", fake.sleep)
    return fake


def test_rate_limiter_allows_burst_up_to_the_limit(clock):
    limiter = RateLimiter(max_calls=5, per_seconds=60)

    waits = [limiter.acquire() for _ in range(5)]

    assert waits == [0.0] * 5
    assert clock.slept == []


def test_rate_limiter_blocks_the_sixth_call(clock):
    """Polygon's free tier is 5 per rolling 60s; the 6th must wait."""
    limiter = RateLimiter(max_calls=5, per_seconds=60)
    for _ in range(5):
        limiter.acquire()

    clock.now += 10  # 10s into the window
    waited = limiter.acquire()

    assert waited == pytest.approx(50.0)


def test_rate_limiter_window_is_rolling_not_fixed(clock):
    """Calls leaving the window free up slots one at a time."""
    limiter = RateLimiter(max_calls=2, per_seconds=60)
    limiter.acquire()
    clock.now += 30
    limiter.acquire()

    clock.now += 31  # first call has now aged out, second has not
    assert limiter.acquire() == 0.0


# -- quality checks ----------------------------------------------------------


def test_non_positive_prices_are_dropped(config):
    frame = make_frame([100.0, -5.0, 102.0])

    result = check_price_frame("TEST", frame, config.quality)

    assert len(result.frame) == 2
    assert any("non-positive" in p for p in result.problems)


def test_nan_close_rows_are_dropped(config):
    frame = make_frame([100.0, 101.0, 102.0])
    frame.iloc[-1, frame.columns.get_loc("close")] = float("nan")

    result = check_price_frame("TEST", frame, config.quality)

    # The NaN row is dropped, leaving a valid latest close.
    assert not result.frame.empty
    assert result.frame["close"].iloc[-1] == 101.0


def test_implausible_move_is_flagged_but_the_level_is_kept(config):
    """A futures roll must not delete the price, only discredit the return."""
    frame = make_frame([115.0, 114.0, 78.0, 77.0])

    result = check_price_frame("SI=F", frame, config.quality)

    assert len(result.frame) == 4, "levels must be preserved"
    assert len(result.suspect_dates) == 1
    assert result.suspect_dates[0] == date(2026, 1, 5)


def test_volatility_and_crypto_get_wider_move_thresholds(config):
    assert move_category("^VIX") == "volatility"
    assert move_category("BTC-USD") == "crypto"
    assert move_category("XLK") == "default"
    assert max_plausible_move("^VIX", config.quality) > max_plausible_move("XLK", config.quality)


def test_a_thirty_percent_vix_move_is_not_flagged(config):
    """VIX routinely moves 30% in a day; flagging it would be noise."""
    frame = make_frame([15.0, 19.5])

    assert check_price_frame("^VIX", frame, config.quality).suspect_dates == []


# -- staleness ---------------------------------------------------------------


def test_staleness_prefers_the_scheduled_release_date(config):
    """Monthly CPI dated 2026-08-01 is current if the next release is today."""
    result = compute_staleness(
        observation_date=date(2026, 8, 1),
        frequency="monthly",
        run_date=date(2026, 9, 11),
        config=config.quality,
        next_release=date(2026, 9, 11),
    )

    assert result.age_days == 41
    assert not result.is_stale


def test_a_missed_release_is_stale(config):
    result = compute_staleness(
        observation_date=date(2026, 8, 1),
        frequency="monthly",
        run_date=date(2026, 9, 11),
        config=config.quality,
        next_release=date(2026, 9, 4),
    )

    assert result.is_stale


def test_staleness_falls_back_to_the_grace_period(config):
    fresh = compute_staleness(date(2026, 9, 9), "daily", date(2026, 9, 11), config.quality)
    old = compute_staleness(date(2026, 8, 1), "daily", date(2026, 9, 11), config.quality)

    assert not fresh.is_stale
    assert old.is_stale


def test_missing_observation_is_stale(config):
    assert compute_staleness(None, "daily", date(2026, 9, 11), config.quality).is_stale


# -- sessions ----------------------------------------------------------------


def test_pre_open_run_uses_the_previous_session():
    """At 08:00 ET the report is about yesterday, never a partial bar today."""
    assert last_completed_session(datetime(2026, 9, 11, 8, 0, tzinfo=ET)) == date(2026, 9, 10)


def test_after_the_close_todays_session_counts():
    assert last_completed_session(datetime(2026, 9, 11, 17, 0, tzinfo=ET)) == date(2026, 9, 11)


def test_market_holidays_are_skipped():
    """4 July 2026 falls on a Saturday and is observed on Friday the 3rd."""
    assert not is_session(date(2026, 7, 3))
    assert last_completed_session(date(2026, 7, 6)) == date(2026, 7, 2)


def test_session_n_ago_counts_trading_days():
    assert session_n_ago(date(2026, 9, 10), 0) == date(2026, 9, 10)
    assert session_n_ago(date(2026, 9, 10), 1) == date(2026, 9, 9)


# -- news deduplication ------------------------------------------------------


def headline(title: str, minutes_ago: int = 0) -> NewsItem:
    return NewsItem(
        title=title,
        publisher="Test Wire",
        url="https://example.com",
        published_at=datetime(2026, 9, 11, 12, 0, tzinfo=ET) - timedelta(minutes=minutes_ago),
        source=SourceRef("test"),
    )


def test_syndicated_duplicates_collapse():
    items = [
        headline("Fed holds rates steady as expected", 0),
        headline("Fed Holds Rates Steady, As Expected", 5),
    ]

    assert len(deduplicate(items)) == 1


def test_distinct_headlines_are_kept():
    items = [
        headline("Fed holds rates steady", 0),
        headline("Oil prices surge on supply disruption", 5),
    ]

    assert len(deduplicate(items)) == 2


def test_deduplicate_returns_newest_first():
    items = [headline("Older story about gold", 60), headline("Newer story about oil", 1)]

    assert deduplicate(items)[0].title.startswith("Newer")


# -- fixtures ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [("^GSPC", "GSPC"), ("GC=F", "GC_F"), ("BTC-USD", "BTC-USD"), ("DX-Y.NYB", "DX-Y.NYB")],
)
def test_safe_name_sanitises_symbols(key, expected):
    assert safe_name(key) == expected


def test_fixture_frames_round_trip(tmp_path):
    """Ticker symbols must survive the filename sanitisation intact."""
    store = FixtureStore(tmp_path)
    frames = {"^GSPC": make_frame([1.0, 2.0]), "GC=F": make_frame([3.0, 4.0])}

    store.save_frames("prices", frames)
    loaded = store.load_frames("prices")

    assert set(loaded) == {"^GSPC", "GC=F"}
    pd.testing.assert_frame_equal(loaded["^GSPC"], frames["^GSPC"], check_freq=False)


def test_fixture_series_round_trip(tmp_path):
    store = FixtureStore(tmp_path)
    series = {"DGS10": pd.Series([4.1, 4.2], index=pd.to_datetime(["2026-09-09", "2026-09-10"]))}

    store.save_series("fred", series)
    loaded = store.load_series("fred")

    assert loaded["DGS10"].tolist() == [4.1, 4.2]


def test_missing_fixture_group_returns_empty(tmp_path):
    assert FixtureStore(tmp_path).load_frames("nothing-here") == {}
