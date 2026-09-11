"""Tests for the schedule guard's decision table.

The guard is what makes two daily cron firings safe, so every branch is
exercised here with an injected clock rather than by waiting for a scheduler.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tearsheet.config import load_settings
from tearsheet.guard import evaluate, report_path
from tearsheet.models import RunMode, RunOptions

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def options(tmp_path: Path, *, force: bool = False) -> RunOptions:
    return RunOptions(
        mode=RunMode.LIVE,
        send_email=False,
        build_pdf=False,
        force=force,
        open_result=False,
        publish=True,
        record=False,
        verbose=False,
        config_path=PROJECT_ROOT / "config.yaml",
        output_dir=tmp_path,
        run_date=date(2026, 9, 11),
    )


def at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    """A moment on Friday 11 September 2026 (a normal trading day), in ET."""
    return datetime(2026, 9, day, hour, minute, tzinfo=ET)


# -- the happy path ----------------------------------------------------------


def test_runs_inside_the_window_on_a_trading_day(settings, tmp_path):
    decision = evaluate(settings, options(tmp_path), at(8, 0))

    assert decision.should_run
    assert "inside the publish window" in decision.reason


@pytest.mark.parametrize("hour", [7, 8, 9, 10, 11])
def test_the_window_is_wide_enough_for_a_late_scheduled_start(settings, tmp_path, hour):
    """GitHub cron can fire tens of minutes late; the window must absorb that."""
    assert evaluate(settings, options(tmp_path), at(hour, 0)).should_run


# -- both cron firings -------------------------------------------------------


@pytest.mark.parametrize(("utc_hour", "label"), [(12, "first firing"), (13, "second firing")])
def test_both_utc_cron_firings_land_inside_the_window(settings, tmp_path, utc_hour, label):
    """12:00 and 13:00 UTC must both qualify in EDT and in EST."""
    moment = datetime(2026, 9, 11, utc_hour, 0, tzinfo=UTC)

    assert evaluate(settings, options(tmp_path), moment).should_run, label


def test_the_second_firing_stands_down_once_a_sheet_exists(settings, tmp_path):
    """This idempotency check — not the clock — is what prevents two sheets."""
    opts = options(tmp_path)
    published = report_path(settings, opts, date(2026, 9, 11))
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text("<html></html>")

    decision = evaluate(settings, opts, at(9, 0))

    assert not decision.should_run
    assert "already exists" in decision.reason


# -- calendar ----------------------------------------------------------------


def test_weekends_are_skipped(settings, tmp_path):
    decision = evaluate(settings, options(tmp_path), at(8, 0, day=12))  # Saturday

    assert not decision.should_run
    assert "weekend" in decision.reason


def test_market_holidays_are_skipped(settings, tmp_path):
    """3 July 2026 is the observed Independence Day holiday."""
    moment = datetime(2026, 7, 3, 8, 0, tzinfo=ET)

    decision = evaluate(settings, options(tmp_path), moment)

    assert not decision.should_run
    assert "trading session" in decision.reason


# -- window ------------------------------------------------------------------


@pytest.mark.parametrize(("hour", "minute"), [(6, 59), (11, 1), (16, 0), (3, 0)])
def test_outside_the_window_is_skipped(settings, tmp_path, hour, minute):
    decision = evaluate(settings, options(tmp_path), at(hour, minute))

    assert not decision.should_run
    assert "outside the publish window" in decision.reason


# -- force -------------------------------------------------------------------


def test_force_bypasses_the_window(settings, tmp_path):
    assert evaluate(settings, options(tmp_path, force=True), at(3, 0)).should_run


def test_force_bypasses_a_weekend(settings, tmp_path):
    assert evaluate(settings, options(tmp_path, force=True), at(8, 0, day=12)).should_run


def test_force_overwrites_an_existing_sheet(settings, tmp_path):
    opts = options(tmp_path, force=True)
    published = report_path(settings, opts, date(2026, 9, 11))
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text("<html></html>")

    decision = evaluate(settings, opts, at(9, 0))

    assert decision.should_run
    assert "--force" in decision.reason


def test_the_decision_is_reported_in_eastern_time(settings, tmp_path):
    """Every timestamp in this project is ET, including the guard's."""
    decision = evaluate(settings, options(tmp_path), datetime(2026, 9, 11, 13, 0, tzinfo=UTC))

    assert decision.now_et.hour == 9
    assert str(decision.now_et.tzinfo) == "America/New_York"
