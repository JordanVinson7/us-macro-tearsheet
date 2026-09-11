"""Tests for configuration loading and secret handling."""

from __future__ import annotations

import pytest
import yaml

from tearsheet.config import Config, ConfigError, MissingSecretError, Secrets
from tearsheet.pipeline import SECTION_TITLES
from tests.conftest import PROJECT_ROOT


def test_real_config_validates(config):
    assert config.meta.timezone == "America/New_York"
    assert config.enabled_sections


def test_every_enabled_section_has_a_builder_slot(config):
    """config.yaml must not name a section the pipeline does not know about."""
    assert set(config.sections) <= set(SECTION_TITLES)


def test_unknown_config_key_is_rejected(tmp_path):
    """extra='forbid' turns a config typo into a startup error, not silence."""
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    raw["meta"]["tiimezone"] = "America/New_York"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="tiimezone"):
        Config.load(path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "nope.yaml")


def test_all_yf_tickers_are_deduplicated_and_include_ratio_legs(config):
    tickers = config.universe.all_yf_tickers()
    assert len(tickers) == len(set(tickers))
    for ratio in config.universe.ratios:
        assert ratio.numerator in tickers
        assert ratio.denominator in tickers
    assert "SPY" in tickers  # support ticker, has no display row of its own


def test_fred_ids_include_fallback_levels(config):
    ids = config.fred.all_series_ids()
    assert len(ids) == len(set(ids))
    assert set(config.fred.fallback_levels.values()) <= set(ids)


def test_regime_weights_sum_to_one(config):
    weights = config.regime.normalised_weights()
    assert pytest.approx(sum(weights.values())) == 1.0


def test_regime_weights_renormalise_when_components_are_missing(config):
    """A missing input must not be scored as zero — the rest re-weight."""
    available = set(config.regime.components) - {"vix_term"}
    weights = config.regime.normalised_weights(available)
    assert "vix_term" not in weights
    assert pytest.approx(sum(weights.values())) == 1.0


def test_regime_with_no_components_raises(config):
    with pytest.raises(ConfigError):
        config.regime.normalised_weights(set())


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-2.0, "Risk-on"), (-0.5, "Mildly risk-on"), (0.0, "Neutral"), (5.0, "Risk-off")],
)
def test_regime_bands(config, value, expected):
    assert config.regime.band_for(value).label == expected


def test_placeholder_credentials_are_treated_as_missing(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "your_32_char_fred_key_here")
    monkeypatch.setenv("POLYGON_API_KEY", "abc123def456ghi789")
    secrets = Secrets.from_env()

    assert not secrets.has("fred_api_key")
    assert secrets.has("polygon_api_key")
    assert secrets.status()["FRED_API_KEY"] == "placeholder"


def test_status_never_exposes_a_credential_value(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "topsecretvalue123")
    secrets = Secrets.from_env()

    assert "topsecretvalue123" not in str(secrets.status())
    assert "topsecretvalue123" not in repr(secrets)
    assert secrets.reveal("fred_api_key") == "topsecretvalue123"


def test_require_lists_every_missing_credential_at_once(monkeypatch):
    for name in Secrets.ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    secrets = Secrets.from_env()

    with pytest.raises(MissingSecretError) as excinfo:
        secrets.require("fred_api_key", "polygon_api_key")

    assert "FRED_API_KEY" in str(excinfo.value)
    assert "POLYGON_API_KEY" in str(excinfo.value)
