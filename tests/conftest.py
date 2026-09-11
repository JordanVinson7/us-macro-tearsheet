"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tearsheet.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config() -> Config:
    """The project's real config.yaml, validated."""
    return Config.load(PROJECT_ROOT / "config.yaml")
