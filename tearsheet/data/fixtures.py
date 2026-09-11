"""Recording and replaying data for ``--mock`` mode.

Mock mode exists so the page, charts and templates can be developed offline
with zero API calls — which matters when Polygon allows 5 requests a minute and
a template tweak should not cost a rate-limit window.

Fixtures are stored as CSV and JSON rather than pickle. They are a little more
code to write, but they are diffable in git, inspectable in an editor, and
carry no arbitrary-code-execution risk when loaded — all of which matter for a
file set that lives in a public repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tearsheet.observability import get_logger

logger = get_logger("data.fixtures")

#: Bumped when the on-disk layout changes so a stale fixture set fails loudly.
FIXTURE_VERSION = 1

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(key: str) -> str:
    """Turn a ticker or series ID into a filesystem-safe stem.

    ``^GSPC`` and ``GC=F`` cannot be used as filenames as-is, so they are
    sanitised here and the original key is preserved in ``index.json``.
    """
    return _UNSAFE.sub("_", key).strip("_") or "unnamed"


def _json_default(value: Any) -> Any:
    """Serialise dates, times and pandas scalars that json cannot handle."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class FixtureStore:
    """Reads and writes the fixture tree under ``fixtures/``.

    Attributes:
        root: Directory holding the fixture set.
    """

    root: Path

    # -- frames --------------------------------------------------------------

    def save_frames(self, group: str, frames: dict[str, pd.DataFrame]) -> None:
        """Write a group of DataFrames as CSVs plus a key index."""
        directory = self.root / group
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.csv"):
            stale.unlink()

        index: dict[str, str] = {}
        for key, frame in frames.items():
            stem = safe_name(key)
            index[stem] = key
            frame.to_csv(directory / f"{stem}.csv", index_label="timestamp")

        (directory / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
        logger.debug("Saved %d frames to %s", len(frames), directory)

    def load_frames(self, group: str) -> dict[str, pd.DataFrame]:
        """Read a group of DataFrames back, restoring the original keys."""
        directory = self.root / group
        index_path = directory / "index.json"
        if not index_path.exists():
            return {}

        index: dict[str, str] = json.loads(index_path.read_text())
        frames: dict[str, pd.DataFrame] = {}
        for stem, key in index.items():
            path = directory / f"{stem}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
            frame.index.name = None
            frames[key] = frame
        return frames

    def save_series(self, group: str, series: dict[str, pd.Series]) -> None:
        """Write a group of Series as single-column CSVs."""
        self.save_frames(group, {k: v.to_frame("value") for k, v in series.items()})

    def load_series(self, group: str) -> dict[str, pd.Series]:
        """Read a group of Series back."""
        return {
            key: frame["value"].rename(key)
            for key, frame in self.load_frames(group).items()
            if "value" in frame.columns
        }

    # -- documents -----------------------------------------------------------

    def save_json(self, name: str, payload: Any) -> None:
        """Write one JSON document."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default, sort_keys=True))

    def load_json(self, name: str, fallback: Any = None) -> Any:
        """Read one JSON document, or return ``fallback`` when absent."""
        path = self.root / f"{name}.json"
        if not path.exists():
            return fallback
        return json.loads(path.read_text())

    # -- status --------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """Whether a usable fixture set is present."""
        return (self.root / "manifest.json").exists()

    def manifest(self) -> dict[str, Any]:
        """The manifest describing when and how the fixtures were recorded."""
        return self.load_json("manifest", {}) or {}

    def write_manifest(self, run_date: date, session: date, counts: dict[str, int]) -> None:
        """Record what was captured, so a stale fixture set is obvious."""
        self.save_json(
            "manifest",
            {
                "version": FIXTURE_VERSION,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "run_date": run_date,
                "session": session,
                "counts": counts,
            },
        )
