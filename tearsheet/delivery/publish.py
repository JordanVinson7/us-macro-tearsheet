"""Writing the rendered sheet to disk, and maintaining the archive.

Output layout, served by GitHub Pages from ``docs/`` on the main branch::

    docs/index.html                 the latest sheet
    docs/archive/YYYY-MM-DD.html    one file per trading day
    docs/archive/index.html         a listing of every archived sheet

Local runs write the same structure into ``output/`` instead, so a run on a
laptop can never collide with the tree that CI commits.

The archive is pruned to ``output.retain_days``. Each sheet is 0.5–1.5MB of
embedded chart data, so an unpruned archive would add a few hundred megabytes
to the repository each year — and git never forgets. Pruned dates stay listed
in the index, marked as expired, so the record of what was published survives
even when the file does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tearsheet.config import Settings
from tearsheet.models import RunOptions
from tearsheet.observability import RunReport, get_logger
from tearsheet.render.html import build_environment, read_stylesheets

logger = get_logger("delivery.publish")

ARCHIVE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.html$")


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One dated sheet in the archive index.

    Attributes:
        date: The report date.
        filename: File name within the archive directory.
        exists: False once the file has been pruned.
        size_kb: File size in kilobytes, zero when pruned.
    """

    date: date
    filename: str
    exists: bool
    size_kb: int


def archive_entries(archive_dir: Path) -> list[ArchiveEntry]:
    """List archived sheets, newest first."""
    if not archive_dir.exists():
        return []

    entries = []
    for path in archive_dir.glob("*.html"):
        match = ARCHIVE_PATTERN.match(path.name)
        if not match:
            continue
        entries.append(
            ArchiveEntry(
                date=date.fromisoformat(match.group(1)),
                filename=path.name,
                exists=True,
                size_kb=round(path.stat().st_size / 1024),
            )
        )
    return sorted(entries, key=lambda e: e.date, reverse=True)


def prune_archive(archive_dir: Path, retain_days: int, today: date) -> list[str]:
    """Delete archived sheets older than the retention window.

    Args:
        archive_dir: The archive directory.
        retain_days: Days to keep. ``0`` disables pruning entirely.
        today: The report date, used as the reference point.

    Returns:
        Names of the files removed.
    """
    if retain_days <= 0:
        return []

    cutoff = today - timedelta(days=retain_days)
    removed = []
    for entry in archive_entries(archive_dir):
        if entry.date < cutoff:
            (archive_dir / entry.filename).unlink()
            removed.append(entry.filename)

    if removed:
        logger.info("Pruned %d archived sheet(s) older than %s", len(removed), cutoff)
    return removed


def render_archive_index(
    entries: list[ArchiveEntry], settings: Settings, pruned_after_days: int
) -> str:
    """Render the archive listing page."""
    environment = build_environment()
    return environment.get_template("archive.html.j2").render(
        entries=entries,
        meta=settings.config.meta,
        stylesheet=read_stylesheets(),
        generated_at=datetime.now(ZoneInfo(settings.config.meta.timezone)),
        retain_days=pruned_after_days,
    )


def write_outputs(
    html: str, settings: Settings, options: RunOptions, report: RunReport
) -> dict[str, Path]:
    """Write the sheet, refresh the archive, and rebuild the archive index.

    Args:
        html: The rendered page.
        settings: Configuration.
        options: Run options, including the output directory.
        report: Run report, for recording the produced artifacts.

    Returns:
        Paths of the files written, keyed by kind.
    """
    root = options.output_dir
    archive_dir = root / settings.config.output.archive_dir
    archive_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    latest = root / "index.html"
    latest.write_text(html, encoding="utf-8")
    written["html"] = latest

    dated = archive_dir / f"{options.run_date.isoformat()}.html"
    dated.write_text(html, encoding="utf-8")
    written["archive"] = dated

    prune_archive(archive_dir, settings.config.output.retain_days, options.run_date)

    index = archive_dir / "index.html"
    index.write_text(
        render_archive_index(
            archive_entries(archive_dir), settings, settings.config.output.retain_days
        ),
        encoding="utf-8",
    )
    written["archive_index"] = index

    for kind, path in written.items():
        report.add_artifact(kind, path)

    size_kb = latest.stat().st_size / 1024
    logger.info("Wrote %s (%.0f KB) and archived as %s", latest, size_kb, dated.name)
    return written
