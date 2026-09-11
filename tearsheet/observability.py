"""Logging, secret redaction and the end-of-run report.

Three things need the same run metadata: the page footer (which sources fed
which section), the console summary, and the failure email. They all read it
off a single :class:`RunReport` accumulated as the pipeline executes.

The :class:`SecretRedactor` is a defensive backstop, not the primary control.
The primary control is simply never passing a key to a logger; the redactor
catches the case where a key ends up inside a third-party exception message or
a URL that gets logged by accident.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from tearsheet.models import RunMode, SectionResult, SectionStatus, SourceRef

_CONSOLE: Console | None = None

DEFAULT_TIMEZONE = "America/New_York"
"""Report timezone. Overridden from ``meta.timezone`` in config.yaml."""

REDACTION_PLACEHOLDER = "***REDACTED***"
MIN_REDACTABLE_LENGTH = 8


def get_console() -> Console:
    """Return the shared Rich console, creating it on first use."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(highlight=False)
    return _CONSOLE


class SecretRedactor(logging.Filter):
    """Scrubs registered secret values out of every log record.

    Secrets are registered once at startup from the loaded environment. Short
    values are ignored: redacting a 3-character string would mangle unrelated
    log text without protecting anything meaningful.
    """

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def register(self, value: str | None) -> None:
        """Add a value to the redaction set, ignoring blanks and short values."""
        if value and len(value) >= MIN_REDACTABLE_LENGTH:
            self._secrets.add(value)

    def redact(self, text: str) -> str:
        """Replace every registered secret occurrence in ``text``."""
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTION_PLACEHOLDER)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record in place; never drops records."""
        if not self._secrets:
            return True
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # malformed args — leave it alone
            return True
        redacted = self.redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


REDACTOR = SecretRedactor()
"""Process-wide redactor. Populated by :func:`tearsheet.config.load_settings`."""


class ETFormatter(logging.Formatter):
    """Formatter that stamps records in the report timezone, not the host's.

    The pipeline is developed in one timezone, runs on GitHub Actions in UTC
    and reports on a New York trading session. Logging in ET keeps log
    timestamps directly comparable with everything on the page.
    """

    def __init__(self, fmt: str, timezone: str) -> None:
        super().__init__(fmt)
        self._tz = ZoneInfo(timezone)

    def formatTime(  # noqa: N802 — overriding logging's camelCase API
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Render the record's timestamp in the configured timezone."""
        stamped = datetime.fromtimestamp(record.created, self._tz)
        return stamped.strftime(datefmt or "%Y-%m-%d %H:%M:%S %Z")


def setup_logging(
    log_dir: Path, *, verbose: bool = False, timezone: str = DEFAULT_TIMEZONE
) -> logging.Logger:
    """Configure console and rotating-file logging for the run.

    Args:
        log_dir: Directory for ``tearsheet.log``. Created if absent.
        verbose: Emit DEBUG to the console instead of INFO. The file handler
            always records DEBUG so a failed CI run can be diagnosed after
            the fact.
        timezone: Timezone for log timestamps.

    Returns:
        The configured ``tearsheet`` logger.
    """
    tz = ZoneInfo(timezone)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("tearsheet")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    console_handler = RichHandler(
        console=get_console(),
        show_path=False,
        rich_tracebacks=True,
        omit_repeated_times=False,
        # Rich hands us a naive host-local datetime; restate it in the report
        # timezone so console and page timestamps always agree.
        log_time_format=lambda when: Text(when.astimezone(tz).strftime("%H:%M:%S")),
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(REDACTOR)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "tearsheet.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        ETFormatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", timezone)
    )
    file_handler.addFilter(REDACTOR)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO; their warnings still surface.
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``tearsheet`` namespace."""
    return logging.getLogger(f"tearsheet.{name}")


@dataclass
class RunReport:
    """Accumulates what happened during a run.

    Feeds the console summary, the page footer and the failure email.

    Attributes:
        mode: Live or mock.
        timezone: Display timezone for every timestamp (ET by default).
        started_at: Run start, in ``timezone``.
        api_calls: Call counts keyed by source name.
        timings: Wall-clock seconds keyed by step name.
        warnings: Non-fatal problems raised anywhere in the pipeline.
        sections: Section results in page order.
        artifacts: Paths of files produced, keyed by kind.
    """

    mode: RunMode
    timezone: str = DEFAULT_TIMEZONE
    started_at: datetime = field(default_factory=datetime.now)
    api_calls: Counter[str] = field(default_factory=Counter)
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sections: list[SectionResult] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise ``started_at`` into the configured display timezone."""
        self.started_at = datetime.now(ZoneInfo(self.timezone))

    # -- accumulation --------------------------------------------------------

    def record_api_call(self, source: str, count: int = 1) -> None:
        """Increment the call counter for ``source``."""
        self.api_calls[source] += count

    def warn(self, message: str) -> None:
        """Record a non-fatal problem and log it at WARNING."""
        self.warnings.append(message)
        get_logger("run").warning(message)

    def add_section(self, result: SectionResult) -> None:
        """Register a completed section and absorb its warnings."""
        self.sections.append(result)
        self.warnings.extend(result.warnings)

    def add_artifact(self, kind: str, path: Path) -> None:
        """Record a produced file, e.g. ``("html", output/index.html)``."""
        self.artifacts[kind] = path

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """Time a block of work and store the result under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = time.perf_counter() - start

    # -- derived views -------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        """Total wall-clock seconds since the run started."""
        return (datetime.now(ZoneInfo(self.timezone)) - self.started_at).total_seconds()

    @property
    def total_api_calls(self) -> int:
        """Sum of calls across every source."""
        return sum(self.api_calls.values())

    def sources_by_section(self) -> dict[str, list[SourceRef]]:
        """Map each section key to the sources that fed it, for the footer."""
        return {section.key: section.sources for section in self.sections}

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary, used by the footer and the failure email."""
        return {
            "mode": self.mode.value,
            "started_at": self.started_at.isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "api_calls": dict(self.api_calls),
            "total_api_calls": self.total_api_calls,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "warnings": self.warnings,
            "sections": {s.key: s.status.value for s in self.sections},
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
        }

    # -- presentation --------------------------------------------------------

    def print_summary(self) -> None:
        """Print the end-of-run summary table to the console."""
        console = get_console()
        console.print()

        status_styles = {
            SectionStatus.OK: "green",
            SectionStatus.PARTIAL: "yellow",
            SectionStatus.UNAVAILABLE: "red",
            SectionStatus.DISABLED: "dim",
            SectionStatus.PENDING: "dim cyan",
        }

        sections_table = Table(
            title="Sections", title_justify="left", header_style="bold", expand=False
        )
        sections_table.add_column("Section")
        sections_table.add_column("Status")
        sections_table.add_column("Sources")
        sections_table.add_column("Time", justify="right")
        for section in self.sections:
            sources = ", ".join(str(s) for s in section.sources) or "—"
            sections_table.add_row(
                section.title,
                f"[{status_styles[section.status]}]{section.status.value}[/]",
                sources,
                f"{section.duration_seconds:.2f}s",
            )
        console.print(sections_table)

        run_table = Table(title="Run", title_justify="left", header_style="bold", expand=False)
        run_table.add_column("Metric")
        run_table.add_column("Value", justify="right")
        run_table.add_row("Mode", self.mode.value)
        run_table.add_row("Started", self.started_at.strftime("%Y-%m-%d %H:%M:%S %Z"))
        run_table.add_row("Elapsed", f"{self.elapsed_seconds:.2f}s")
        for source, count in sorted(self.api_calls.items()):
            run_table.add_row(f"API calls · {source}", str(count))
        run_table.add_row("API calls · total", str(self.total_api_calls))
        run_table.add_row("Warnings", str(len(self.warnings)))
        console.print(run_table)

        if self.timings:
            timings_table = Table(
                title="Timings", title_justify="left", header_style="bold", expand=False
            )
            timings_table.add_column("Step")
            timings_table.add_column("Seconds", justify="right")
            for name, seconds in sorted(self.timings.items(), key=lambda kv: -kv[1]):
                timings_table.add_row(name, f"{seconds:.3f}")
            console.print(timings_table)

        if self.warnings:
            console.print("\n[bold yellow]Warnings[/]")
            for warning in self.warnings:
                console.print(f"  [yellow]•[/] {warning}")

        if self.artifacts:
            console.print("\n[bold]Artifacts[/]")
            for kind, path in self.artifacts.items():
                console.print(f"  [green]•[/] {kind}: {path}")
