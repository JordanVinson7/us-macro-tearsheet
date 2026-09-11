"""Shared value types that cross package boundaries.

These exist so the data layer, the analytics layer and the renderer agree on
what they are passing each other. In particular :class:`SectionResult` is the
contract that makes section isolation possible: every section returns one,
whether it succeeded, degraded or failed outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class RunMode(StrEnum):
    """Where the pipeline gets its data from."""

    LIVE = "live"
    """Hit the real APIs."""

    MOCK = "mock"
    """Replay saved fixtures — no network calls at all."""


class SectionStatus(StrEnum):
    """Outcome of building one tear-sheet section."""

    OK = "ok"
    """Rendered with complete data."""

    PARTIAL = "partial"
    """Rendered, but something was missing or fell back to another source."""

    UNAVAILABLE = "unavailable"
    """Could not be built; the page shows a 'data unavailable' note."""

    DISABLED = "disabled"
    """Switched off in config.yaml."""

    PENDING = "pending"
    """Not implemented yet (used while the project is being built out)."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Provenance for one piece of data, surfaced in the page footer.

    Attributes:
        name: Short source identifier, e.g. ``"yfinance"`` or ``"fred"``.
        detail: Optional human-readable qualifier, e.g. the endpoint used.
        is_fallback: True when this source was reached via the fallback chain
            rather than being the first choice. Fallbacks are called out
            explicitly in the footer so a reader knows the data is second-best.
    """

    name: str
    detail: str | None = None
    is_fallback: bool = False

    def __str__(self) -> str:
        """Render as ``name (detail) [fallback]`` for display."""
        text = self.name
        if self.detail:
            text += f" ({self.detail})"
        if self.is_fallback:
            text += " [fallback]"
        return text


@dataclass(frozen=True, slots=True)
class Staleness:
    """Whether a series is older than its expected update cadence.

    Attributes:
        observation_date: Date of the most recent observation.
        expected_frequency: The cadence configured for the series.
        age_days: Calendar days between ``observation_date`` and the run date.
        is_stale: True when ``age_days`` exceeds the configured grace period.
    """

    observation_date: date | None
    expected_frequency: str
    age_days: int | None
    is_stale: bool


@dataclass
class SectionResult:
    """The output of one tear-sheet section.

    Every section returns one of these regardless of outcome, so the renderer
    never has to reason about exceptions and a single failed source can never
    take down the page.

    Attributes:
        key: Stable identifier matching the ``sections`` block in config.yaml.
        title: Display heading.
        status: Whether the section is usable.
        context: Template variables for this section's Jinja partial.
        sources: Provenance entries, aggregated into the footer.
        notes: Methodology or caveat lines shown beneath the section.
        warnings: Non-fatal problems, also collected into the run summary.
        error: Short error summary when ``status`` is UNAVAILABLE.
        duration_seconds: Wall-clock build time, for the run summary.
    """

    key: str
    title: str
    status: SectionStatus = SectionStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceRef] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def is_renderable(self) -> bool:
        """True when the section has content worth putting on the page."""
        return self.status in (SectionStatus.OK, SectionStatus.PARTIAL)


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Resolved command-line options for a single pipeline run.

    Attributes:
        mode: Live APIs or saved fixtures.
        send_email: Whether to deliver the email.
        build_pdf: Whether to render the PDF.
        force: Bypass the schedule/holiday guard and overwrite an existing report.
        open_result: Open the rendered HTML in the default browser afterwards.
        publish: Write into ``docs/`` (CI) rather than ``output/`` (local).
        record: Save the collected data as fixtures for --mock mode.
        verbose: Enable debug-level console logging.
        config_path: Path to config.yaml.
        output_dir: Directory the run writes into.
        run_date: The report date, in the configured timezone.
    """

    mode: RunMode
    send_email: bool
    build_pdf: bool
    force: bool
    open_result: bool
    publish: bool
    record: bool
    verbose: bool
    config_path: Path
    output_dir: Path
    run_date: date

    @property
    def is_mock(self) -> bool:
        """True when running from fixtures."""
        return self.mode is RunMode.MOCK


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Whether this invocation should produce a report.

    Attributes:
        should_run: The decision itself.
        reason: Human-readable explanation, logged either way.
        now_et: The evaluated time in the configured timezone.
    """

    should_run: bool
    reason: str
    now_et: datetime


# ---------------------------------------------------------------------------
# Data containers
#
# These carry pandas objects and so are the one part of this module that is not
# dependency-free. They live here anyway because they cross the data →
# analytics → sections boundary, and splitting the shared vocabulary across two
# modules would be worse than the extra import.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Quote:
    """A point-in-time price, used for overnight and pre-open levels.

    Attributes:
        ticker: Yahoo symbol.
        price: Latest traded price.
        previous_close: The prior session's close, for the overnight change.
        as_of: Timestamp of the quote, in the report timezone.
        source: Where the quote came from.
    """

    ticker: str
    price: float
    previous_close: float | None
    as_of: datetime | None
    source: SourceRef

    @property
    def change_pct(self) -> float | None:
        """Percentage change versus the previous close, or None if unknown."""
        if not self.previous_close:
            return None
        return (self.price / self.previous_close - 1.0) * 100.0


@dataclass
class PriceData:
    """Daily OHLCV history for a set of instruments.

    Attributes:
        frames: Per-ticker OHLCV frames, indexed by date.
        sources: Provenance per ticker, so a ticker sourced from a fallback can
            be marked as such in the footer.
        missing: Tickers no source could supply.
        as_of: The last completed session the data was trimmed to.
        suspect_returns: Dates per ticker whose one-day return is implausible
            and should be excluded from return distributions. On continuous
            futures these are almost always contract rolls: the price steps to
            a new contract and stays there, so the "return" is an artefact of
            the series construction, not a market move.
    """

    frames: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, SourceRef] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    as_of: date | None = None
    suspect_returns: dict[str, list[date]] = field(default_factory=dict)

    def __contains__(self, ticker: str) -> bool:
        """Whether a ticker has usable history."""
        return ticker in self.frames

    @property
    def tickers(self) -> list[str]:
        """Tickers with usable history."""
        return list(self.frames)

    def close(self, ticker: str) -> Any:
        """The close series for one ticker, or None when unavailable."""
        frame = self.frames.get(ticker)
        return None if frame is None else frame["close"]

    def closes(self) -> Any:
        """Every close series as one wide DataFrame, columns in insertion order."""
        import pandas as pd

        if not self.frames:
            return pd.DataFrame()
        return pd.DataFrame({t: f["close"] for t, f in self.frames.items()})

    def merge(self, other: PriceData) -> None:
        """Absorb another PriceData, without overwriting existing tickers.

        Used by the fallback chain: the primary source is merged first, then
        each fallback fills only the gaps it can.
        """
        for ticker, frame in other.frames.items():
            if ticker not in self.frames:
                self.frames[ticker] = frame
                self.sources[ticker] = other.sources[ticker]
                if ticker in other.suspect_returns:
                    self.suspect_returns[ticker] = other.suspect_returns[ticker]
        self.missing = [t for t in self.missing if t not in self.frames]


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One market headline.

    Only the headline, publisher, timestamp and link are ever stored — article
    text is never copied.

    Attributes:
        title: The headline as published.
        publisher: Name of the publication.
        url: Link to the original article.
        published_at: Publication time, in the report timezone.
        tickers: Symbols the publisher associated with the item.
        source: Which provider supplied the headline.
    """

    title: str
    publisher: str
    url: str
    published_at: datetime | None
    tickers: tuple[str, ...] = ()
    source: SourceRef | None = None
