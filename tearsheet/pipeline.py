"""Run orchestration.

The pipeline walks a fixed list of sections. Each one is built behind
:func:`build_section`, which converts any exception into an UNAVAILABLE
:class:`~tearsheet.models.SectionResult` rather than letting it propagate.
That single choke point is what makes the brief's isolation requirement true
by construction: one dead API cannot take down the page or the email.

Phase 1 registers the section list and reports configuration; the section
builders themselves arrive in later phases.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.panel import Panel
from rich.table import Table

from tearsheet.charts.template import register_template
from tearsheet.config import Settings
from tearsheet.data.bundle import DataBundle, collect, save_fixtures
from tearsheet.data.summary import print_data_summary
from tearsheet.delivery.email import EmailDeliveryError, deliver
from tearsheet.delivery.pdf import pdf_filename, render_pdf
from tearsheet.delivery.publish import write_outputs
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport, get_console, get_logger
from tearsheet.render.html import render_page
from tearsheet.sections import BUILDERS

logger = get_logger("pipeline")

SECTION_TITLES: dict[str, str] = {
    "header": "Header & key takeaways",
    "calendar": "Today's calendar",
    "overnight": "Overnight & futures",
    "equities": "Equities",
    "rates": "Rates & the Fed",
    "credit": "Credit",
    "macro": "Macro dashboard",
    "fx_commodities": "FX, commodities & crypto",
    "cross_asset": "Cross-asset & regime",
    "headlines": "Headlines",
}
"""Section keys in page order, mapped to their display headings."""

SectionBuilder = Callable[[DataBundle, Settings, RunOptions, RunReport], SectionResult]

#: Returned when the sheet was built but could not be delivered.
EXIT_DELIVERY_FAILED = 3

#: Section builders. A key with no entry renders as PENDING.
SECTION_BUILDERS: dict[str, SectionBuilder] = dict(BUILDERS)


def build_section(
    key: str,
    bundle: DataBundle,
    settings: Settings,
    options: RunOptions,
    report: RunReport,
) -> SectionResult:
    """Build one section, converting any failure into a renderable placeholder.

    Args:
        key: Section key, matching ``sections`` in config.yaml.
        bundle: The collected data.
        settings: Loaded configuration and credentials.
        options: Resolved command-line options.
        report: Run report to accumulate timings and warnings into.

    Returns:
        A section result. Never raises: an exception inside a builder becomes
        ``SectionStatus.UNAVAILABLE`` with the error summarised for the page.
    """
    title = SECTION_TITLES.get(key, key.replace("_", " ").title())

    if not settings.config.section_enabled(key):
        return SectionResult(key=key, title=title, status=SectionStatus.DISABLED)

    builder = SECTION_BUILDERS.get(key)
    if builder is None:
        return SectionResult(
            key=key,
            title=title,
            status=SectionStatus.PENDING,
            notes=["Not implemented yet."],
        )

    started = time.perf_counter()
    try:
        result = builder(bundle, settings, options, report)
    except Exception as exc:  # noqa: BLE001 — deliberate isolation boundary
        logger.exception("Section %r failed", key)
        result = SectionResult(
            key=key,
            title=title,
            status=SectionStatus.UNAVAILABLE,
            error=f"{type(exc).__name__}: {exc}",
            notes=["Data unavailable — this section could not be built."],
        )
    result.duration_seconds = time.perf_counter() - started
    return result


def print_config_summary(settings: Settings, options: RunOptions) -> None:
    """Print the resolved configuration so a run is self-documenting.

    Safe to call in CI logs: credential values are never included, only
    whether each one is set.
    """
    console = get_console()
    config = settings.config

    console.print(
        Panel(
            f"[bold]{config.meta.title}[/]\n[dim]{config.meta.subtitle}[/]\n\n"
            f"Report date  [bold]{options.run_date:%A %d %B %Y}[/]\n"
            f"Mode         [bold]{options.mode.value}[/]\n"
            f"Output       [bold]{options.output_dir}[/]",
            expand=False,
            border_style="blue",
        )
    )

    universe = Table(title="Universe", title_justify="left", header_style="bold", expand=False)
    universe.add_column("Group")
    universe.add_column("Count", justify="right")
    universe.add_column("Members", overflow="fold")
    for name, instruments in config.universe.instrument_groups.items():
        universe.add_row(
            name.replace("_", " "),
            str(len(instruments)),
            ", ".join(i.ticker for i in instruments),
        )
    universe.add_row(
        "intraday (polygon)",
        str(len(config.universe.intraday.tickers)),
        ", ".join(config.universe.intraday.tickers),
    )
    universe.add_row(
        "ratios", str(len(config.universe.ratios)), ", ".join(r.key for r in config.universe.ratios)
    )
    console.print(universe)

    fred_table = Table(title="FRED series", title_justify="left", header_style="bold", expand=False)
    fred_table.add_column("Group")
    fred_table.add_column("Count", justify="right")
    fred_table.add_column("IDs", overflow="fold")
    for name, series in config.fred.series_groups.items():
        fred_table.add_row(
            name.replace("_", " "), str(len(series)), ", ".join(s.id for s in series)
        )
    fred_table.add_row(
        "fallback levels",
        str(len(config.fred.fallback_levels)),
        ", ".join(config.fred.fallback_levels.values()),
    )
    console.print(fred_table)

    settings_table = Table(
        title="Key settings", title_justify="left", header_style="bold", expand=False
    )
    settings_table.add_column("Setting")
    settings_table.add_column("Value", justify="right")
    settings_table.add_row("Timezone", config.meta.timezone)
    settings_table.add_row(
        "Publish window (ET)",
        f"{config.schedule.window_start:%H:%M}–{config.schedule.window_end:%H:%M}",
    )
    settings_table.add_row("Exchange calendar", config.schedule.exchange_calendar)
    settings_table.add_row(
        "Polygon rate limit",
        f"{config.sources.polygon.rate_limit.max_calls} / "
        f"{config.sources.polygon.rate_limit.per_seconds:.0f}s",
    )
    settings_table.add_row("Polygon call budget", str(config.sources.polygon.call_budget))
    settings_table.add_row("yfinance fallbacks", " → ".join(config.sources.yfinance.fallback_chain))
    settings_table.add_row("Archive retention", f"{config.output.retain_days} days")
    settings_table.add_row("Realised vol window", f"{config.analytics.realised_vol_window}d")
    settings_table.add_row("Correlation window", f"{config.analytics.correlation_window}d")
    settings_table.add_row("Flag |z| threshold", f"{config.flags.return_zscore:.1f}")
    settings_table.add_row("Regime components", str(len(config.regime.components)))
    console.print(settings_table)

    secrets_table = Table(
        title="Credentials", title_justify="left", header_style="bold", expand=False
    )
    secrets_table.add_column("Variable")
    secrets_table.add_column("Status", justify="right")
    styles = {"set": "green", "placeholder": "yellow", "missing": "red"}
    for name, status in settings.secrets.status().items():
        secrets_table.add_row(name, f"[{styles[status]}]{status}[/]")
    console.print(secrets_table)
    console.print("[dim]Credential values are never read, printed or logged by this summary.[/]")


def _render(bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport) -> None:
    """Render the page and write it out, without letting a failure abort the run.

    Rendering is guarded like any section: a template error should still leave
    the run summary, the warnings and (in later phases) the failure email
    intact, rather than killing the process with a traceback.
    """
    try:
        with report.timed("render"):
            html = render_page(report.sections, bundle, settings, options, report)
        with report.timed("publish"):
            write_outputs(html, settings, options, report)
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        logger.exception("Rendering failed")
        report.warn(f"Rendering failed: {type(exc).__name__}: {exc}")


def _deliver(settings: Settings, options: RunOptions, report: RunReport) -> int:
    """Render the PDF and send the email.

    A missing PDF is not fatal — the page is already published and the email is
    still worth sending. A *delivery* failure is fatal: the whole point of the
    schedule is that the sheet arrives, so an undelivered email must surface as
    a failed workflow rather than a green tick.

    Returns:
        ``0`` on success, ``EXIT_DELIVERY_FAILED`` if the email could not be sent.
    """
    html_path = report.artifacts.get("html")
    pdf_path = None

    if options.build_pdf and html_path is not None:
        with report.timed("pdf"):
            pdf_path = render_pdf(
                html_path,
                options.output_dir / pdf_filename(settings, options.run_date),
                settings,
                options,
                report,
            )
    elif not options.build_pdf:
        logger.info("PDF rendering skipped (--no-pdf).")

    try:
        with report.timed("email"):
            deliver(report.sections, settings, options, report, pdf_path)
    except EmailDeliveryError as exc:
        report.warn(f"Email delivery failed: {exc}")
        logger.error("Email delivery failed: %s", exc)
        return EXIT_DELIVERY_FAILED

    return 0


def run(settings: Settings, options: RunOptions, report: RunReport) -> int:
    """Execute a full pipeline run.

    Args:
        settings: Loaded configuration and credentials.
        options: Resolved command-line options.
        report: Run report to populate.

    Returns:
        Process exit code: 0 on success, 1 if every section failed.
    """
    logger.info("Starting run for %s in %s mode", options.run_date, options.mode.value)

    # The Plotly template must be registered BEFORE any figure is created: a
    # figure captures pio.templates.default at construction time, so
    # registering it later would leave every chart on Plotly's stock theme.
    # Doing it here rather than at import time keeps chart modules free of
    # global side effects.
    register_template()

    unknown = set(settings.config.sections) - set(SECTION_TITLES)
    for key in sorted(unknown):
        report.warn(f"config.yaml enables unknown section {key!r}; it will be ignored.")

    print_config_summary(settings, options)

    with report.timed("collect"):
        bundle = collect(settings, options, report)
    print_data_summary(bundle, settings)

    if options.record and not options.is_mock:
        counts = save_fixtures(bundle, settings.project_root / "fixtures")
        logger.info("Recorded fixtures: %s", counts)

    for key in SECTION_TITLES:
        with report.timed(f"section:{key}"):
            report.add_section(build_section(key, bundle, settings, options, report))

    options.output_dir.mkdir(parents=True, exist_ok=True)
    _render(bundle, settings, options, report)
    delivery_code = _deliver(settings, options, report)

    renderable = [s for s in report.sections if s.is_renderable]
    # PENDING sections are unbuilt scaffolding, not failures, so they are not
    # counted when deciding whether the run produced anything usable.
    inactive = (SectionStatus.DISABLED, SectionStatus.PENDING)
    active = [s for s in report.sections if s.status not in inactive]
    if active and not renderable:
        logger.error("No section produced usable output.")
        return 1

    logger.info(
        "Run complete: %d/%d sections renderable, %d API calls, %.2fs",
        len(renderable),
        len(active),
        report.total_api_calls,
        report.elapsed_seconds,
    )
    return delivery_code
