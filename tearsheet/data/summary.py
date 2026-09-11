"""Console rendering of a collected :class:`~tearsheet.data.bundle.DataBundle`.

Its job is to make a run auditable at a glance: every series, its latest value,
the date that value was observed, which source supplied it, and whether it is
stale. Provenance is shown per row rather than per run, because the fallback
chain can leave a single page mixing three different sources.
"""

from __future__ import annotations

from rich.table import Table

from tearsheet.config import Settings
from tearsheet.data.bundle import DataBundle
from tearsheet.observability import get_console


def _fmt(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separators and fixed decimals."""
    return f"{value:,.{decimals}f}"


def print_prices(bundle: DataBundle, settings: Settings) -> None:
    """Print the daily price table, grouped by universe group."""
    console = get_console()
    universe = settings.config.universe

    table = Table(
        title=f"Daily prices — last completed session {bundle.session}",
        title_justify="left",
        header_style="bold",
        expand=False,
    )
    table.add_column("Group")
    table.add_column("Ticker")
    table.add_column("Name", max_width=24)
    table.add_column("Last", justify="right")
    table.add_column("1D %", justify="right")
    table.add_column("Obs date", justify="right")
    table.add_column("Bars", justify="right")
    table.add_column("Source")

    for group_name, instruments in universe.instrument_groups.items():
        for instrument in instruments:
            frame = bundle.prices.frames.get(instrument.ticker)
            if frame is None:
                table.add_row(
                    group_name,
                    instrument.ticker,
                    instrument.name,
                    "[red]—[/]",
                    "—",
                    "—",
                    "0",
                    "[red]unavailable[/]",
                )
                continue

            close = frame["close"]
            change = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else None
            change_text = (
                f"[green]+{change:.2f}[/]"
                if change is not None and change >= 0
                else f"[red]{change:.2f}[/]"
                if change is not None
                else "—"
            )
            source = bundle.prices.sources[instrument.ticker]
            source_text = f"[yellow]{source}[/]" if source.is_fallback else source.name

            table.add_row(
                group_name,
                instrument.ticker,
                instrument.name,
                _fmt(close.iloc[-1]),
                change_text,
                f"{close.index[-1]:%Y-%m-%d}",
                str(len(close)),
                source_text,
            )

    console.print(table)


def print_fred(bundle: DataBundle, settings: Settings) -> None:
    """Print every FRED series with its observation date and staleness."""
    console = get_console()
    next_release = bundle.calendar.next_release_by_series if bundle.calendar else {}

    table = Table(title="FRED series", title_justify="left", header_style="bold", expand=False)
    table.add_column("Group")
    table.add_column("Series ID")
    table.add_column("Name", max_width=28)
    table.add_column("Latest", justify="right")
    table.add_column("Obs date", justify="right")
    table.add_column("Age", justify="right")
    table.add_column("State")
    table.add_column("Next release", justify="right")

    for group_name, specs in settings.config.fred.series_groups.items():
        for spec in specs:
            values = bundle.fred.get(spec.id)
            staleness = bundle.staleness.get(spec.id)

            if values is None or values.empty:
                state = "[red]missing[/]"
                latest = obs = age = "—"
            else:
                latest = _fmt(values.iloc[-1] * spec.scale, 2)
                obs = f"{values.index[-1]:%Y-%m-%d}"
                age = (
                    f"{staleness.age_days}d"
                    if staleness and staleness.age_days is not None
                    else "—"
                )
                state = "[yellow]stale[/]" if staleness and staleness.is_stale else "[green]ok[/]"

            upcoming = next_release.get(spec.id)
            table.add_row(
                group_name.replace("_", " "),
                spec.id,
                spec.name,
                latest,
                obs,
                age,
                state,
                upcoming.isoformat() if upcoming else "—",
            )

    console.print(table)


def print_intraday(bundle: DataBundle) -> None:
    """Print the previous-session minute-bar coverage."""
    if not bundle.intraday:
        return
    console = get_console()
    table = Table(
        title=f"Intraday minute bars (Polygon) — session {bundle.session}",
        title_justify="left",
        header_style="bold",
        expand=False,
    )
    table.add_column("Ticker")
    table.add_column("Bars", justify="right")
    table.add_column("From", justify="right")
    table.add_column("To", justify="right")
    table.add_column("Open", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("VWAP", justify="right")

    for ticker, bars in bundle.intraday.items():
        vwap = (bars["vwap"] * bars["volume"]).sum() / bars["volume"].sum()
        table.add_row(
            ticker,
            str(len(bars)),
            f"{bars.index[0]:%H:%M}",
            f"{bars.index[-1]:%H:%M}",
            _fmt(bars["open"].iloc[0]),
            _fmt(bars["close"].iloc[-1]),
            _fmt(vwap),
        )
    console.print(table)


def print_calendar(bundle: DataBundle) -> None:
    """Print today's releases, the week ahead, and the next FOMC."""
    if bundle.calendar is None:
        return
    console = get_console()
    calendar = bundle.calendar

    table = Table(
        title="Economic calendar", title_justify="left", header_style="bold", expand=False
    )
    table.add_column("When")
    table.add_column("Time (typical)", justify="right")
    table.add_column("Release")

    for entry in calendar.today:
        table.add_row("[bold]TODAY[/]", entry.typical_time_et or "—", entry.name)
    for entry in calendar.upcoming:
        table.add_row(f"{entry.date:%a %d %b}", entry.typical_time_et or "—", entry.name)
    if not calendar.today and not calendar.upcoming:
        table.add_row("—", "—", "[dim]no whitelisted releases in the window[/]")

    console.print(table)
    if calendar.next_fomc:
        sep = " with SEP" if calendar.next_fomc.sep else ""
        console.print(
            f"[dim]Next FOMC decision: {calendar.next_fomc.date:%A %d %B %Y} "
            f"({calendar.days_to_fomc} days){sep}.[/]"
        )


def print_earnings(bundle: DataBundle) -> None:
    """Print notable earnings for the report date."""
    if bundle.earnings is None:
        return
    console = get_console()
    earnings = bundle.earnings

    if earnings.total == 0:
        console.print(
            f"[dim]Earnings: none of the {earnings.checked} watchlist names report on "
            f"{bundle.run_date}.[/]"
        )
        return

    table = Table(
        title="Notable earnings today", title_justify="left", header_style="bold", expand=False
    )
    table.add_column("Session")
    table.add_column("Ticker")
    table.add_column("Time", justify="right")
    table.add_column("EPS est.", justify="right")

    for label, group in (
        ("Before open", earnings.before_open),
        ("After close", earnings.after_close),
        ("Unscheduled", earnings.unscheduled),
    ):
        for event in group:
            table.add_row(
                label,
                event.ticker,
                event.time_label,
                _fmt(event.eps_estimate) if event.eps_estimate else "—",
            )
    console.print(table)


def print_headlines(bundle: DataBundle) -> None:
    """Print the deduplicated headlines."""
    if not bundle.headlines:
        return
    console = get_console()
    table = Table(title="Headlines", title_justify="left", header_style="bold", expand=False)
    table.add_column("Time", justify="right")
    table.add_column("Publisher", max_width=22)
    table.add_column("Headline", max_width=62)

    for item in bundle.headlines:
        when = f"{item.published_at:%d %b %H:%M}" if item.published_at else "—"
        table.add_row(when, item.publisher, item.title)
    console.print(table)


def print_sources(bundle: DataBundle) -> None:
    """Print which source supplied each domain, including fallbacks."""
    console = get_console()
    table = Table(
        title="Sources by domain", title_justify="left", header_style="bold", expand=False
    )
    table.add_column("Domain")
    table.add_column("Source(s)", overflow="fold")

    for domain, refs in sorted(bundle.sources.items()):
        rendered = ", ".join(f"[yellow]{ref}[/]" if ref.is_fallback else str(ref) for ref in refs)
        table.add_row(domain, rendered)
    console.print(table)


def print_data_summary(bundle: DataBundle, settings: Settings) -> None:
    """Print the full data summary for a run."""
    console = get_console()
    console.rule("[bold]Data summary[/]")
    print_prices(bundle, settings)
    print_fred(bundle, settings)
    print_intraday(bundle)
    print_calendar(bundle)
    print_earnings(bundle)
    print_headlines(bundle)
    print_sources(bundle)
    console.rule()
