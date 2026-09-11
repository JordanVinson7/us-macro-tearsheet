"""HTML page assembly.

The page is a single self-contained file: the stylesheet is inlined and only
Plotly is fetched from a CDN, once. Inlining the CSS keeps every archived sheet
correct forever — an archive from six months ago still renders exactly as it
did, even if the stylesheet has since changed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from tearsheet import __version__
from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport, get_logger
from tearsheet.render.filters import FILTERS

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle

logger = get_logger("render.html")

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

#: Pinned so an archived page keeps rendering identically years from now.
PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.0.1.min.js"


def build_environment() -> Environment:
    """Construct the Jinja environment.

    ``StrictUndefined`` is deliberate: a typo in a template variable raises at
    render time instead of silently producing a blank cell in a financial
    table, which is exactly the kind of error nobody notices.
    """
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters.update(FILTERS)
    return environment


def read_stylesheets() -> str:
    """Concatenate the screen and print stylesheets for inlining."""
    parts = []
    for name in ("screen.css", "print.css"):
        path = STATIC_DIR / name
        if path.exists():
            parts.append(f"/* --- {name} --- */\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def section_template(environment: Environment, result: SectionResult) -> str:
    """Pick the partial for a section, falling back to the placeholder."""
    if not result.is_renderable:
        return "sections/_unavailable.html.j2"
    candidate = f"sections/{result.key}.html.j2"
    try:
        environment.get_template(candidate)
    except Exception:  # noqa: BLE001 — a missing partial must not break the page
        logger.warning("No template for section %r; rendering placeholder.", result.key)
        return "sections/_unavailable.html.j2"
    return candidate


def render_page(
    sections: list[SectionResult],
    bundle: DataBundle,
    settings: Settings,
    options: RunOptions,
    report: RunReport,
) -> str:
    """Render the complete tear-sheet.

    Args:
        sections: Section results in page order.
        bundle: The collected data, used for footer provenance.
        settings: Configuration.
        options: Run options.
        report: Run report, for the footer's source and timing summary.

    Returns:
        The full HTML document.
    """
    environment = build_environment()
    timezone = ZoneInfo(settings.config.meta.timezone)
    generated_at = datetime.now(timezone)

    visible = [s for s in sections if s.status is not SectionStatus.DISABLED]

    context: dict[str, Any] = {
        "config": settings.config,
        "meta": settings.config.meta,
        "options": options,
        "sections": [{"result": s, "template": section_template(environment, s)} for s in visible],
        "nav": [{"key": s.key, "title": s.title} for s in visible],
        "run_date": options.run_date,
        "session": bundle.session,
        "generated_at": generated_at,
        "version": __version__,
        "stylesheet": read_stylesheets(),
        "plotly_cdn": PLOTLY_CDN,
        "footer": build_footer(sections, bundle, settings, report),
        # The masthead shows "data as of" per source, which the header
        # section computes; surface it at page level for the template.
        "footer_as_of": next(
            (s.context.get("as_of", {}) for s in sections if s.key == "header"), {}
        ),
        "disclaimer": settings.config.display.disclaimer,
        "is_mock": options.is_mock,
    }
    return environment.get_template("base.html.j2").render(**context)


def build_footer(
    sections: list[SectionResult],
    bundle: DataBundle,
    settings: Settings,
    report: RunReport,
) -> dict[str, Any]:
    """Assemble the footer: sources per section, methodology, and run metadata."""
    source_rows = [
        {
            "section": s.title,
            "sources": [str(ref) for ref in s.sources] or ["—"],
            "used_fallback": any(ref.is_fallback for ref in s.sources),
            "status": s.status.value,
        }
        for s in sections
        if s.status is not SectionStatus.DISABLED
    ]

    methodology = [{"section": s.title, "notes": s.notes} for s in sections if s.notes]

    return {
        "source_rows": source_rows,
        "methodology": methodology,
        "warnings": report.warnings,
        "api_calls": dict(report.api_calls),
        "total_api_calls": report.total_api_calls,
        "elapsed": report.elapsed_seconds,
        "missing_prices": bundle.prices.missing,
        "invalid_series": bundle.fred_invalid,
        "suspect_returns": bundle.prices.suspect_returns,
        "mode": report.mode.value,
    }
