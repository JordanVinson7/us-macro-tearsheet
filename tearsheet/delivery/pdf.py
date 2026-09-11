"""PDF rendering via headless Chromium.

The PDF is the *same* HTML as the web page, printed through the print
stylesheet — not a separately built document. That guarantees the two can never
drift apart.

Two details make the difference between a usable PDF and a page of blank
rectangles:

* **Wait for the charts.** Plotly draws asynchronously after the library loads
  from its CDN. Printing on ``load`` reliably produces empty chart frames, so
  the renderer blocks until every ``.plotly-graph-div`` has actually produced
  SVG.
* **Wait for the fonts.** The page uses Google Fonts; printing before
  ``document.fonts.ready`` resolves gives fallback metrics and reflowed tables.

The running header and footer come from Chromium's own templates rather than
CSS: Chromium does not implement CSS ``@page`` margin boxes, so
``headerTemplate``/``footerTemplate`` is the only route to a page number.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from tearsheet.config import Settings
from tearsheet.models import RunOptions
from tearsheet.observability import RunReport, get_logger

logger = get_logger("delivery.pdf")

#: Generous, because a cold Chromium plus a CDN fetch plus fourteen charts is
#: not fast on a shared CI runner.
CHART_TIMEOUT_MS = 45_000
PAGE_TIMEOUT_MS = 60_000

#: Chromium substitutes the contents of these classes when printing.
_HEADER_TEMPLATE = """
<div style="font-family: Georgia, serif; font-size: 8pt; color: #6b7280;
            width: 100%; padding: 0 12mm; display: flex;
            justify-content: space-between; border-bottom: 0.5pt solid #e7e9ee;
            padding-bottom: 3pt;">
  <span>{title}</span>
  <span>{report_date}</span>
</div>
"""

_FOOTER_TEMPLATE = """
<div style="font-family: Georgia, serif; font-size: 7.5pt; color: #9aa1ab;
            width: 100%; padding: 0 12mm; display: flex;
            justify-content: space-between; border-top: 0.5pt solid #e7e9ee;
            padding-top: 3pt;">
  <span>For education and information only — not investment advice.</span>
  <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
</div>
"""

#: Plotly bakes a pixel width into each figure when it draws, sized to the
#: screen viewport. Switching to print media narrows the container but leaves
#: those widths untouched, so every chart prints as a small box in the corner
#: of a large empty area. Resizing after the media switch is the fix.
_RESIZE_CHARTS = """
() => {
  if (!window.Plotly) return 0;
  const divs = Array.from(document.querySelectorAll('.plotly-graph-div'));
  divs.forEach(d => window.Plotly.Plots.resize(d));
  return divs.length;
}
"""

#: Resolves once the document has finished parsing AND every expected Plotly
#: figure has produced SVG.
#:
#: The expected count comes from the HTML source rather than the live DOM.
#: Counting the DOM alone is a race: evaluated before the body has parsed it
#: sees too few divs — or none — and a naive "all of them are ready" test then
#: passes trivially, printing blank chart frames.
_CHARTS_READY = """
(expected) => {
  if (document.readyState !== 'complete') return false;
  const divs = Array.from(document.querySelectorAll('.plotly-graph-div'));
  if (divs.length < expected) return false;
  return divs.every(d => d.querySelector('.main-svg') !== null);
}
"""

#: Marks each figure container emitted by plotly.io.to_html.
_CHART_MARKER = 'class="plotly-graph-div"'


class PdfUnavailableError(RuntimeError):
    """Raised when Playwright or its Chromium build is not usable."""


def render_pdf(
    html_path: Path,
    output_path: Path,
    settings: Settings,
    options: RunOptions,
    report: RunReport,
) -> Path | None:
    """Print the rendered sheet to A4 PDF.

    Args:
        html_path: The HTML file to print.
        output_path: Where to write the PDF.
        settings: Configuration, for the header text.
        options: Run options, for the report date.
        report: Run report, for warnings and artifact recording.

    Returns:
        The PDF path, or None if rendering failed. A missing PDF is never fatal:
        the web page is already published and the email can still be sent
        without an attachment.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on the environment
        report.warn(f"Playwright is not installed; skipping the PDF: {exc}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--disable-dev-shm-usage"])
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1600})
                page.set_default_timeout(PAGE_TIMEOUT_MS)

                expected = html_path.read_text(encoding="utf-8").count(_CHART_MARKER)

                page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
                page.wait_for_function(_CHARTS_READY, arg=expected, timeout=CHART_TIMEOUT_MS)
                page.evaluate("() => document.fonts.ready")

                rendered = page.evaluate(
                    "() => document.querySelectorAll('.plotly-graph-div').length"
                )
                logger.info("All %d of %d charts rendered; printing to PDF", rendered, expected)

                # Print media must be emulated explicitly: page.pdf() applies it
                # for the print itself, but any layout the page computed on load
                # would otherwise still be the screen layout.
                page.emulate_media(media="print")
                # A4 content width at the margins below, in CSS pixels.
                page.set_viewport_size({"width": 760, "height": 1123})
                page.evaluate(_RESIZE_CHARTS)
                page.wait_for_timeout(600)

                page.pdf(
                    path=str(output_path),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=False,
                    display_header_footer=True,
                    header_template=_build_header(settings, options.run_date),
                    footer_template=_FOOTER_TEMPLATE,
                    margin={
                        "top": "18mm",
                        "bottom": "16mm",
                        "left": "11mm",
                        "right": "11mm",
                    },
                )
            finally:
                browser.close()

    except PlaywrightError as exc:
        report.warn(
            f"PDF rendering failed: {exc}. "
            "If Chromium is missing, run: playwright install --with-deps chromium"
        )
        return None
    except Exception as exc:  # noqa: BLE001 — the PDF is optional, the page is not
        report.warn(f"PDF rendering failed: {type(exc).__name__}: {exc}")
        return None

    size_kb = output_path.stat().st_size / 1024
    logger.info("Wrote %s (%.0f KB)", output_path, size_kb)
    report.add_artifact("pdf", output_path)
    return output_path


def _build_header(settings: Settings, run_date: date) -> str:
    """Fill the running header template with the report title and date."""
    return _HEADER_TEMPLATE.format(
        title=settings.config.meta.title,
        report_date=run_date.strftime("%d %B %Y"),
    )


def pdf_filename(settings: Settings, run_date: date) -> str:
    """The configured PDF file name for a report date."""
    return settings.config.output.pdf_filename.format(date=run_date.isoformat())
