"""Tests for formatting filters, the archive, and full-page rendering.

The render test is the important one: it builds the entire page from fixtures,
which is what catches a broken Jinja template or a renamed context key before
it reaches a 07:00 cron run.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tearsheet.config import load_settings
from tearsheet.data.bundle import load_from_fixtures
from tearsheet.delivery.publish import archive_entries, prune_archive, write_outputs
from tearsheet.models import RunMode, RunOptions, SectionStatus
from tearsheet.observability import RunReport
from tearsheet.render.filters import (
    basis_points,
    day,
    level,
    number,
    ordinal,
    percent,
    signed,
    tone,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -- filters -----------------------------------------------------------------


def test_changes_always_carry_an_explicit_sign():
    """Colour must never be the only signal that a number is negative."""
    assert percent(2.5).startswith("+")
    assert signed(2.5).startswith("+")
    assert basis_points(4.0).startswith("+")


def test_negatives_use_a_typographic_minus_so_columns_align():
    assert "−" in percent(-2.5)
    assert "-" not in percent(-2.5)


def test_missing_values_render_as_na_not_zero():
    """The analytics return None for 'not enough history'; that must survive."""
    for renderer in (number, signed, percent, basis_points, level, ordinal):
        assert "n/a" in renderer(None)


def test_number_formatting():
    assert number(1234.5) == "1,234.50"
    assert percent(-1.239) == "−1.24%"
    assert basis_points(12.4) == "+12bp"
    assert level(4.827) == "4.83"


@pytest.mark.parametrize(
    ("value", "expected"), [(1.0, "pos"), (-1.0, "neg"), (0.0, "flat"), (None, "flat")]
)
def test_tone_classes(value, expected):
    assert tone(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (73, "73rd")],
)
def test_ordinal_suffixes(value, expected):
    assert ordinal(value) == expected


def test_day_formats_or_dashes():
    assert day(date(2026, 9, 11)) == "11 Sep 2026"
    assert day(None) == "—"


# -- archive -----------------------------------------------------------------


def make_archive(tmp_path: Path, days: list[str]) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    for stamp in days:
        (archive / f"{stamp}.html").write_text("<html></html>")
    return archive


def test_archive_entries_are_listed_newest_first(tmp_path):
    archive = make_archive(tmp_path, ["2026-09-09", "2026-09-11", "2026-09-10"])

    entries = archive_entries(archive)

    assert [e.date.day for e in entries] == [11, 10, 9]


def test_non_dated_files_are_ignored(tmp_path):
    archive = make_archive(tmp_path, ["2026-09-10"])
    (archive / "index.html").write_text("<html></html>")

    assert len(archive_entries(archive)) == 1


def test_pruning_removes_sheets_beyond_the_retention_window(tmp_path):
    """Each sheet is ~0.5MB, so an unpruned archive grows the repo unboundedly."""
    archive = make_archive(tmp_path, ["2026-01-02", "2026-08-20", "2026-09-10"])

    removed = prune_archive(archive, retain_days=30, today=date(2026, 9, 11))

    assert removed == ["2026-01-02.html"]
    assert len(archive_entries(archive)) == 2


def test_retention_of_zero_disables_pruning(tmp_path):
    archive = make_archive(tmp_path, ["2020-01-02"])

    assert prune_archive(archive, retain_days=0, today=date(2026, 9, 11)) == []


# -- full page ---------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> tuple[str, Path]:
    """Render the whole sheet from fixtures, exactly as --mock does."""
    from tearsheet.charts.template import register_template
    from tearsheet.pipeline import SECTION_TITLES, build_section
    from tearsheet.render.html import render_page

    register_template()
    settings = load_settings()
    output = tmp_path_factory.mktemp("out")
    options = RunOptions(
        mode=RunMode.MOCK,
        send_email=False,
        build_pdf=False,
        force=False,
        open_result=False,
        publish=False,
        record=False,
        verbose=False,
        config_path=PROJECT_ROOT / "config.yaml",
        output_dir=output,
        run_date=date(2026, 9, 11),
    )
    report = RunReport(mode=RunMode.MOCK)
    bundle = load_from_fixtures(settings, options, report)

    for key in SECTION_TITLES:
        report.add_section(build_section(key, bundle, settings, options, report))

    html = render_page(report.sections, bundle, settings, options, report)
    write_outputs(html, settings, options, report)
    return html, output


def test_every_section_renders(rendered):
    html, _ = rendered

    for key in [
        "header",
        "calendar",
        "overnight",
        "equities",
        "rates",
        "credit",
        "macro",
        "fx_commodities",
        "cross_asset",
        "headlines",
    ]:
        assert f'id="section-{key}"' in html


def test_no_section_failed_to_build(rendered):
    """A template or context error must not silently become a placeholder."""
    from tearsheet.pipeline import SECTION_TITLES, build_section

    _, _ = rendered
    settings = load_settings()
    options = RunOptions(
        RunMode.MOCK,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        PROJECT_ROOT / "config.yaml",
        Path("output"),
        date(2026, 9, 11),
    )
    report = RunReport(mode=RunMode.MOCK)
    bundle = load_from_fixtures(settings, options, report)

    statuses = {
        key: build_section(key, bundle, settings, options, report).status for key in SECTION_TITLES
    }

    assert SectionStatus.UNAVAILABLE not in statuses.values(), statuses
    assert SectionStatus.PENDING not in statuses.values(), statuses


def test_no_unrendered_template_syntax_survives(rendered):
    html, _ = rendered

    assert "{{" not in html
    assert "{%" not in html


def test_plotly_is_loaded_once_from_a_cdn(rendered):
    """Embedding the library per figure would add ~3MB for each of 14 charts."""
    html, _ = rendered

    assert html.count('<script src="https://cdn.plot.ly') == 1
    assert html.count('class="plotly-graph-div"') >= 10


def test_the_page_is_self_contained_apart_from_plotly_and_fonts(rendered):
    html, _ = rendered

    assert "<style>" in html, "the stylesheet must be inlined for the archive"
    assert 'rel="stylesheet"' in html  # only the Google Fonts link


def test_outputs_include_the_latest_sheet_and_a_dated_archive(rendered):
    _, output = rendered

    assert (output / "index.html").exists()
    assert (output / "archive" / "2026-09-11.html").exists()
    assert (output / "archive" / "index.html").exists()


def test_the_disclaimer_is_present(rendered):
    html, _ = rendered

    assert "not investment advice" in html


def test_the_regime_is_labelled_as_not_a_trading_signal(rendered):
    html, _ = rendered

    assert "not a trading signal" in html


def test_charts_use_the_project_plotly_template(rendered):
    """Charts must carry the project template, not Plotly's stock theme.

    A figure captures the default template when it is built, not when it is
    serialised, so registration has to happen before any section runs.
    """
    import json
    import re

    html, _ = rendered
    payloads = re.findall(
        r'Plotly\.newPlot\(\s*"[^"]+",\s*\[.*?\],\s*(\{.*?\}),\s*\{"displayModeBar"', html, re.S
    )
    assert payloads, "no Plotly figures found in the page"

    layout = json.loads(payloads[0])
    template = layout.get("template", {}).get("layout", {})

    assert template.get("plot_bgcolor") == "#ffffff", "charts fell back to Plotly's stock theme"
    assert template.get("legend", {}).get("orientation") == "h"


def test_pdf_expected_chart_count_matches_the_page(rendered):
    """The PDF renderer derives how many charts to wait for from the HTML source.

    If that marker ever stops matching what plotly.io.to_html emits, the PDF
    would wait for zero charts and print blank frames.
    """
    from tearsheet.delivery.pdf import _CHART_MARKER

    html, _ = rendered

    assert html.count(_CHART_MARKER) == html.count('class="plotly-graph-div"')
    assert html.count(_CHART_MARKER) >= 10
