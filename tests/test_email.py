"""Tests for email composition. Nothing here sends a message."""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pytest

from tearsheet.config import load_settings
from tearsheet.delivery import email as mail
from tearsheet.models import RunMode, RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_options(send_email: bool = True) -> RunOptions:
    return RunOptions(
        mode=RunMode.MOCK,
        send_email=send_email,
        build_pdf=False,
        force=False,
        open_result=False,
        publish=False,
        record=False,
        verbose=False,
        config_path=PROJECT_ROOT / "config.yaml",
        output_dir=Path("output"),
        run_date=date(2026, 9, 11),
    )


def header_section() -> SectionResult:
    return SectionResult(
        key="header",
        title="Key takeaways",
        status=SectionStatus.OK,
        context={
            "kpis": [
                {
                    "key": "^GSPC",
                    "label": "SPX",
                    "value": 7591.70,
                    "value_suffix": "",
                    "change": -0.58,
                    "change_suffix": "%",
                    "change_decimals": 2,
                },
                {
                    "key": "DGS10",
                    "label": "10Y",
                    "value": 4.95,
                    "value_suffix": "%",
                    "change": 12.0,
                    "change_suffix": "bp",
                    "change_decimals": 0,
                },
                {
                    "key": "^VIX",
                    "label": "VIX",
                    "value": 17.84,
                    "value_suffix": "",
                    "change": 8.38,
                    "change_suffix": "%",
                    "change_decimals": 2,
                },
                {
                    "key": "DX-Y.NYB",
                    "label": "DXY",
                    "value": 99.09,
                    "value_suffix": "",
                    "change": 0.32,
                    "change_suffix": "%",
                    "change_decimals": 2,
                },
            ],
            "flags": [],
            "session": date(2026, 9, 10),
        },
    )


# -- subject -----------------------------------------------------------------


def test_subject_quotes_levels_and_changes_by_market_convention():
    """VIX and a bond yield are quoted as levels; an index as a change."""
    settings = load_settings()

    subject = mail.build_subject([header_section()], settings, make_options())

    assert "SPX −0.58%" in subject
    assert "10Y 4.95%" in subject
    assert "VIX 17.84" in subject, "the VIX level, not its percentage change"
    assert "DXY +0.32%" in subject
    assert "11 Sep" in subject


def test_subject_degrades_when_a_figure_is_missing():
    settings = load_settings()
    empty = SectionResult(key="header", title="Key takeaways", context={"kpis": []})

    subject = mail.build_subject([empty], settings, make_options())

    assert "n/a" in subject


def test_subject_survives_a_broken_template():
    """A typo in the config template must not stop the email going out."""
    import dataclasses

    settings = load_settings()
    broken = dataclasses.replace(
        settings,
        config=settings.config.model_copy(
            update={
                "email": settings.config.email.model_copy(
                    update={"subject_template": "{nonexistent_field}"}
                )
            }
        ),
    )

    subject = mail.build_subject([header_section()], broken, make_options())

    assert "11 Sep" in subject
    assert settings.config.meta.title in subject


# -- body --------------------------------------------------------------------


def test_the_body_is_email_safe():
    """Outlook strips style blocks, web fonts and scripts; inline CSS survives."""
    settings = load_settings()
    report = RunReport(mode=RunMode.MOCK)
    sections = [header_section()]

    message = mail.build_message(sections, settings, make_options(), report)

    lowered = message.html.lower()
    assert "<style" not in lowered
    assert "<script" not in lowered
    assert "fonts.googleapis" not in lowered
    assert "plotly" not in lowered
    assert "style=" in lowered, "styling must be inline"


def test_the_body_carries_the_disclaimer():
    settings = load_settings()
    message = mail.build_message(
        [header_section()], settings, make_options(), RunReport(mode=RunMode.MOCK)
    )

    assert "not investment advice" in message.html


def test_a_quiet_day_is_stated_explicitly():
    settings = load_settings()
    message = mail.build_message(
        [header_section()], settings, make_options(), RunReport(mode=RunMode.MOCK)
    )

    assert "quiet session" in message.html


def test_the_failure_message_names_the_error_and_section_states():
    settings = load_settings()
    report = RunReport(mode=RunMode.MOCK)
    report.add_section(SectionResult(key="rates", title="Rates", status=SectionStatus.UNAVAILABLE))

    message = mail.build_failure_message(settings, make_options(), report, "HTTP 503 from FRED")

    assert "FAILED" in message.subject
    assert "HTTP 503 from FRED" in message.html
    assert "unavailable" in message.html


# -- attachments -------------------------------------------------------------


def test_attachment_is_base64_encoded(tmp_path):
    pdf = tmp_path / "sheet.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    payload = mail._attachment_payload(pdf, RunReport(mode=RunMode.MOCK))

    assert payload["filename"] == "sheet.pdf"
    assert base64.b64decode(payload["content"]) == b"%PDF-1.4 fake"
    assert payload["content_type"] == "application/pdf"


def test_an_oversized_attachment_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A huge PDF should cost the attachment, not the whole email."""
    monkeypatch.setattr(mail, "MAX_ATTACHMENT_BYTES", 10)
    pdf = tmp_path / "sheet.pdf"
    pdf.write_bytes(b"x" * 100)
    report = RunReport(mode=RunMode.MOCK)

    assert mail._attachment_payload(pdf, report) is None
    assert any("above the" in w for w in report.warnings)


def test_a_missing_attachment_is_skipped_not_fatal(tmp_path):
    report = RunReport(mode=RunMode.MOCK)

    assert mail._attachment_payload(tmp_path / "nope.pdf", report) is None
    assert report.warnings


# -- delivery guards ---------------------------------------------------------


def test_no_email_flag_skips_delivery():
    settings = load_settings()

    result = mail.deliver(
        [header_section()], settings, make_options(send_email=False), RunReport(mode=RunMode.MOCK)
    )

    assert result is None


def test_delivery_refuses_when_nothing_rendered():
    """Sending an empty sheet would look like a quiet market rather than a fault."""
    settings = load_settings()
    dead = [SectionResult(key="header", title="H", status=SectionStatus.UNAVAILABLE)]

    with pytest.raises(mail.EmailDeliveryError, match="refusing to send"):
        mail.deliver(dead, settings, make_options(), RunReport(mode=RunMode.MOCK))


def test_addresses_are_masked_in_logs():
    masked = mail._mask("jordanvinson7@outlook.com")

    assert masked.startswith("jo")
    assert "rdanvinson7" not in masked
    assert masked.endswith("@outlook.com")
