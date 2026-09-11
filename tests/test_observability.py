"""Tests for logging redaction and the run report."""

from __future__ import annotations

import logging

from tearsheet.models import RunMode, SectionResult, SectionStatus
from tearsheet.observability import RunReport, SecretRedactor


def test_redactor_scrubs_registered_secrets():
    redactor = SecretRedactor()
    redactor.register("a_very_secret_key_value")

    assert "a_very_secret_key_value" not in redactor.redact(
        "GET https://api.example.com?apikey=a_very_secret_key_value"
    )


def test_redactor_ignores_short_values():
    """Redacting a 3-character string would mangle text without protecting it."""
    redactor = SecretRedactor()
    redactor.register("abc")

    assert redactor.redact("abc def") == "abc def"


def test_redactor_rewrites_log_records():
    redactor = SecretRedactor()
    redactor.register("supersecrettoken")
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=%s",
        args=("supersecrettoken",),
        exc_info=None,
    )

    redactor.filter(record)

    assert "supersecrettoken" not in record.getMessage()


def test_run_report_accumulates():
    report = RunReport(mode=RunMode.MOCK)
    report.record_api_call("fred", 3)
    report.record_api_call("polygon")
    report.add_section(
        SectionResult(key="rates", title="Rates", status=SectionStatus.OK, warnings=["late data"])
    )

    assert report.total_api_calls == 4
    assert report.warnings == ["late data"]
    assert report.as_dict()["sections"] == {"rates": "ok"}
