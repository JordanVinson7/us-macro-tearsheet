"""Email delivery via Resend.

Two messages can be sent:

* the **morning summary**, with the KPI table, the flags, today's calendar, a
  link to the live page and the PDF attached; and
* a **failure notice**, when the run could not produce a sheet — a silent
  failure is otherwise indistinguishable from a quiet market.

The body is email-safe HTML: tables and inline styles only, no ``<style>``
block, no web fonts, no JavaScript and no charts. Outlook strips or ignores
most of those, and an email that arrives as unstyled text is worse than one
designed to be plain.

**Resend free plan.** The sender must be ``onboarding@resend.dev`` and mail is
delivered only to the address the Resend account was registered with. That is
the intended recipient here, so the restriction costs nothing — but it does
mean the address in ``EMAIL_TO`` and the Resend account address must match.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tearsheet.config import MissingSecretError, Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport, get_logger
from tearsheet.render.filters import basis_points, level, percent
from tearsheet.render.html import build_environment

logger = get_logger("delivery.email")

POSITIVE = "#0b7a5a"
NEGATIVE = "#b3261e"
NEUTRAL = "#6b7280"

#: Resend rejects attachments above roughly 40MB; the sheet is under 1MB, so
#: this is a guard against a pathological run rather than a real limit.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

#: Subject-line fields: the KPI key, and whether the market convention is to
#: quote the level or the change. The VIX and a bond yield are quoted as levels
#: ("VIX 17.84", "10Y 4.95%"); an index or a currency as a change.
SUBJECT_KEYS: dict[str, tuple[str, str]] = {
    "spx": ("^GSPC", "change"),
    "ten_year": ("DGS10", "level"),
    "vix": ("^VIX", "level"),
    "dxy": ("DX-Y.NYB", "change"),
}


class EmailDeliveryError(RuntimeError):
    """Raised when Resend could not be reached or rejected the message."""


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """A composed message, ready to hand to Resend.

    Attributes:
        subject: Subject line.
        html: Email-safe HTML body.
        attachment: Optional file to attach.
    """

    subject: str
    html: str
    attachment: Path | None = None


def _section(sections: list[SectionResult], key: str) -> SectionResult | None:
    """Find a section result by key."""
    return next((s for s in sections if s.key == key), None)


def _kpi_rows(header: SectionResult | None) -> list[dict[str, Any]]:
    """Format the KPI strip for the email table."""
    if header is None:
        return []

    rows = []
    for kpi in header.context.get("kpis", []):
        change = kpi.get("change")
        rows.append(
            {
                "label": kpi["label"],
                "value_text": f"{level(kpi['value'])}{kpi['value_suffix']}",
                "change_text": (
                    basis_points(change, kpi["change_decimals"])
                    if kpi["change_suffix"] == "bp"
                    else percent(change, kpi["change_decimals"])
                ),
                "colour": NEUTRAL if change is None else (POSITIVE if change >= 0 else NEGATIVE),
            }
        )
    return rows


def build_subject(sections: list[SectionResult], settings: Settings, options: RunOptions) -> str:
    """Build the subject line: the date plus a few headline numbers.

    The subject has to be readable in a phone notification, so it carries the
    figures themselves rather than a generic title.
    """
    header = _section(sections, "header")
    kpis = {k["key"]: k for k in (header.context.get("kpis", []) if header else [])}

    values: dict[str, str] = {"date": options.run_date.strftime("%d %b")}
    for field, (key, convention) in SUBJECT_KEYS.items():
        kpi = kpis.get(key)
        if kpi is None:
            values[field] = "n/a"
        elif convention == "level":
            values[field] = f"{level(kpi['value'])}{kpi['value_suffix']}"
        else:
            values[field] = percent(kpi.get("change"), kpi["change_decimals"])

    try:
        return settings.config.email.subject_template.format(**values)
    except KeyError as exc:
        logger.warning("Subject template references unknown field %s", exc)
        return f"{values['date']} — {settings.config.meta.title}"


def build_message(
    sections: list[SectionResult],
    settings: Settings,
    options: RunOptions,
    report: RunReport,
    pdf_path: Path | None = None,
) -> EmailMessage:
    """Compose the morning summary email."""
    header = _section(sections, "header")
    calendar = _section(sections, "calendar")
    calendar_context = calendar.context if calendar else {}

    earnings = calendar_context.get("earnings")
    earnings_line = None
    if earnings is not None and earnings.total:
        earnings_line = (
            f"{earnings.total} notable earnings today "
            f"({len(earnings.before_open)} before open, {len(earnings.after_close)} after close)."
        )

    chart_count = sum(len(s.context.get("charts", {})) for s in sections)
    html = (
        build_environment()
        .get_template("email.html.j2")
        .render(
            title=settings.config.meta.title,
            run_date=options.run_date,
            session=next(
                (s.context.get("session") for s in sections if s.key == "header"), options.run_date
            ),
            kpis=_kpi_rows(header),
            flags=header.context.get("flags", []) if header else [],
            releases=calendar_context.get("today", []),
            fomc=calendar_context.get("fomc"),
            days_to_fomc=calendar_context.get("days_to_fomc"),
            earnings_line=earnings_line,
            site_url=settings.secrets.site_url,
            chart_count=chart_count,
            section_count=len([s for s in sections if s.is_renderable]),
            has_attachment=pdf_path is not None and pdf_path.exists(),
            disclaimer=settings.config.display.disclaimer,
            generated_at=datetime.now(ZoneInfo(settings.config.meta.timezone)),
            warning_count=len(report.warnings),
        )
    )

    return EmailMessage(
        subject=build_subject(sections, settings, options),
        html=html,
        attachment=pdf_path,
    )


def build_failure_message(
    settings: Settings, options: RunOptions, report: RunReport, error: str
) -> EmailMessage:
    """Compose the failure notice."""
    html = (
        build_environment()
        .get_template("email_failure.html.j2")
        .render(
            run_date=options.run_date,
            generated_at=datetime.now(ZoneInfo(settings.config.meta.timezone)),
            error=error,
            sections={s.title: s.status.value for s in report.sections},
            warnings=report.warnings,
        )
    )
    return EmailMessage(
        subject=f"FAILED — {settings.config.meta.title}, {options.run_date:%d %b %Y}",
        html=html,
    )


def _attachment_payload(path: Path, report: RunReport) -> dict[str, str] | None:
    """Base64-encode a file for Resend, or warn and skip it if too large."""
    if not path.exists():
        report.warn(f"Attachment {path} does not exist; sending without it.")
        return None

    size = path.stat().st_size
    if size > MAX_ATTACHMENT_BYTES:
        report.warn(
            f"Attachment {path.name} is {size / 1e6:.1f}MB, above the "
            f"{MAX_ATTACHMENT_BYTES / 1e6:.0f}MB limit; sending without it."
        )
        return None

    return {
        "filename": path.name,
        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
        "content_type": "application/pdf",
    }


def send(message: EmailMessage, settings: Settings, report: RunReport) -> str:
    """Send a composed message through Resend.

    Args:
        message: The composed message.
        settings: Configuration and credentials.
        report: Run report, for warnings.

    Returns:
        The Resend message id.

    Raises:
        EmailDeliveryError: If credentials are missing or Resend rejects the
            message. The caller exits non-zero on this, so that a delivery
            failure surfaces as a failed workflow rather than passing silently.
    """
    import resend

    try:
        settings.secrets.require("resend_api_key", "email_to")
    except MissingSecretError as exc:
        raise EmailDeliveryError(str(exc)) from exc

    resend.api_key = settings.secrets.reveal("resend_api_key")
    recipient = settings.secrets.reveal("email_to")

    params: dict[str, Any] = {
        "from": settings.config.email.from_address,
        "to": [recipient],
        "subject": message.subject,
        "html": message.html,
    }
    if message.attachment is not None:
        payload = _attachment_payload(message.attachment, report)
        if payload:
            params["attachments"] = [payload]

    try:
        response = resend.Emails.send(params)
    except Exception as exc:  # noqa: BLE001 — re-raised as a typed error below
        raise EmailDeliveryError(f"Resend rejected the message: {exc}") from exc

    message_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", "")
    logger.info(
        "Email sent to %s (id %s)%s",
        _mask(recipient),
        message_id,
        " with PDF attached" if "attachments" in params else "",
    )
    return str(message_id)


def _mask(address: str) -> str:
    """Partially mask an email address for logging."""
    name, _, domain = address.partition("@")
    visible = name[:2] if len(name) > 2 else name[:1]
    return f"{visible}{'*' * max(1, len(name) - len(visible))}@{domain}"


def deliver(
    sections: list[SectionResult],
    settings: Settings,
    options: RunOptions,
    report: RunReport,
    pdf_path: Path | None = None,
) -> str | None:
    """Compose and send the morning summary.

    Returns:
        The Resend message id, or None when email is disabled for this run.

    Raises:
        EmailDeliveryError: If delivery was attempted and failed.
    """
    if not options.send_email:
        logger.info("Email delivery skipped (--no-email).")
        return None

    renderable = [s for s in sections if s.status is not SectionStatus.DISABLED]
    if not any(s.is_renderable for s in renderable):
        raise EmailDeliveryError("No section produced usable output; refusing to send.")

    return send(build_message(sections, settings, options, report, pdf_path), settings, report)


def deliver_failure(
    settings: Settings, options: RunOptions, report: RunReport, error: str
) -> str | None:
    """Send the failure notice, if email is enabled for this run."""
    if not options.send_email:
        logger.warning("Run failed and email is disabled; no notice sent. Error: %s", error)
        return None
    return send(build_failure_message(settings, options, report, error), settings, report)
