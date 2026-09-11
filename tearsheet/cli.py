"""Command-line entry point.

Usage::

    python -m tearsheet run [--no-email] [--no-pdf] [--force] [--mock] [--open]

Local runs write to ``output/`` (gitignored) so they can never collide with
the ``docs/`` tree that CI commits. Pass ``--publish`` to write into ``docs/``,
which is what the GitHub Actions workflow does.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tearsheet import __version__
from tearsheet.config import ConfigError, load_settings
from tearsheet.models import RunMode, RunOptions
from tearsheet.observability import RunReport, get_console, get_logger, setup_logging

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_CONFIG_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tearsheet",
        description="US Macro & Markets Morning Tear-Sheet.",
    )
    parser.add_argument("--version", action="version", version=f"tearsheet {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Build and deliver the tear-sheet.")
    run_parser.add_argument("--no-email", action="store_true", help="Skip email delivery.")
    run_parser.add_argument("--no-pdf", action="store_true", help="Skip PDF rendering.")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the schedule/holiday guard and overwrite today's report.",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Render from saved fixtures instead of live APIs. Makes no network calls.",
    )
    run_parser.add_argument(
        "--open", dest="open_result", action="store_true", help="Open the result in a browser."
    )
    run_parser.add_argument(
        "--publish",
        action="store_true",
        help="Write into docs/ for GitHub Pages instead of output/. Used by CI.",
    )
    run_parser.add_argument(
        "--record",
        action="store_true",
        help="Save the data collected by this run as fixtures for --mock mode.",
    )
    run_parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml.")
    run_parser.add_argument("--out", type=Path, default=None, help="Override the output directory.")
    run_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level console logging."
    )
    return parser


def _resolve_options(args: argparse.Namespace, settings_root: Path, timezone: str) -> RunOptions:
    """Turn parsed arguments into a resolved :class:`RunOptions`."""
    mode = RunMode.MOCK if args.mock else RunMode.LIVE
    run_date = datetime.now(ZoneInfo(timezone)).date()

    if args.out is not None:
        output_dir = args.out
    elif args.publish:
        output_dir = settings_root / "docs"
    else:
        output_dir = settings_root / "output"

    return RunOptions(
        mode=mode,
        send_email=not args.no_email,
        build_pdf=not args.no_pdf,
        force=args.force,
        open_result=args.open_result,
        publish=args.publish,
        record=args.record,
        verbose=args.verbose,
        config_path=args.config or settings_root / "config.yaml",
        output_dir=output_dir,
        run_date=run_date,
    )


def _notify_failure(settings: object, options: object, report: object, summary: str) -> None:
    """Send the failure notice, without letting that failure mask the original.

    If the failure email itself cannot be sent there is nothing further to try,
    so it is logged and the original non-zero exit stands — which is what makes
    GitHub Actions surface the run.
    """
    logger = get_logger("cli")
    try:
        from tearsheet.delivery.email import deliver_failure

        deliver_failure(settings, options, report, summary)
    except Exception as exc:  # noqa: BLE001 — last resort
        logger.error("Could not send the failure notice: %s", exc)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    # Config is loaded first so that logging can be stamped in the report
    # timezone. Config errors are reported straight to the console.
    try:
        settings = load_settings(config_path=args.config)
    except ConfigError as exc:
        get_console().print(f"[bold red]Configuration error[/]\n{exc}")
        return EXIT_CONFIG_ERROR

    setup_logging(
        project_root / "logs",
        verbose=args.verbose,
        timezone=settings.config.meta.timezone,
    )
    logger = get_logger("cli")
    logger.debug("Config loaded from %s", settings.config_path)
    logger.debug(
        "Credentials from %s",
        settings.env_path or "ambient environment variables (no .env present)",
    )

    options = _resolve_options(args, settings.project_root, settings.config.meta.timezone)
    report = RunReport(mode=options.mode, timezone=settings.config.meta.timezone)

    # The guard runs before any heavy import or network call: a firing that
    # should not publish must cost nothing. Standing down is a success, not a
    # failure, so it exits 0 and leaves the workflow green.
    from tearsheet.guard import evaluate

    decision = evaluate(settings, options)
    if not decision.should_run:
        logger.info("Standing down: %s", decision.reason)
        get_console().print(
            f"[yellow]No report published.[/] {decision.reason}\n"
            "[dim]Use --force to override the schedule guard.[/]"
        )
        return EXIT_OK
    logger.info("Proceeding: %s", decision.reason)

    # Imported here so a config error is reported before the heavier imports.
    from tearsheet import pipeline

    try:
        exit_code = pipeline.run(settings, options, report)
    except Exception as exc:  # noqa: BLE001 — top-level guard; details are logged
        logger.exception("Run failed")
        summary = f"{type(exc).__name__}: {exc}"
        report.warn(f"Run failed: {summary}")
        exit_code = EXIT_RUN_FAILED
        _notify_failure(settings, options, report, summary)

    report.print_summary()

    if options.open_result and "html" in report.artifacts:
        webbrowser.open(report.artifacts["html"].resolve().as_uri())

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
