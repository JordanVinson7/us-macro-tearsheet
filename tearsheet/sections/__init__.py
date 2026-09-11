"""Section builders, one module per tear-sheet section.

Each module exposes ``build(bundle, settings, options, report) -> SectionResult``.
The registry below is what :mod:`tearsheet.pipeline` walks; a section with no
entry renders as PENDING, and a builder that raises is caught at the pipeline
boundary and rendered as "data unavailable".
"""

from __future__ import annotations

from tearsheet.sections import (
    calendar,
    credit,
    cross_asset,
    equities,
    fx,
    header,
    headlines,
    macro,
    overnight,
    rates,
)

BUILDERS = {
    "header": header.build,
    "calendar": calendar.build,
    "overnight": overnight.build,
    "equities": equities.build,
    "rates": rates.build,
    "credit": credit.build,
    "macro": macro.build,
    "fx_commodities": fx.build,
    "cross_asset": cross_asset.build,
    "headlines": headlines.build,
}
"""Section key to builder function, in page order."""

__all__ = ["BUILDERS"]
