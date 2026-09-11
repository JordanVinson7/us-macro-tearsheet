"""Section 10 — market headlines.

Only headline, publisher, time and link are shown. Article text is never
reproduced: the sheet links out to the original.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tearsheet.config import Settings
from tearsheet.models import RunOptions, SectionResult, SectionStatus
from tearsheet.observability import RunReport

if TYPE_CHECKING:
    from tearsheet.data.bundle import DataBundle


def build(
    bundle: DataBundle, settings: Settings, options: RunOptions, report: RunReport
) -> SectionResult:
    """Build the headlines section."""
    config = settings.config.sources.news
    result = SectionResult(key="headlines", title="Headlines")

    if not bundle.headlines:
        result.status = SectionStatus.UNAVAILABLE
        result.error = "No headlines could be collected."
        return result

    result.context = {
        "headlines": [
            {
                "title": item.title,
                "publisher": item.publisher,
                "url": item.url,
                "time": item.published_at.strftime("%d %b, %H:%M ET")
                if item.published_at
                else None,
                "tickers": list(item.tickers),
            }
            for item in bundle.headlines
        ]
    }
    result.sources = list(dict.fromkeys(i.source for i in bundle.headlines if i.source))
    result.notes = [
        "Headlines only — no article text is reproduced. Links open the original.",
        "Near-duplicate headlines from syndicated wire stories are collapsed.",
    ]
    result.status = (
        SectionStatus.OK if len(bundle.headlines) >= config.min_headlines else SectionStatus.PARTIAL
    )
    return result
