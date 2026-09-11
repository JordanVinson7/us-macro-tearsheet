"""Market headlines.

Only headline, publisher, timestamp and link are ever stored. Article text is
never copied, summarised or republished — the tear-sheet links out instead.

Headlines are gathered from up to three sources and merged:

1. Polygon's news endpoint, when the plan allows it.
2. yfinance news for a few broad-market ETFs.
3. Optional RSS feeds, such as Federal Reserve press releases.

Wire stories are frequently syndicated, so near-identical headlines arrive from
several publishers at once. :func:`deduplicate` collapses them using token
overlap rather than exact matching, which is what actually catches
"Fed holds rates steady" against "Fed Holds Rates Steady as Expected".
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf

from tearsheet.config import NewsSourceConfig
from tearsheet.data.polygon import PolygonClient
from tearsheet.models import NewsItem, SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.news")

YF_NEWS_SOURCE = SourceRef(name="yfinance", detail="Yahoo Finance news")
RSS_SOURCE = SourceRef(name="RSS", detail="publisher feeds")

#: Two headlines counted as duplicates above this token-overlap ratio.
SIMILARITY_THRESHOLD = 0.7

#: Words too common in financial headlines to carry any distinguishing signal.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "after",
        "amid",
        "says",
        "say",
        "new",
        "us",
        "u.s.",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenise(title: str) -> frozenset[str]:
    """Reduce a headline to its distinguishing lowercase tokens."""
    return frozenset(_TOKEN_PATTERN.findall(title.lower())) - _STOPWORDS


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap between two token sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Collapse syndicated duplicates, keeping the earliest version of each.

    Args:
        items: Headlines, in any order.

    Returns:
        Headlines with near-duplicates removed, newest first.
    """
    ordered = sorted(
        items,
        key=lambda i: i.published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC")),
        reverse=True,
    )
    kept: list[NewsItem] = []
    seen: list[frozenset[str]] = []

    for item in ordered:
        tokens = _tokenise(item.title)
        if any(_similarity(tokens, other) >= SIMILARITY_THRESHOLD for other in seen):
            continue
        kept.append(item)
        seen.append(tokens)

    return kept


def _from_yfinance(tickers: list[str], report: RunReport, timezone: str) -> list[NewsItem]:
    """Collect headlines from Yahoo for a few broad-market ETFs."""
    items: list[NewsItem] = []
    tz = ZoneInfo(timezone)

    for ticker in tickers:
        try:
            report.record_api_call("yfinance")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stories = yf.Ticker(ticker).news or []
        except Exception as exc:  # noqa: BLE001 — headlines are never critical
            logger.debug("Yahoo news for %s failed: %s", ticker, exc)
            continue

        for story in stories:
            # yfinance nests everything under "content" in current versions.
            content = story.get("content", story)
            title = (content.get("title") or "").strip()
            if not title:
                continue
            url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get(
                "url", ""
            )
            items.append(
                NewsItem(
                    title=title,
                    publisher=(content.get("provider") or {}).get("displayName", "Yahoo Finance"),
                    url=url,
                    published_at=_parse_iso(content.get("pubDate"), tz),
                    tickers=(ticker,),
                    source=YF_NEWS_SOURCE,
                )
            )

    return items


def _from_rss(config: NewsSourceConfig, report: RunReport, timezone: str) -> list[NewsItem]:
    """Collect headlines from the configured RSS feeds."""
    import feedparser

    items: list[NewsItem] = []
    tz = ZoneInfo(timezone)

    for feed_config in config.rss_feeds:
        try:
            report.record_api_call("rss")
            feed = feedparser.parse(feed_config.url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("RSS feed %s failed: %s", feed_config.name, exc)
            continue

        for entry in feed.entries[: config.max_headlines]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            published = None
            if parsed := entry.get("published_parsed"):
                published = datetime(*parsed[:6], tzinfo=ZoneInfo("UTC")).astimezone(tz)
            items.append(
                NewsItem(
                    title=title,
                    publisher=feed_config.name,
                    url=entry.get("link", ""),
                    published_at=published,
                    source=RSS_SOURCE,
                )
            )

    return items


def collect_headlines(
    config: NewsSourceConfig,
    report: RunReport,
    *,
    polygon: PolygonClient | None = None,
    timezone: str = "America/New_York",
) -> list[NewsItem]:
    """Gather, deduplicate and trim market headlines.

    Args:
        config: News settings.
        report: Run report for call accounting.
        polygon: Polygon client, if its news endpoint should be used.
        timezone: Timezone for the returned timestamps.

    Returns:
        Up to ``max_headlines`` deduplicated headlines, newest first.
    """
    items: list[NewsItem] = []

    if polygon is not None:
        items.extend(polygon.fetch_news(limit=config.max_headlines * 2, timezone=timezone))

    items.extend(_from_yfinance(config.yfinance_tickers, report, timezone))
    items.extend(_from_rss(config, report, timezone))

    deduplicated = deduplicate(items)[: config.max_headlines]

    if len(deduplicated) < config.min_headlines:
        report.warn(
            f"Only {len(deduplicated)} headline(s) available "
            f"(wanted at least {config.min_headlines})."
        )

    logger.info("Headlines: %d collected, %d after deduplication", len(items), len(deduplicated))
    return deduplicated


def _parse_iso(value: str | None, tz: ZoneInfo) -> datetime | None:
    """Parse an ISO-8601 timestamp into the report timezone."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(tz)
    except ValueError:
        return None
