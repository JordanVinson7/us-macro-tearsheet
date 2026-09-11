"""FRED (Federal Reserve Economic Data) client.

Covers the four things the tear-sheet needs from FRED: series metadata,
observations, the list of releases, and release dates for the calendar.

Two behaviours are deliberate:

* **Series IDs are validated at startup** against ``fred/series``. An ID that
  FRED does not recognise is dropped with a warning rather than crashing the
  run, so a typo in config.yaml costs one row on the page, not the report.
* **Available history is recorded per series.** Several ICE BofA credit spread
  series start much later than the other series, so lookback windows are sized
  against what actually exists rather than against an assumed 1-year history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from tearsheet.config import FredSeries, FredSourceConfig
from tearsheet.data.base import ApiClient, DataSourceError, PermanentApiError
from tearsheet.models import SourceRef
from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.fred")

FRED_SOURCE = SourceRef(name="FRED", detail="api.stlouisfed.org")

#: FRED encodes a missing observation as a bare full stop.
_MISSING_VALUE = "."


@dataclass(frozen=True, slots=True)
class FredSeriesMeta:
    """Metadata returned by ``fred/series`` for one series.

    Attributes:
        series_id: The FRED ID.
        title: FRED's official title.
        units: FRED's units string, e.g. "Percent".
        frequency: FRED's frequency string, e.g. "Daily, Close".
        last_updated: When FRED last revised the series.
        observation_start: First available observation date.
        observation_end: Last available observation date.
    """

    series_id: str
    title: str
    units: str
    frequency: str
    last_updated: datetime | None
    observation_start: date
    observation_end: date

    @property
    def history_years(self) -> float:
        """Approximate years of available history."""
        return (self.observation_end - self.observation_start).days / 365.25


@dataclass(frozen=True, slots=True)
class ReleaseDate:
    """One scheduled release date from ``fred/releases/dates``.

    Attributes:
        release_id: FRED release ID.
        release_name: FRED's official release name.
        date: Scheduled publication date.
    """

    release_id: int
    release_name: str
    date: date


class FredClient:
    """Typed wrapper over the FRED REST API."""

    def __init__(self, config: FredSourceConfig, api_key: str, report: RunReport) -> None:
        """Initialise the client.

        Args:
            config: FRED client settings from config.yaml.
            api_key: The FRED API key. Held only for the lifetime of the client.
            report: Run report for call accounting.
        """
        self._api_key = api_key
        self._client = ApiClient(
            name="fred",
            base_url=config.base_url,
            report=report,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self.report = report

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._client.close()

    def _get(self, path: str, **params: object) -> dict:
        """Issue a GET with the API key and JSON format attached."""
        return self._client.get(path, {**params, "api_key": self._api_key, "file_type": "json"})

    # -- metadata ------------------------------------------------------------

    def fetch_metadata(self, series_id: str) -> FredSeriesMeta:
        """Fetch metadata for one series.

        Args:
            series_id: FRED series ID.

        Returns:
            The parsed metadata.

        Raises:
            PermanentApiError: If FRED does not recognise the ID.
        """
        payload = self._get("series", series_id=series_id)
        entries = payload.get("seriess") or []
        if not entries:
            raise PermanentApiError(f"fred: no metadata returned for {series_id}")
        entry = entries[0]
        return FredSeriesMeta(
            series_id=entry["id"],
            title=entry.get("title", series_id),
            units=entry.get("units", ""),
            frequency=entry.get("frequency", ""),
            last_updated=_parse_last_updated(entry.get("last_updated")),
            observation_start=date.fromisoformat(entry["observation_start"]),
            observation_end=date.fromisoformat(entry["observation_end"]),
        )

    def validate_series(
        self, series: list[FredSeries]
    ) -> tuple[dict[str, FredSeriesMeta], list[str]]:
        """Check every configured series ID against FRED.

        Args:
            series: Configured series definitions.

        Returns:
            A ``(metadata_by_id, invalid_ids)`` pair. Invalid IDs are logged as
            warnings on the run report and excluded from the returned mapping,
            so downstream code can assume anything it receives is real.
        """
        metadata: dict[str, FredSeriesMeta] = {}
        invalid: list[str] = []

        for spec in series:
            if spec.id in metadata:
                continue
            try:
                metadata[spec.id] = self.fetch_metadata(spec.id)
            except DataSourceError as exc:
                invalid.append(spec.id)
                self.report.warn(f"FRED series {spec.id!r} ({spec.name}) is unavailable: {exc}")

        logger.info("Validated %d FRED series (%d unavailable)", len(metadata), len(invalid))
        return metadata, invalid

    # -- observations --------------------------------------------------------

    def fetch_observations(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        """Fetch a series as a float-valued, date-indexed pandas Series.

        Args:
            series_id: FRED series ID.
            start: Earliest observation date to request.
            end: Latest observation date to request.

        Returns:
            Observations indexed by date, ascending, with FRED's ``"."``
            missing marker converted to ``NaN`` and dropped.
        """
        params: dict[str, object] = {"series_id": series_id, "sort_order": "asc"}
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()

        payload = self._get("series/observations", **params)
        observations = payload.get("observations") or []

        index = pd.to_datetime([obs["date"] for obs in observations])
        values = [
            float("nan") if obs["value"] == _MISSING_VALUE else float(obs["value"])
            for obs in observations
        ]
        series = pd.Series(values, index=index, name=series_id, dtype="float64")
        return series.dropna()

    def fetch_many(
        self,
        series: list[FredSeries],
        start: date,
        metadata: dict[str, FredSeriesMeta] | None = None,
    ) -> dict[str, pd.Series]:
        """Fetch observations for several series, skipping ones that fail.

        Args:
            series: Configured series definitions.
            start: Earliest observation date to request.
            metadata: Validated metadata. When supplied, the request start is
                clamped to each series' own ``observation_start`` — this is
                what keeps short-history credit spread series from silently
                producing an under-filled lookback window.

        Returns:
            Series data keyed by FRED ID. Failures are warned about and omitted.
        """
        results: dict[str, pd.Series] = {}
        for spec in series:
            if spec.id in results:
                continue
            effective_start = start
            if metadata and spec.id in metadata:
                effective_start = max(start, metadata[spec.id].observation_start)
            try:
                values = self.fetch_observations(spec.id, start=effective_start)
            except DataSourceError as exc:
                self.report.warn(f"FRED observations for {spec.id!r} failed: {exc}")
                continue
            if values.empty:
                self.report.warn(f"FRED series {spec.id!r} returned no observations.")
                continue
            results[spec.id] = values
        return results

    # -- releases ------------------------------------------------------------

    def fetch_releases(self) -> dict[str, int]:
        """Map every FRED release name to its ID.

        Release IDs are resolved by name rather than hardcoded, so the calendar
        whitelist in config.yaml stays readable and cannot drift out of date.
        """
        payload = self._get("releases", limit=1000)
        return {r["name"]: int(r["id"]) for r in payload.get("releases", [])}

    def fetch_series_release(self, series_id: str) -> tuple[int, str] | None:
        """Find which FRED release publishes a series.

        Used by the macro dashboard to show each indicator's next scheduled
        release date. Looked up rather than hardcoded so the mapping cannot
        drift out of step with FRED.

        Args:
            series_id: FRED series ID.

        Returns:
            An ``(id, name)`` pair, or None if the lookup failed.
        """
        try:
            payload = self._get("series/release", series_id=series_id)
        except DataSourceError as exc:
            logger.debug("Release lookup for %s failed: %s", series_id, exc)
            return None
        releases = payload.get("releases") or []
        if not releases:
            return None
        return int(releases[0]["id"]), releases[0].get("name", "")

    def fetch_release_dates(
        self,
        release_id: int,
        start: date,
        end: date,
    ) -> list[ReleaseDate]:
        """Fetch scheduled dates for ONE release.

        Querying per release rather than asking for every FRED release at once
        matters: the unfiltered ``releases/dates`` query spans 330+ releases and
        reliably times out, whereas each per-release query returns a handful of
        rows in well under a second.

        ``include_release_dates_with_no_data=true`` is essential — without it
        FRED returns only dates that have already published, which is the
        opposite of what a forward-looking calendar needs.

        Args:
            release_id: FRED release ID.
            start: Window start, inclusive.
            end: Window end, inclusive.

        Returns:
            Matching release dates, ascending. Empty on failure.
        """
        try:
            payload = self._get(
                "release/dates",
                release_id=release_id,
                realtime_start=start.isoformat(),
                realtime_end=end.isoformat(),
                include_release_dates_with_no_data="true",
                sort_order="asc",
                limit=1000,
            )
        except DataSourceError as exc:
            logger.debug("Release dates for %s failed: %s", release_id, exc)
            return []

        dates = [
            ReleaseDate(
                release_id=int(entry.get("release_id", release_id)),
                release_name=entry.get("release_name", ""),
                date=date.fromisoformat(entry["date"]),
            )
            for entry in payload.get("release_dates", [])
        ]
        return [d for d in dates if start <= d.date <= end]


def _parse_last_updated(value: str | None) -> datetime | None:
    """Parse FRED's ``last_updated`` stamp, which carries a ``-05`` style offset."""
    if not value:
        return None
    # FRED emits e.g. "2026-09-10 15:16:35-05"; Python needs "-05:00".
    text = value.strip()
    if len(text) > 3 and text[-3] in "+-" and text[-2:].isdigit():
        text = f"{text}:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.debug("Could not parse FRED last_updated %r", value)
        return None
