"""Loading and validation of config.yaml and .env.

Two deliberate choices here:

1. **The whole of config.yaml is modelled with Pydantic, with ``extra="forbid"``.**
   A typo in a config key raises at startup with the exact path, instead of
   silently disabling a feature that then quietly goes missing from the page.
2. **Secrets are Pydantic ``SecretStr`` and never returned as plain text by
   any display helper.** :meth:`Secrets.status` reports only set/missing/
   placeholder, so a config dump can be printed in CI logs safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from tearsheet.observability import REDACTOR

Transform = Literal["level", "diff", "mom_pct", "yoy_pct", "diff_3m_avg", "avg_4w"]
"""How a raw FRED series is converted into its display value."""

Frequency = Literal["daily", "weekly", "monthly", "quarterly"]
"""Expected update cadence, used for the staleness badge."""

Tone = Literal["positive", "neutral", "negative"]

#: Marks values in .env that are still the .env.example placeholders.
_PLACEHOLDER_MARKERS = ("your_", "_here", "your-github-username", "you@example.com")


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed or internally inconsistent."""


class MissingSecretError(ConfigError):
    """Raised when a required credential is absent or still a placeholder."""


class _Model(BaseModel):
    """Base for every config model: immutable, and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class Instrument(_Model):
    """A tradable instrument quoted through yfinance.

    Attributes:
        ticker: Yahoo Finance symbol.
        name: Display name.
        short: Abbreviation used in dense tables and the KPI strip.
        cash: For futures, the cash index to compare against.
        invert: Quote the pair the other way round for display.
        optional: Do not warn if this symbol is unavailable.
    """

    ticker: str
    name: str
    short: str | None = None
    cash: str | None = None
    invert: bool = False
    optional: bool = False

    @property
    def label(self) -> str:
        """Short label if one is configured, otherwise the full name."""
        return self.short or self.name


class FredSeries(_Model):
    """One FRED series and how to present it.

    Attributes:
        id: FRED series ID, validated against ``fred/series`` at startup.
        name: Display name.
        short: Abbreviation for dense tables.
        transform: Conversion applied before display.
        units: Unit label shown next to the value.
        frequency: Expected cadence, used for staleness checks.
        scale: Multiplier applied to raw values, e.g. 0.001 to show thousands.
        yoy_units: How to express year-on-year change. ``auto`` infers from
            units and history; set explicitly for series that cross zero.
        tenor_years: For curve points, the maturity in years.
        proxy_for: Set when this series stands in for an unavailable indicator;
            surfaced on the page so the substitution is never hidden.
    """

    id: str
    name: str
    short: str | None = None
    transform: Transform = "level"
    units: str = "%"
    frequency: Frequency = "daily"
    scale: float = 1.0
    yoy_units: Literal["auto", "%", "pp", "pts", "none"] = "auto"
    tenor_years: float | None = None
    proxy_for: str | None = None

    @property
    def label(self) -> str:
        """Short label if one is configured, otherwise the full name."""
        return self.short or self.name


# ---------------------------------------------------------------------------
# Top-level blocks
# ---------------------------------------------------------------------------


class MetaConfig(_Model):
    """Document identity and display timezone."""

    title: str
    subtitle: str
    timezone: str
    author: str


class ScheduleConfig(_Model):
    """The guard that decides whether a given invocation publishes a report."""

    target_time: time
    window_start: time
    window_end: time
    weekdays_only: bool
    skip_market_holidays: bool
    exchange_calendar: str

    @field_validator("target_time", "window_start", "window_end", mode="before")
    @classmethod
    def _parse_time(cls, value: Any) -> Any:
        """Accept ``"HH:MM"`` from YAML as well as a real time object."""
        if isinstance(value, str):
            hours, _, minutes = value.partition(":")
            return time(int(hours), int(minutes))
        return value


class OutputConfig(_Model):
    """Where artifacts are written and how long the archive is kept."""

    docs_dir: str
    local_dir: str
    archive_dir: str
    retain_days: int = Field(ge=0)
    pdf_filename: str


class RateLimitConfig(_Model):
    """Rolling-window rate limit for an API."""

    max_calls: int = Field(gt=0)
    per_seconds: float = Field(gt=0)


class FredSourceConfig(_Model):
    """FRED API client settings."""

    base_url: str
    timeout_seconds: float
    max_retries: int
    validate_series_on_startup: bool


class PolygonSourceConfig(_Model):
    """Polygon/Massive API client settings, including the free-tier limiter."""

    base_url: str
    timeout_seconds: float
    max_retries: int
    rate_limit: RateLimitConfig
    call_budget: int
    news_enabled: bool


class YFinanceSourceConfig(_Model):
    """yfinance batching, retries and the ordered fallback chain."""

    batch_size: int = Field(gt=0)
    max_retries: int
    history_days: int
    fallback_chain: list[str]


class RssFeed(_Model):
    """A headline RSS source."""

    name: str
    url: str


class NewsSourceConfig(_Model):
    """Headline collection settings."""

    max_headlines: int
    min_headlines: int
    yfinance_tickers: list[str]
    rss_feeds: list[RssFeed]


class SourcesConfig(_Model):
    """All data-source client settings."""

    fred: FredSourceConfig
    polygon: PolygonSourceConfig
    yfinance: YFinanceSourceConfig
    news: NewsSourceConfig


class IntradayConfig(_Model):
    """Tickers fetched as previous-session minute bars from Polygon."""

    tickers: list[str]
    bar_size: str


class RatioSpec(_Model):
    """A relative-performance ratio chart."""

    numerator: str
    denominator: str
    name: str

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``RSP/SPY``."""
        return f"{self.numerator}/{self.denominator}"


class UniverseConfig(_Model):
    """Every instrument the tear-sheet covers."""

    indices: list[Instrument]
    sectors: list[Instrument]
    futures: list[Instrument]
    volatility: list[Instrument]
    credit_etfs: list[Instrument]
    fx: list[Instrument]
    commodities: list[Instrument]
    crypto: list[Instrument]
    intraday: IntradayConfig
    ratios: list[RatioSpec]
    support_tickers: list[str]

    @property
    def instrument_groups(self) -> dict[str, list[Instrument]]:
        """Named groups of instruments, in page order."""
        return {
            "indices": self.indices,
            "sectors": self.sectors,
            "futures": self.futures,
            "volatility": self.volatility,
            "credit_etfs": self.credit_etfs,
            "fx": self.fx,
            "commodities": self.commodities,
            "crypto": self.crypto,
        }

    def all_instruments(self) -> list[Instrument]:
        """Flatten every instrument group, preserving order."""
        return [inst for group in self.instrument_groups.values() for inst in group]

    def all_yf_tickers(self) -> list[str]:
        """Every Yahoo symbol needed for a run, de-duplicated and ordered.

        Includes ratio legs and support tickers, which have no display row of
        their own but are required by charts, breadth and cross-asset.
        """
        tickers = [inst.ticker for inst in self.all_instruments()]
        for ratio in self.ratios:
            tickers.extend([ratio.numerator, ratio.denominator])
        tickers.extend(self.support_tickers)
        return list(dict.fromkeys(tickers))

    def find(self, ticker: str) -> Instrument | None:
        """Look up an instrument by Yahoo symbol."""
        return next((i for i in self.all_instruments() if i.ticker == ticker), None)


class FredConfig(_Model):
    """Every FRED series, grouped by the section that consumes it."""

    treasury_curve: list[FredSeries]
    rates: list[FredSeries]
    credit: list[FredSeries]
    macro: list[FredSeries]
    fallback_levels: dict[str, str]

    @property
    def series_groups(self) -> dict[str, list[FredSeries]]:
        """Named groups of FRED series."""
        return {
            "treasury_curve": self.treasury_curve,
            "rates": self.rates,
            "credit": self.credit,
            "macro": self.macro,
        }

    def all_series(self) -> list[FredSeries]:
        """Flatten every FRED series group, preserving order."""
        return [s for group in self.series_groups.values() for s in group]

    def all_series_ids(self) -> list[str]:
        """Every configured FRED series ID, de-duplicated."""
        ids = [s.id for s in self.all_series()]
        ids.extend(self.fallback_levels.values())
        return list(dict.fromkeys(ids))

    def find(self, series_id: str) -> FredSeries | None:
        """Look up a series definition by ID."""
        return next((s for s in self.all_series() if s.id == series_id), None)


class FomcMeeting(_Model):
    """One FOMC decision day.

    Attributes:
        date: The decision day — the second day of a two-day meeting.
        sep: Whether a Summary of Economic Projections accompanies it.
    """

    date: date
    sep: bool


class CalendarConfig(_Model):
    """Economic release calendar and FOMC dates."""

    lookahead_days: int
    release_whitelist: list[str]
    typical_times_et: dict[str, str]
    fomc_meetings: list[FomcMeeting]
    fomc_note: str
    next_release_horizon_days: int

    def next_fomc(self, on_or_after: date) -> FomcMeeting | None:
        """The next scheduled FOMC decision on or after a given date."""
        upcoming = sorted(m.date for m in self.fomc_meetings if m.date >= on_or_after)
        if not upcoming:
            return None
        return next(m for m in self.fomc_meetings if m.date == upcoming[0])


class EarningsConfig(_Model):
    """Notable-earnings watchlist and call budget."""

    enabled: bool
    market_cap_threshold_usd: int
    max_calls: int
    watchlist: list[str]


class AnalyticsConfig(_Model):
    """Windows and thresholds shared by every analytics function."""

    return_windows: dict[str, int]
    realised_vol_window: int
    annualisation_factor: int
    zscore_lookback: int
    correlation_window: int
    percentile_lookback: int
    moving_averages: list[int]
    min_observations: int


class FlagsConfig(_Model):
    """Thresholds for the rule-based 'What stood out' panel."""

    return_zscore: float
    high_low_window: int
    ma_cross_tolerance_pct: float
    vix_backwardation_ratio: float
    credit_spread_move_bp: float
    curve_inversion_tolerance_bp: float
    max_flags: int


class CurveConfig(_Model):
    """Inputs to the bull/bear steepener/flattener classification."""

    classification_threshold_bp: float
    short_tenor: str
    long_tenor: str
    comparison_windows: list[str]
    history_windows_ago: list[str]


class RegimeComponent(_Model):
    """One z-scored input to the risk regime composite.

    Attributes:
        weight: Relative weight before re-normalisation.
        sign: ``+1`` if a higher value means more risk-off, ``-1`` otherwise.
        source: Key of the underlying series or derived measure.
        label: Display name in the contribution chart.
    """

    weight: float = Field(gt=0)
    sign: Literal[-1, 1]
    source: str
    label: str


class RegimeBand(_Model):
    """A labelled range of composite values."""

    max: float
    label: str
    tone: Tone


class RegimeConfig(_Model):
    """Risk regime composite definition.

    This is a descriptive summary of conditions, not a trading signal, and is
    labelled as such wherever it appears.
    """

    lookback: int
    label: str
    components: dict[str, RegimeComponent]
    bands: list[RegimeBand]

    def normalised_weights(self, available: set[str] | None = None) -> dict[str, float]:
        """Weights re-normalised over the components that actually have data.

        Args:
            available: Component keys with usable data. ``None`` means all.

        Returns:
            Component key to weight, summing to 1.0.

        Raises:
            ConfigError: If no components are available.
        """
        keys = set(self.components) if available is None else set(available) & set(self.components)
        if not keys:
            raise ConfigError("Risk regime composite has no available components.")
        total = sum(self.components[k].weight for k in keys)
        return {k: self.components[k].weight / total for k in sorted(keys)}

    def band_for(self, value: float) -> RegimeBand:
        """Return the band a composite value falls into."""
        for band in self.bands:
            if value <= band.max:
                return band
        return self.bands[-1]


class CrossAssetMember(_Model):
    """One row/column of the cross-asset correlation matrix."""

    key: str
    label: str
    is_spread: bool = False


class CrossAssetConfig(_Model):
    """Cross-asset correlation matrix membership."""

    members: list[CrossAssetMember]


class QualityConfig(_Model):
    """Sanity and staleness checks applied to every series."""

    staleness_grace_days: dict[Frequency, int]
    max_plausible_daily_move_pct: dict[str, float]
    reject_non_positive_prices: bool
    max_price_age_days: int

    def grace_days(self, frequency: Frequency) -> int:
        """Grace period in calendar days for a given cadence."""
        return self.staleness_grace_days[frequency]


class DisplayConfig(_Model):
    """Formatting and layout choices."""

    kpi_strip: list[str]
    decimals: dict[str, int]
    heatmap_windows: list[str]
    small_multiples_years: int
    disclaimer: str


class EmailConfig(_Model):
    """Email composition settings. The API key itself lives in .env."""

    from_address: str
    subject_template: str
    include_sections: list[str]


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class Config(_Model):
    """The fully validated contents of config.yaml."""

    meta: MetaConfig
    schedule: ScheduleConfig
    output: OutputConfig
    sections: dict[str, bool]
    sources: SourcesConfig
    universe: UniverseConfig
    fred: FredConfig
    calendar: CalendarConfig
    earnings: EarningsConfig
    analytics: AnalyticsConfig
    flags: FlagsConfig
    curve: CurveConfig
    regime: RegimeConfig
    cross_asset: CrossAssetConfig
    quality: QualityConfig
    display: DisplayConfig
    email: EmailConfig

    def section_enabled(self, key: str) -> bool:
        """Whether a section is switched on. Unknown sections default to off."""
        return self.sections.get(key, False)

    @property
    def enabled_sections(self) -> list[str]:
        """Section keys that are switched on, in config order."""
        return [key for key, on in self.sections.items() if on]

    @classmethod
    def load(cls, path: Path) -> Config:
        """Parse and validate config.yaml.

        Args:
            path: Path to the YAML file.

        Returns:
            The validated configuration.

        Raises:
            ConfigError: If the file is missing, unparseable or fails validation.
        """
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            details = "\n".join(
                f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise ConfigError(f"{path} failed validation:\n{details}") from exc


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def _is_placeholder(value: str) -> bool:
    """True if a value still looks like the .env.example placeholder text."""
    lowered = value.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


class Secrets(BaseModel):
    """Credentials loaded from the environment.

    Values are held as ``SecretStr`` so they cannot be printed by accident:
    ``repr`` and ``str`` both render ``**********``. Use
    :meth:`Secrets.reveal` at the single point of use — the HTTP client — and
    nowhere else.
    """

    model_config = ConfigDict(frozen=True)

    polygon_api_key: Annotated[SecretStr | None, Field(default=None)]
    fred_api_key: Annotated[SecretStr | None, Field(default=None)]
    resend_api_key: Annotated[SecretStr | None, Field(default=None)]
    email_to: Annotated[str | None, Field(default=None)]
    site_url: Annotated[str | None, Field(default=None)]

    #: Env var name for each field.
    ENV_NAMES: ClassVar[dict[str, str]] = {
        "polygon_api_key": "POLYGON_API_KEY",
        "fred_api_key": "FRED_API_KEY",
        "resend_api_key": "RESEND_API_KEY",
        "email_to": "EMAIL_TO",
        "site_url": "SITE_URL",
    }

    @classmethod
    def from_env(cls) -> Secrets:
        """Read credentials from ``os.environ``, treating placeholders as unset."""
        values: dict[str, Any] = {}
        for field_name, env_name in cls.ENV_NAMES.items():
            raw = os.environ.get(env_name, "").strip()
            values[field_name] = None if not raw or _is_placeholder(raw) else raw
        return cls(**values)

    def reveal(self, field_name: str) -> str:
        """Return the plain-text value of one credential.

        This is the only sanctioned way to read a secret. Call it at the point
        of use and never assign the result to anything long-lived.

        Args:
            field_name: One of the keys in :attr:`ENV_NAMES`.

        Returns:
            The credential as plain text.

        Raises:
            MissingSecretError: If the credential is absent or a placeholder.
        """
        value = getattr(self, field_name, None)
        if value is None:
            raise MissingSecretError(
                f"{self.ENV_NAMES.get(field_name, field_name)} is not set. "
                "Add it to .env (see .env.example)."
            )
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)

    def has(self, field_name: str) -> bool:
        """Whether a credential is present and not a placeholder."""
        return getattr(self, field_name, None) is not None

    def require(self, *field_names: str) -> None:
        """Assert that the named credentials are usable.

        Args:
            *field_names: Field names to check.

        Raises:
            MissingSecretError: Listing every missing credential at once, so a
                first run reports all the gaps rather than one per attempt.
        """
        missing = [self.ENV_NAMES[name] for name in field_names if not self.has(name)]
        if missing:
            raise MissingSecretError(
                "Missing or placeholder credentials: "
                + ", ".join(missing)
                + ". Add them to .env (see .env.example)."
            )

    def status(self) -> dict[str, str]:
        """Presence report for display. Never includes any credential value."""
        report = {}
        for field_name, env_name in self.ENV_NAMES.items():
            raw = os.environ.get(env_name, "").strip()
            if self.has(field_name):
                report[env_name] = "set"
            elif raw and _is_placeholder(raw):
                report[env_name] = "placeholder"
            else:
                report[env_name] = "missing"
        return report

    def register_for_redaction(self) -> None:
        """Teach the log redactor every secret value held here."""
        for field_name in self.ENV_NAMES:
            value = getattr(self, field_name, None)
            if isinstance(value, SecretStr):
                REDACTOR.register(value.get_secret_value())


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything a run needs to know that is not a command-line option.

    Attributes:
        config: Validated config.yaml.
        secrets: Credentials from the environment.
        config_path: Where the config was loaded from, for the run summary.
        project_root: Directory containing config.yaml.
        env_path: The .env that was loaded, or None when credentials came from
            ambient environment variables (as they do in CI).
    """

    config: Config
    secrets: Secrets
    config_path: Path
    project_root: Path
    env_path: Path | None = None


def load_settings(config_path: Path | None = None, env_path: Path | None = None) -> Settings:
    """Load config.yaml and .env into a validated :class:`Settings`.

    Args:
        config_path: Path to config.yaml. Defaults to the project root.
        env_path: Path to .env. Defaults to the project root. A missing .env is
            not an error: in CI the values come from real environment variables.

    Returns:
        The loaded settings, with secrets registered for log redaction.

    Raises:
        ConfigError: If config.yaml is missing or invalid.
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = config_path or project_root / "config.yaml"
    env_path = env_path or project_root / ".env"

    env_loaded = env_path.exists()
    if env_loaded:
        from dotenv import load_dotenv

        # override=False: a real environment variable (as set by GitHub Actions)
        # always wins over a stale local .env.
        load_dotenv(env_path, override=False)

    config = Config.load(config_path)
    secrets = Secrets.from_env()
    secrets.register_for_redaction()

    return Settings(
        config=config,
        secrets=secrets,
        config_path=config_path,
        project_root=project_root,
        env_path=env_path if env_loaded else None,
    )
