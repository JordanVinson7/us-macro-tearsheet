"""Shared HTTP plumbing for every REST data source.

Three concerns live here so that no individual client has to re-solve them:

* **Rate limiting** — :class:`RateLimiter` enforces a rolling-window budget.
  Polygon's free tier allows 5 requests per 60 seconds, and exceeding it costs
  a 429 plus a backoff, so the limiter blocks pre-emptively instead.
* **Retries** — transient failures (429, 5xx, timeouts) are retried with
  exponential backoff via ``tenacity``. A ``Retry-After`` header, when the
  server sends one, always wins over the computed backoff.
* **Provenance and budget accounting** — every call is counted into the
  :class:`~tearsheet.observability.RunReport` so the footer and the run summary
  can report exactly how many calls each source received.

Request parameters are never logged. Credentials are passed as query
parameters by both FRED and Polygon, so logging a full URL would leak a key
into CI output; only the path and the parameter *names* are logged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import requests
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base, wait_exponential

from tearsheet.observability import RunReport, get_logger

logger = get_logger("data.base")

#: HTTP statuses worth retrying: rate limit, plus transient server faults.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class DataSourceError(RuntimeError):
    """Base class for data-source failures."""


class TransientApiError(DataSourceError):
    """A failure worth retrying.

    Attributes:
        retry_after: Seconds requested by the server's ``Retry-After`` header,
            if any. Overrides the computed backoff.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentApiError(DataSourceError):
    """A failure that retrying cannot fix, e.g. 400 or 404."""


class RateLimiter:
    """Blocking rolling-window rate limiter.

    Allows at most ``max_calls`` acquisitions in any ``per_seconds`` window.
    Unlike a fixed-window limiter this cannot be defeated by bunching calls
    either side of a window boundary, which is what actually triggers Polygon's
    429s in practice.
    """

    def __init__(self, max_calls: int, per_seconds: float) -> None:
        """Initialise the limiter.

        Args:
            max_calls: Maximum calls allowed per window.
            per_seconds: Window length in seconds.
        """
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Drop timestamps that have fallen out of the window."""
        cutoff = now - self.per_seconds
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def time_until_slot(self, now: float | None = None) -> float:
        """Seconds until a slot frees up. Zero when a call may proceed."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            if len(self._calls) < self.max_calls:
                return 0.0
            return max(0.0, self._calls[0] + self.per_seconds - now)

    def acquire(self) -> float:
        """Block until a slot is available, then consume it.

        Returns:
            Seconds spent waiting, for logging and the run summary.
        """
        waited = 0.0
        while True:
            delay = self.time_until_slot()
            if delay <= 0:
                break
            logger.debug("Rate limit reached; sleeping %.1fs", delay)
            time.sleep(delay)
            waited += delay
        with self._lock:
            self._calls.append(time.monotonic())
        return waited


class _WaitWithRetryAfter(wait_base):
    """Exponential backoff that defers to a server-supplied ``Retry-After``."""

    def __init__(self, fallback: wait_base) -> None:
        self._fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        """Return the number of seconds to wait before the next attempt."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, TransientApiError) and exc.retry_after:
            return exc.retry_after
        return self._fallback(retry_state)


class ApiClient:
    """A rate-limited, retrying JSON HTTP client for one data source."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        report: RunReport,
        timeout: float = 20.0,
        max_retries: int = 4,
        rate_limiter: RateLimiter | None = None,
        call_budget: int | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            name: Source name used in logs, call counts and provenance.
            base_url: Root URL; request paths are appended to it.
            report: Run report that call counts are accumulated into.
            timeout: Per-request timeout in seconds.
            max_retries: Total attempts before giving up.
            rate_limiter: Optional limiter applied before every request.
            call_budget: Soft cap on calls per run. Exceeding it warns but
                does not block, so a budget mistake degrades rather than fails.
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.report = report
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter
        self.call_budget = call_budget

        self._session = requests.Session()
        self._session.headers["User-Agent"] = "tearsheet/0.1 (+https://github.com)"
        self._budget_warned = False
        self.wait_seconds = 0.0
        """Cumulative time spent blocked by the rate limiter."""

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> ApiClient:
        """Support use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session on exit."""
        self.close()

    # -- internals -----------------------------------------------------------

    def _check_budget(self) -> None:
        """Warn once when the configured call budget is exceeded."""
        if self.call_budget is None or self._budget_warned:
            return
        if self.report.api_calls[self.name] >= self.call_budget:
            self._budget_warned = True
            self.report.warn(
                f"{self.name}: call budget of {self.call_budget} reached; "
                "further calls will still be attempted."
            )

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Perform one attempt, translating HTTP failures into exceptions."""
        if self.rate_limiter is not None:
            self.wait_seconds += self.rate_limiter.acquire()

        url = f"{self.base_url}/{path.lstrip('/')}"
        # Parameter *values* are never logged: both FRED and Polygon carry the
        # API key in the query string.
        logger.debug("GET %s params=%s", url, sorted(params))

        self.report.record_api_call(self.name)
        self._check_budget()

        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
        except requests.Timeout as exc:
            raise TransientApiError(f"{self.name}: request to {path} timed out") from exc
        except requests.RequestException as exc:
            raise TransientApiError(f"{self.name}: request to {path} failed: {exc}") from exc

        if response.status_code in RETRYABLE_STATUSES:
            retry_after = response.headers.get("Retry-After")
            raise TransientApiError(
                f"{self.name}: HTTP {response.status_code} from {path}",
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        if not response.ok:
            raise PermanentApiError(
                f"{self.name}: HTTP {response.status_code} from {path}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentApiError(f"{self.name}: {path} returned non-JSON content") from exc
        if not isinstance(payload, dict):
            raise PermanentApiError(f"{self.name}: {path} returned {type(payload).__name__}")
        return payload

    # -- public API ----------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a JSON endpoint, retrying transient failures.

        Args:
            path: Path relative to ``base_url``.
            params: Query parameters, including any credential.

        Returns:
            The decoded JSON object.

        Raises:
            TransientApiError: If every retry was exhausted.
            PermanentApiError: On a non-retryable response.
        """
        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=_WaitWithRetryAfter(wait_exponential(multiplier=1, min=2, max=60)),
            retry=retry_if_exception_type(TransientApiError),
            reraise=True,
            before_sleep=lambda state: logger.warning(
                "%s: attempt %d failed (%s); retrying",
                self.name,
                state.attempt_number,
                state.outcome.exception() if state.outcome else "unknown",
            ),
        )
        return retryer(self._request, path, params or {})
