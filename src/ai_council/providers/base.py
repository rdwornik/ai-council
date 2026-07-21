"""Abstract base for all AI model providers.

Template-method design (A1): the base owns the invariant `generate()` skeleton — timing,
the `timeout_sec` guard, the two error wrappers, the empty-content check, logging, and the
`ModelResponse` construction. A concrete provider implements only its three genuinely
divergent hooks: `_configure` (build the SDK client), `_invoke` (the SDK call), `_parse`
(raw SDK response -> `_Parsed`). Providers remain separate classes in separate files — this
is a shared base, NOT a provider merge (CLAUDE.md 5.7 / ADR-12 no-merge rule).
"""

import asyncio
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ai_council.models import ModelResponse
from config.config_loader import ModelConfig

logger = logging.getLogger(__name__)

# Error categories returned by classify_error()
RETRYABLE_ERRORS: frozenset[str] = frozenset(
    {"timeout", "rate_limit", "connection_error", "server_error"}
)
NON_RETRYABLE_ERRORS: frozenset[str] = frozenset(
    {"auth", "model_not_found", "content_policy", "invalid_request", "billing"}
)

# CLI-backend fallback causes — the single shared source for both classify_cli_failure()
# (built in the CLI-seats arc) and the seats[].fallback_events[].cause sidecar field
# (L-CLI IF#4: "one source, not two lists to drift"). Deliberately SEPARATE from
# classify_error's API vocabulary above: ADR-12 §4's "extend classify_error's categories"
# is honored by this shared constant, not by widening classify_error itself — healthcheck.py
# depends on classify_error's API contract (do-not-touch) and subprocess-failure
# classification is a genuinely different mechanism from API-exception classification.
CLI_FALLBACK_CAUSES: frozenset[str] = frozenset(
    {"quota", "timeout", "parse", "identity-unreadable", "process-error"}
)


# HTTP status -> category, for the typed dispatch below. Any other 5xx falls through to
# "server_error"; anything unmapped falls through to class-name dispatch then the string arms.
_STATUS_CATEGORIES: dict[int, str] = {
    400: "invalid_request",
    401: "auth",
    403: "auth",
    404: "model_not_found",
    429: "rate_limit",
}

# SDK exception class name -> category. Dispatch is by NAME, not isinstance, deliberately:
# the openai and anthropic hierarchies use identical names, so one table covers both without
# importing either SDK into this module (and without breaking on a future/absent SDK).
_EXC_NAME_CATEGORIES: dict[str, str] = {
    "APIConnectionError": "connection_error",
    "APIConnectionTimeoutError": "timeout",
    "APITimeoutError": "timeout",
    "AuthenticationError": "auth",
    "BadRequestError": "invalid_request",
    "InternalServerError": "server_error",
    "NotFoundError": "model_not_found",
    "PermissionDeniedError": "auth",
    "RateLimitError": "rate_limit",
    "UnprocessableEntityError": "invalid_request",
}

# Billing exhaustion is a MESSAGE-only distinction: Anthropic sends it as a 400 and OpenAI as
# a 429, so the status code alone would mislabel both. Checked ahead of typed dispatch.
_BILLING_MARKERS: tuple[str, ...] = (
    "credit balance",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "plans & billing",
    "billing details",
)


def _cause_chain(exc: BaseException) -> list[BaseException]:
    """The exception and its explicit `raise ... from` causes, outermost first.

    Only ``__cause__`` is followed, never ``__context__`` — implicit chaining would drag in
    unrelated exceptions that merely happened to be in flight. Guarded against cycles.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    return chain


def _http_code_in(msg: str, *codes: str) -> bool:
    """True when `msg` contains one of `codes` as a standalone token.

    Word-boundary matched so request ids, model names and token counts don't register as
    status codes — ``gpt-4290`` is not a 429 and ``req_5031`` is not a 503. ``.`` is excluded
    on both sides too, so version strings like ``1.429.0`` don't match either.
    """
    return any(re.search(rf"(?<![\w.]){code}(?![\w.])", msg) for code in codes)


def _classify_typed(exc: BaseException) -> str | None:
    """Classify off the SDK exception itself — status code first, then class name."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        mapped = _STATUS_CATEGORIES.get(status)
        if mapped is not None:
            return mapped
        if 500 <= status <= 599:
            return "server_error"
    return _EXC_NAME_CATEGORIES.get(type(exc).__name__)


def classify_error(exc: Exception) -> str:
    """Map an exception to a canonical error category string.

    Returns one of: timeout, rate_limit, auth, model_not_found, connection_error,
    server_error, content_policy, invalid_request, billing, unknown.
    Used by healthcheck and retry logic to produce specific messages.

    Three ordered stages: billing markers (message-only, so they must outrank the typed 400 /
    429 those failures arrive as), then typed dispatch over the ``__cause__`` chain (the
    authoritative path — ``generate()`` wraps SDK errors with ``from exc``, so the typed
    exception survives inside the ProviderError), then a string fallback for exceptions
    carrying no type signal at all.
    """
    chain = _cause_chain(exc)

    for link in chain:
        link_msg = str(link).lower()
        if any(marker in link_msg for marker in _BILLING_MARKERS):
            return "billing"

    for link in chain:
        typed = _classify_typed(link)
        if typed is not None:
            return typed

    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if _http_code_in(msg, "429") or "rate limit" in msg or "rate_limit" in msg:
        return "rate_limit"
    # server_error is checked BEFORE auth: a 5xx that merely mentions authentication is a
    # recoverable server failure, and the old auth-first order marked it non-retryable, which
    # made debate.py break the retry loop and burn the seat on a fully recoverable error.
    if _http_code_in(msg, "500", "502", "503") or "server error" in msg:
        return "server_error"
    if (
        _http_code_in(msg, "401", "403")
        or "unauthorized" in msg
        or "forbidden" in msg
        or "auth" in msg
        or "api key" in msg
        or "api_key" in msg
    ):
        return "auth"
    if _http_code_in(msg, "404") or "model_not_found" in msg:
        return "model_not_found"
    if (
        "connection" in msg
        or "unreachable" in msg
        or "network" in msg
        or "connect" in msg
    ):
        return "connection_error"
    if "content_policy" in msg or "content policy" in msg or "safety" in msg:
        return "content_policy"
    if "invalid" in msg:
        return "invalid_request"
    return "unknown"


def is_retryable(error_type: str) -> bool:
    """True when the error category warrants a single retry attempt."""
    return error_type in RETRYABLE_ERRORS


def classify_cli_failure(exc: Exception) -> str:
    """Map a CLI-seat subprocess failure to one of the shared CLI_FALLBACK_CAUSES tokens.

    Sibling of classify_error (which stays API-only for API exceptions): CLI failures are
    subprocess / parse / identity conditions — a different mechanism. The returned token IS
    the value written to seats[].fallback_events[].cause, so the classifier and the record
    share one vocabulary by construction (L-CLI IF#4). Always returns a member of
    CLI_FALLBACK_CAUSES; unrecognized failures fall through to ``process-error``.
    """
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if (
        "quota" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
        or "usage limit" in msg
        or "429" in msg
    ):
        return "quota"
    if "identity" in msg or "modelusage" in msg or "no served model" in msg or "banner" in msg:
        return "identity-unreadable"
    if "parse" in msg or "json" in msg or "decode" in msg or "unexpected output" in msg:
        return "parse"
    return "process-error"


class ProviderError(Exception):
    """Raised when a provider call fails."""

    def __init__(self, provider_name: str, message: str) -> None:
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


@dataclass
class _Parsed:
    """Normalized provider response — the output of AIProvider._parse().

    Return `_Parsed("")` (empty content) to trigger the base's generic
    "Empty response content" error; raise ProviderError from `_parse` for a
    provider-specific empty/degenerate message.
    """

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_count: int | None = None


def parse_openai_chat(raw: Any) -> _Parsed:
    """Shared `_parse` for the OpenAI-compatible chat family (openai, xai, deepseek).

    Duck-types the OpenAI ChatCompletion response shape (`.choices[0].message.content`
    + `.usage`), so the three OpenAI-compatible seats reduce to one parser without merging
    their classes.
    """
    choice = raw.choices[0] if raw.choices else None
    if not choice or not choice.message.content:
        return _Parsed("")
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_count: int | None = None
    if raw.usage:
        input_tokens = raw.usage.prompt_tokens
        output_tokens = raw.usage.completion_tokens
        token_count = raw.usage.total_tokens
    return _Parsed(
        content=choice.message.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_count=token_count,
    )


class AIProvider(ABC):
    """Abstract base for all AI model providers."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._api_key = os.environ.get(config.api_key_env, "").strip()
        if not self._api_key:
            raise ProviderError(config.name, f"Missing API key: {config.api_key_env}")
        self._configure()

    def name(self) -> str:
        """Return the short provider name (e.g. 'gemini', 'claude')."""
        return self._config.name

    def model_string(self) -> str:
        """Return the actual model identifier string."""
        return self._config.model

    @property
    def timeout_sec(self) -> float:
        """The seat's configured per-call timeout budget (seconds). Public read so the retry
        contract can grow the budget per attempt without reaching into ``_config``."""
        return self._config.timeout_sec

    async def generate(
        self, prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The full prompt text to send.
            round_number: The debate round number (1-indexed).
            timeout: Optional per-call timeout override (seconds); falls back to the
                seat's configured ``timeout_sec``. Used by the retry contract to grow the
                budget per attempt without mutating provider state.

        Returns:
            ModelResponse dataclass with content and metadata.

        Raises:
            ProviderError: On API failure, timeout, or invalid response.
        """
        effective_timeout = timeout if timeout is not None else self._config.timeout_sec
        start = time.monotonic()
        try:
            raw = await asyncio.wait_for(self._invoke(prompt), timeout=effective_timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise ProviderError(
                self._config.name,
                f"Request timed out after {effective_timeout}s",
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self._config.name, f"API call failed: {exc}") from exc

        parsed = self._parse(raw)
        if not parsed.content:
            raise ProviderError(self._config.name, "Empty response content")

        latency = time.monotonic() - start
        logger.info(
            "%s round %d: %.2fs, %s tokens",
            self._config.name,
            round_number,
            latency,
            parsed.token_count,
        )
        return ModelResponse(
            provider=self._config.name,
            model=self._config.model,
            round_number=round_number,
            content=parsed.content,
            latency_sec=latency,
            token_count=parsed.token_count,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
        )

    def _configure(self) -> None:
        """Build the SDK client. Default: no-op (gemini builds a client per call).

        The base has already set ``self._config`` and ``self._api_key`` before this runs.
        """

    @abstractmethod
    async def _invoke(self, prompt: str) -> Any:
        """Execute the SDK call and return its raw (unparsed) response."""
        ...

    @abstractmethod
    def _parse(self, raw: Any) -> _Parsed:
        """Reduce a raw SDK response to `_Parsed` (text + token counts)."""
        ...
