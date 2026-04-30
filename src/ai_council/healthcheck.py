"""Provider health checks — ping each API before starting a debate."""

import asyncio
import logging

from ai_council.providers.base import AIProvider, classify_error

logger = logging.getLogger(__name__)

_HEALTHCHECK_MESSAGES: dict[str, str] = {
    "timeout": "health check timed out",
    "rate_limit": "rate limited during health check",
    "auth": "authentication failed (check API key)",
    "model_not_found": "model not found (check model string in settings.yaml)",
    "connection_error": "endpoint unreachable",
    "server_error": "server error during health check",
    "content_policy": "content policy rejection during health check",
    "invalid_request": "invalid request during health check",
}

_PING_PROMPT = "Reply with the word OK only."
_DEFAULT_TIMEOUT_SEC = 30.0
_MAX_TIMEOUT_SEC = 60.0


def _ping_timeout(provider: AIProvider) -> float:
    """Return a per-provider health check timeout.

    Uses the provider's configured timeout capped at _MAX_TIMEOUT_SEC,
    so slow providers (Gemini, DeepSeek) get proportionally more time
    while the overall check never hangs indefinitely.
    """
    cfg = getattr(provider, "_config", None)
    if cfg is not None and hasattr(cfg, "timeout_sec"):
        return min(float(cfg.timeout_sec), _MAX_TIMEOUT_SEC)
    return _DEFAULT_TIMEOUT_SEC


async def _check_one(name: str, provider: AIProvider) -> tuple[str, bool, str]:
    """Ping a single provider. Returns (name, ok, error_message)."""
    timeout = _ping_timeout(provider)
    try:
        await asyncio.wait_for(
            provider.generate(_PING_PROMPT, round_number=0),
            timeout=timeout,
        )
        return name, True, ""
    except asyncio.TimeoutError:
        return name, False, f"health check timed out after {timeout:.0f}s"
    except Exception as exc:
        error_type = classify_error(exc)
        specific_msg = _HEALTHCHECK_MESSAGES.get(error_type)
        msg = specific_msg if specific_msg else (str(exc) or repr(exc))
        return name, False, msg


async def run_health_checks(
    providers: dict[str, AIProvider],
) -> dict[str, tuple[bool, str]]:
    """Ping all providers in parallel.

    Returns:
        Dict mapping provider name -> (ok, error_message).
        error_message is "" when ok is True.
    """
    results = await asyncio.gather(*(_check_one(n, p) for n, p in providers.items()))
    return {name: (ok, err) for name, ok, err in results}
