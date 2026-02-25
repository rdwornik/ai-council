"""Provider health checks — ping each API before starting a debate."""

import asyncio
import logging

from src.providers.base import AIProvider

logger = logging.getLogger(__name__)

_PING_PROMPT = "Reply with the word OK only."
_TIMEOUT_SEC = 15.0


async def _check_one(name: str, provider: AIProvider) -> tuple[str, bool, str]:
    """Ping a single provider. Returns (name, ok, error_message)."""
    try:
        await asyncio.wait_for(
            provider.generate(_PING_PROMPT, round_number=0),
            timeout=_TIMEOUT_SEC,
        )
        return name, True, ""
    except Exception as exc:
        return name, False, str(exc)


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
