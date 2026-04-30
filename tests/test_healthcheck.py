"""Unit tests for src/healthcheck.py — no real API calls."""

from unittest.mock import AsyncMock

from ai_council.healthcheck import run_health_checks
from ai_council.models import ModelResponse
from ai_council.providers.base import ProviderError
from tests.conftest import MockProvider


def _ok_response(name: str) -> ModelResponse:
    return ModelResponse(
        provider=name,
        model="mock-model",
        round_number=0,
        content="OK",
        latency_sec=0.1,
        token_count=1,
    )


async def test_all_providers_pass():
    """All providers succeed -> all marked ok, no errors."""
    providers = {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
    }
    providers["claude"].generate = AsyncMock(return_value=_ok_response("claude"))
    providers["gemini"].generate = AsyncMock(return_value=_ok_response("gemini"))

    results = await run_health_checks(providers)

    assert results["claude"] == (True, "")
    assert results["gemini"] == (True, "")


async def test_one_provider_fails():
    """A provider that raises returns ok=False with the error message."""
    providers = {
        "claude": MockProvider("claude"),
        "grok": MockProvider("grok"),
    }
    providers["claude"].generate = AsyncMock(return_value=_ok_response("claude"))
    providers["grok"].generate = AsyncMock(
        side_effect=ProviderError("grok", "403 Forbidden")
    )

    results = await run_health_checks(providers)

    assert results["claude"] == (True, "")
    ok, err = results["grok"]
    assert ok is False
    assert "authentication failed" in err


async def test_all_providers_fail():
    """All fail -> all marked False."""
    providers = {
        "openai": MockProvider("openai"),
        "deepseek": MockProvider("deepseek"),
    }
    for name, p in providers.items():
        p.generate = AsyncMock(side_effect=Exception(f"{name} down"))

    results = await run_health_checks(providers)

    for name in providers:
        ok, err = results[name]
        assert ok is False
        assert name in err


async def test_empty_providers():
    """Empty provider dict returns empty results."""
    results = await run_health_checks({})
    assert results == {}


async def test_timeout_counts_as_failure():
    """A provider that hangs past the timeout is marked as failed."""
    import asyncio

    providers = {"slow": MockProvider("slow")}

    async def hang(*args, **kwargs):
        await asyncio.sleep(9999)

    providers["slow"].generate = AsyncMock(side_effect=hang)

    # Patch the default timeout to 0.05s so the test runs fast
    import ai_council.healthcheck as hc

    original = hc._DEFAULT_TIMEOUT_SEC
    hc._DEFAULT_TIMEOUT_SEC = 0.05
    try:
        results = await run_health_checks(providers)
    finally:
        hc._DEFAULT_TIMEOUT_SEC = original

    ok, err = results["slow"]
    assert ok is False
    assert "timed out" in err


# --- Error classification in healthcheck messages ---


async def test_auth_error_returns_specific_message():
    """401/403 errors produce 'authentication failed' message, not raw status code."""
    providers = {"deepseek": MockProvider("deepseek")}
    providers["deepseek"].generate = AsyncMock(
        side_effect=ProviderError("deepseek", "401 Unauthorized")
    )
    results = await run_health_checks(providers)
    ok, err = results["deepseek"]
    assert ok is False
    assert "authentication failed" in err
    assert "check API key" in err


async def test_connection_error_returns_specific_message():
    """Connection errors produce 'endpoint unreachable' message."""
    providers = {"deepseek": MockProvider("deepseek")}
    providers["deepseek"].generate = AsyncMock(
        side_effect=ProviderError("deepseek", "Connection refused")
    )
    results = await run_health_checks(providers)
    ok, err = results["deepseek"]
    assert ok is False
    assert "unreachable" in err


async def test_unknown_error_falls_back_to_raw_message():
    """Unclassified errors return the raw exception message."""
    providers = {"deepseek": MockProvider("deepseek")}
    providers["deepseek"].generate = AsyncMock(
        side_effect=Exception("some completely novel failure xyzzy")
    )
    results = await run_health_checks(providers)
    ok, err = results["deepseek"]
    assert ok is False
    assert "xyzzy" in err
