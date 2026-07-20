"""Tests for src/ai_council/research/headless.py.

The headless executor is the #18 crux-check's retrieval seam. Its whole reason to exist
is that run_research() is UI/cache/write-coupled, so most of these tests assert what the
executor does NOT do: no console, no cache, no file writes, no second LLM call.
"""

import asyncio

import pytest

from ai_council.research.headless import run_research_headless
from ai_council.research.models import MergedResearchReport, ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError
from config.config_loader import AppConfig


def _result(provider: str = "perplexity", content: str = "Evidence body.") -> ResearchResult:
    # NOTE: the URL is per-provider — merge_results counts UNIQUE sources, so a shared
    # URL across mocks would silently collapse the count.
    return ResearchResult(
        provider=provider,
        query="q",
        content=content,
        sources=[Source(title="T", url=f"https://example.test/{provider}")],
        token_count=100,
        cost_usd=0.01,
        duration_sec=1.0,
    )


class _MockResearchProvider(ResearchProvider):
    def __init__(
        self,
        provider_name: str = "perplexity",
        result: ResearchResult | None = None,
        should_raise: Exception | None = None,
        delay_sec: float = 0.0,
    ) -> None:
        self._name = provider_name
        self._result = result if result is not None else _result(provider_name)
        self._should_raise = should_raise
        self._delay_sec = delay_sec
        self.call_count = 0

    def name(self) -> str:
        return self._name

    def model_string(self) -> str:
        return f"{self._name}-model"

    async def research(self, query: str) -> ResearchResult:
        self.call_count += 1
        if self._delay_sec:
            await asyncio.sleep(self._delay_sec)
        if self._should_raise is not None:
            raise self._should_raise
        return self._result


@pytest.fixture
def config(sample_app_config: AppConfig) -> AppConfig:
    return sample_app_config


async def test_returns_none_when_no_providers_built(monkeypatch, config):
    """No API keys / no research section → build returns [], executor returns None."""
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", lambda *a, **k: []
    )
    assert await run_research_headless("q", config, provider_names=["perplexity"]) is None


async def test_merges_successful_results(monkeypatch, config):
    providers = [_MockResearchProvider("perplexity"), _MockResearchProvider("grok")]
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", lambda *a, **k: providers
    )
    report = await run_research_headless("q", config)
    assert isinstance(report, MergedResearchReport)
    assert report.total_sources == 2
    assert all(p.call_count == 1 for p in providers)


async def test_provider_exception_becomes_error_result(monkeypatch, config):
    """ResearchProviderError must not escape — it becomes an error ResearchResult."""
    providers = [
        _MockResearchProvider(
            "perplexity", should_raise=ResearchProviderError("perplexity", "boom")
        )
    ]
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", lambda *a, **k: providers
    )
    report = await run_research_headless("q", config)
    assert report is not None
    assert len(report.results) == 1
    assert report.results[0].error is not None


async def test_unexpected_exception_becomes_error_result(monkeypatch, config):
    """The non-ResearchProviderError branch is the one duplicated from display.py."""
    providers = [_MockResearchProvider("perplexity", should_raise=ValueError("kaboom"))]
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", lambda *a, **k: providers
    )
    report = await run_research_headless("q", config)
    assert report is not None
    assert report.results[0].error is not None


async def test_partial_failure_still_merges(monkeypatch, config):
    """One live provider is enough — ruling 3: >=1 success is grounded."""
    providers = [
        _MockResearchProvider("perplexity"),
        _MockResearchProvider("grok", should_raise=ResearchProviderError("grok", "down")),
    ]
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", lambda *a, **k: providers
    )
    report = await run_research_headless("q", config)
    assert report is not None
    assert not report.degraded  # min_successful=1, so one success is NOT degraded
    assert report.total_sources == 1


async def test_budget_timeout_returns_none(monkeypatch, config):
    providers = [_MockResearchProvider("perplexity", delay_sec=5.0)]
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", lambda *a, **k: providers
    )
    assert await run_research_headless("q", config, budget_sec=0.05) is None


async def test_writes_no_console_output(monkeypatch, config, capsys):
    """run_research prints a header + a Rich Live table. The headless path prints nothing."""
    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers",
        lambda *a, **k: [_MockResearchProvider("perplexity")],
    )
    await run_research_headless("q", config)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


async def test_no_cache_read_or_write(monkeypatch, config):
    """The crux check must not pollute or read the research cache."""
    import ai_council.research.headless as headless_mod

    assert not hasattr(headless_mod, "cache_get")
    assert not hasattr(headless_mod, "cache_put")


async def test_does_not_call_summarize_report(monkeypatch, config):
    """Contract says ONE LLM call (the extraction). summarize_report would be a second."""
    import ai_council.research.headless as headless_mod

    assert not hasattr(headless_mod, "summarize_report")


async def test_provider_names_filter_is_forwarded(monkeypatch, config):
    """The narrow crux panel reaches build_research_providers as models_filter."""
    seen: dict = {}

    def _capture(cfg, deep=False, models_filter=None):
        seen["deep"] = deep
        seen["models_filter"] = models_filter
        return [_MockResearchProvider("perplexity")]

    monkeypatch.setattr(
        "ai_council.research.headless.build_research_providers", _capture
    )
    await run_research_headless("q", config, provider_names=["perplexity"])
    assert seen["models_filter"] == ["perplexity"]
    assert seen["deep"] is False  # never the 1800s deep panel
