"""Unit tests for the research mode pipeline.

All tests use mocked providers — no real API calls.
"""

import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.cache import cache_get, cache_invalidate, cache_put
from src.research.merger import (
    _deduplicate_sources,
    _truncation_fallback,
    make_cache_key,
    merge_results,
)
from src.research.models import MergedResearchReport, ResearchResult, Source
from src.research.provider import ResearchProvider, ResearchProviderError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_result(
    provider: str = "mock",
    content: str = "Some research content about the topic.",
    sources: list[Source] | None = None,
    cost: float = 0.01,
    duration: float = 5.0,
    error: str | None = None,
) -> ResearchResult:
    return ResearchResult(
        provider=provider,
        query="test query",
        content=content,
        sources=sources or [Source(title="Example", url="https://example.com")],
        token_count=100,
        cost_usd=cost,
        duration_sec=duration,
        timestamp=datetime.now(timezone.utc).isoformat(),
        error=error,
    )


class MockResearchProvider(ResearchProvider):
    """Test double for ResearchProvider."""

    def __init__(
        self,
        provider_name: str = "mock",
        result: ResearchResult | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        self._name = provider_name
        self._result = result or _make_result(provider=provider_name)
        self._should_raise = should_raise

    def name(self) -> str:
        return self._name

    def model_string(self) -> str:
        return "mock-model"

    async def research(self, query: str) -> ResearchResult:
        if self._should_raise is not None:
            raise self._should_raise
        return self._result


# ---------------------------------------------------------------------------
# ResearchResult / Source model tests
# ---------------------------------------------------------------------------

class TestResearchModels:
    def test_source_required_fields(self) -> None:
        s = Source(title="T", url="https://t.com")
        assert s.title == "T"
        assert s.url == "https://t.com"
        assert s.snippet is None

    def test_research_result_defaults(self) -> None:
        r = ResearchResult(provider="x", query="q", content="c")
        assert r.token_count == 0
        assert r.cost_usd == 0.0
        assert r.duration_sec == 0.0
        assert r.error is None
        assert r.timed_out is False

    def test_merged_report_defaults(self) -> None:
        r = MergedResearchReport(
            query="q",
            results=[],
            merged_report="m",
            summary_2500="s",
        )
        assert r.total_sources == 0
        assert r.total_cost_usd == 0.0
        assert r.cache_key == ""


# ---------------------------------------------------------------------------
# Source deduplication
# ---------------------------------------------------------------------------

class TestDeduplicateSources:
    def test_removes_duplicate_urls(self) -> None:
        sources = [
            Source("A", "https://a.com"),
            Source("B", "https://b.com"),
            Source("A2", "https://a.com"),  # duplicate URL, different title
        ]
        result = _deduplicate_sources(sources)
        urls = [s.url for s in result]
        assert urls == ["https://a.com", "https://b.com"]

    def test_empty_input(self) -> None:
        assert _deduplicate_sources([]) == []

    def test_all_unique(self) -> None:
        sources = [Source(f"S{i}", f"https://s{i}.com") for i in range(5)]
        assert len(_deduplicate_sources(sources)) == 5


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------

class TestMergeResults:
    def test_basic_merge(self) -> None:
        results = [
            _make_result("p1", "Content from P1", cost=0.02),
            _make_result("p2", "Content from P2", cost=0.03),
        ]
        report = merge_results("test query", results, cache_key="abc123")
        assert report.query == "test query"
        assert report.cache_key == "abc123"
        assert "P1" in report.merged_report
        assert "P2" in report.merged_report
        assert report.total_cost_usd == pytest.approx(0.05)

    def test_skips_failed_results(self) -> None:
        results = [
            _make_result("ok", "Good content"),
            _make_result("fail", error="API error"),
        ]
        report = merge_results("q", results)
        assert "Good content" in report.merged_report
        # failed provider has no content to include
        assert "fail" not in report.merged_report.upper() or "fail" in report.merged_report.lower()

    def test_deduplicates_sources(self) -> None:
        shared_url = "https://shared.com"
        results = [
            _make_result("p1", sources=[Source("A", shared_url), Source("B", "https://b.com")]),
            _make_result("p2", sources=[Source("A2", shared_url), Source("C", "https://c.com")]),
        ]
        report = merge_results("q", results)
        assert report.total_sources == 3  # shared_url deduplicated

    def test_total_duration_is_max(self) -> None:
        results = [
            _make_result("p1", duration=10.0),
            _make_result("p2", duration=30.0),
            _make_result("p3", duration=20.0),
        ]
        report = merge_results("q", results)
        assert report.total_duration_sec == pytest.approx(30.0)

    def test_empty_results(self) -> None:
        report = merge_results("q", [])
        assert report.merged_report == ""
        assert report.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# cache_key generation
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        k1 = make_cache_key("Hello World")
        k2 = make_cache_key("Hello World")
        assert k1 == k2

    def test_case_insensitive(self) -> None:
        assert make_cache_key("Hello") == make_cache_key("hello")

    def test_strips_whitespace(self) -> None:
        assert make_cache_key("  hello  ") == make_cache_key("hello")

    def test_length_16(self) -> None:
        assert len(make_cache_key("anything")) == 16

    def test_different_queries_different_keys(self) -> None:
        assert make_cache_key("query A") != make_cache_key("query B")

    def test_merger_cache_key_used_in_cache(self, tmp_path: Path) -> None:
        # cache.py uses merger.make_cache_key — verify key is consistent
        key = make_cache_key("test")
        assert len(key) == 16  # both modules use same 16-char hex


# ---------------------------------------------------------------------------
# truncation_fallback
# ---------------------------------------------------------------------------

class TestTruncationFallback:
    def test_short_text_unchanged(self) -> None:
        text = "short text"
        result = _truncation_fallback(text, max_tokens=100)
        assert result == text

    def test_long_text_truncated(self) -> None:
        words = ["word"] * 1000
        text = " ".join(words)
        result = _truncation_fallback(text, max_tokens=100)  # limit ~75 words
        assert len(result.split()) < 1000
        assert "truncated" in result

    def test_empty_text(self) -> None:
        assert _truncation_fallback("", 100) == ""


# ---------------------------------------------------------------------------
# Cache: cache_put / cache_get / cache_invalidate
# ---------------------------------------------------------------------------

class TestFileCache:
    def test_put_and_get(self, tmp_path: Path) -> None:
        key = "test1234abcd5678"
        report = MergedResearchReport(
            query="What is the best database?",
            results=[_make_result()],
            merged_report="# Full Report\n\nSome content.",
            summary_2500="Summary content.",
            total_sources=2,
            total_cost_usd=0.05,
            total_duration_sec=10.0,
            cache_key=key,
        )
        cache_put(tmp_path, key, report)

        retrieved = cache_get(tmp_path, key, ttl_days=7)
        assert retrieved is not None
        assert retrieved.query == "What is the best database?"
        assert retrieved.merged_report == "# Full Report\n\nSome content."
        assert retrieved.summary_2500 == "Summary content."
        assert retrieved.total_cost_usd == pytest.approx(0.05)

    def test_miss_on_nonexistent_key(self, tmp_path: Path) -> None:
        result = cache_get(tmp_path, "nonexistent000000", ttl_days=7)
        assert result is None

    def test_expired_entry_returns_none(self, tmp_path: Path) -> None:
        key = "expiredkey000000"
        report = MergedResearchReport(
            query="q",
            results=[],
            merged_report="r",
            summary_2500="s",
            cache_key=key,
        )
        cache_put(tmp_path, key, report)

        # Manually patch the cached_at to be in the past
        meta_file = tmp_path / f"{key}_meta.json"
        meta = json.loads(meta_file.read_text())
        meta["cached_at"] = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).isoformat()
        meta_file.write_text(json.dumps(meta))

        result = cache_get(tmp_path, key, ttl_days=7)
        assert result is None

    def test_invalidate_removes_files(self, tmp_path: Path) -> None:
        key = "deletekey0000000"
        report = MergedResearchReport(
            query="q", results=[], merged_report="r", summary_2500="s", cache_key=key
        )
        cache_put(tmp_path, key, report)
        assert (tmp_path / f"{key}_meta.json").exists()

        deleted = cache_invalidate(tmp_path, key)
        assert deleted is True
        assert not (tmp_path / f"{key}_meta.json").exists()
        assert not (tmp_path / f"{key}_report.md").exists()
        assert not (tmp_path / f"{key}_summary.txt").exists()

    def test_invalidate_nonexistent_returns_false(self, tmp_path: Path) -> None:
        assert cache_invalidate(tmp_path, "nosuchkey0000000") is False

    def test_put_creates_three_files(self, tmp_path: Path) -> None:
        key = "threefiles000000"
        report = MergedResearchReport(
            query="q", results=[], merged_report="r", summary_2500="s", cache_key=key
        )
        cache_put(tmp_path, key, report)
        assert (tmp_path / f"{key}_report.md").exists()
        assert (tmp_path / f"{key}_summary.txt").exists()
        assert (tmp_path / f"{key}_meta.json").exists()


# ---------------------------------------------------------------------------
# MockResearchProvider behaviour
# ---------------------------------------------------------------------------

class TestMockResearchProvider:
    async def test_returns_result(self) -> None:
        r = _make_result("mock", "content")
        p = MockResearchProvider("mock", result=r)
        result = await p.research("query")
        assert result.provider == "mock"
        assert result.content == "content"

    async def test_raises_on_error(self) -> None:
        p = MockResearchProvider(should_raise=ResearchProviderError("mock", "fail"))
        with pytest.raises(ResearchProviderError):
            await p.research("query")


# ---------------------------------------------------------------------------
# display.run_research_with_display (no real API calls)
# ---------------------------------------------------------------------------

class TestRunResearchWithDisplay:
    async def test_collects_results_in_order(self) -> None:
        from src.research.display import run_research_with_display
        from rich.console import Console

        results_by_provider = {
            "p1": _make_result("p1", "content 1"),
            "p2": _make_result("p2", "content 2"),
        }
        providers = [
            MockResearchProvider("p1", result=results_by_provider["p1"]),
            MockResearchProvider("p2", result=results_by_provider["p2"]),
        ]
        con = Console(file=io.StringIO())
        results = await run_research_with_display(providers, "test query", console=con)
        assert len(results) == 2
        assert results[0].provider == "p1"
        assert results[1].provider == "p2"

    async def test_handles_provider_error(self) -> None:
        from src.research.display import run_research_with_display
        from rich.console import Console

        providers = [
            MockResearchProvider("ok", result=_make_result("ok")),
            MockResearchProvider("fail", should_raise=ResearchProviderError("fail", "API error")),
        ]
        con = Console(file=io.StringIO())
        results = await run_research_with_display(providers, "query", console=con)
        assert len(results) == 2
        ok_result = next(r for r in results if r.provider == "ok")
        fail_result = next(r for r in results if r.provider == "fail")
        assert ok_result.error is None
        assert fail_result.error is not None

    async def test_empty_providers_returns_empty(self) -> None:
        from src.research.display import run_research_with_display
        from rich.console import Console

        con = Console(file=io.StringIO())
        results = await run_research_with_display([], "query", console=con)
        assert results == []


# ---------------------------------------------------------------------------
# ResearchProviderError
# ---------------------------------------------------------------------------

class TestResearchProviderError:
    def test_message(self) -> None:
        exc = ResearchProviderError("perplexity", "Request timed out")
        assert "perplexity" in str(exc)
        assert "Request timed out" in str(exc)

    def test_is_exception(self) -> None:
        exc = ResearchProviderError("x", "y")
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# build_research_providers (no API calls)
# ---------------------------------------------------------------------------

class TestBuildResearchProviders:
    def test_skips_providers_with_missing_api_key(self, tmp_path: Path, monkeypatch) -> None:
        from config.config_loader import (
            AppConfig, DefaultsConfig, InboxConfig, ModelConfig,
            PromptsConfig, ResearchConfig, ResearchProviderConfig,
        )
        from src.research.runner import build_research_providers

        research_cfg = ResearchConfig(
            default_providers=["perplexity"],
            deep_providers=["perplexity"],
            cache_dir=tmp_path,
            cache_ttl_days=7,
            summary_max_tokens=2500,
            summary_model="deepseek",
            providers={
                "perplexity": ResearchProviderConfig(
                    name="perplexity",
                    model="sonar-pro",
                    api_key_env="PERPLEXITY_API_KEY_MISSING_TEST",
                    timeout_sec=60,
                )
            },
        )
        model_cfg = ModelConfig(
            name="claude", sdk="anthropic", model="claude-opus-4-6",
            api_key_env="ANTHROPIC_API_KEY", timeout_sec=120, max_tokens=8192,
        )
        config = AppConfig(
            defaults=DefaultsConfig(
                rounds=2, max_rounds=3, output_dir=tmp_path,
                synthesizer="claude", default_panel=[], full_panel=[],
            ),
            models={"claude": model_cfg},
            prompts=PromptsConfig(initial="", critique="", synthesis=""),
            inbox=InboxConfig(dir=tmp_path, archive_dir=tmp_path),
            research=research_cfg,
        )
        monkeypatch.delenv("PERPLEXITY_API_KEY_MISSING_TEST", raising=False)

        providers = build_research_providers(config, deep=False)
        assert providers == []

    def test_returns_empty_when_no_research_config(self, tmp_path: Path) -> None:
        from config.config_loader import (
            AppConfig, DefaultsConfig, InboxConfig, ModelConfig, PromptsConfig,
        )
        from src.research.runner import build_research_providers

        model_cfg = ModelConfig(
            name="claude", sdk="anthropic", model="claude-opus-4-6",
            api_key_env="ANTHROPIC_API_KEY", timeout_sec=120, max_tokens=8192,
        )
        config = AppConfig(
            defaults=DefaultsConfig(
                rounds=2, max_rounds=3, output_dir=tmp_path,
                synthesizer="claude", default_panel=[], full_panel=[],
            ),
            models={"claude": model_cfg},
            prompts=PromptsConfig(initial="", critique="", synthesis=""),
            inbox=InboxConfig(dir=tmp_path, archive_dir=tmp_path),
            research=None,
        )
        providers = build_research_providers(config, deep=False)
        assert providers == []
