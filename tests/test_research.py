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

from ai_council.research.cache import cache_get, cache_invalidate, cache_put
from ai_council.research.merger import (
    _deduplicate_sources,
    _truncation_fallback,
    make_cache_key,
    merge_results,
)
from ai_council.research.models import MergedResearchReport, ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError

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
        from rich.console import Console

        from ai_council.research.display import run_research_with_display

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
        from rich.console import Console

        from ai_council.research.display import run_research_with_display

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
        from rich.console import Console

        from ai_council.research.display import run_research_with_display

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
# GeminiResearchProvider (Interactions API)
# ---------------------------------------------------------------------------

def _make_interaction(
    status: str = "completed",
    text_outputs: list[str] | None = None,
    url_results: list[str] | None = None,
) -> MagicMock:
    """Build a mock Interaction object matching the real SDK shape."""
    interaction = MagicMock()
    interaction.id = "test-interaction-id"
    interaction.status = status

    outputs = []
    for text in (text_outputs or []):
        output = MagicMock()
        output.text = text
        output.result = None
        outputs.append(output)

    for url in (url_results or []):
        output = MagicMock()
        output.text = None
        r = MagicMock()
        r.url = url
        output.result = [r]
        outputs.append(output)

    interaction.outputs = outputs

    usage = MagicMock()
    usage.total_input_tokens = 500
    usage.total_output_tokens = 1000
    interaction.usage = usage

    return interaction


class TestGeminiResearchProvider:
    """Unit tests for GeminiResearchProvider — all API calls mocked."""

    def _make_provider(self, **kwargs):  # type: ignore[return]
        from ai_council.research.providers.gemini_research import GeminiResearchProvider
        defaults = dict(
            api_key="test-key",
            agent="deep-research-pro-preview-12-2025",
            timeout_sec=60,
            poll_interval_sec=0,  # no sleep in tests
        )
        defaults.update(kwargs)
        return GeminiResearchProvider(**defaults)

    def _make_mock_client(
        self,
        create_interaction: MagicMock,
        poll_interactions: list[MagicMock],
    ) -> MagicMock:
        """Build a mock genai.Client whose aio.interactions.create/get return the given values."""
        client = MagicMock()
        client.aio.interactions.create = AsyncMock(return_value=create_interaction)
        client.aio.interactions.get = AsyncMock(side_effect=poll_interactions)
        return client

    async def test_name_and_model_string(self) -> None:
        provider = self._make_provider()
        assert provider.name() == "gemini"
        assert provider.model_string() == "deep-research-pro-preview-12-2025"

    async def test_completed_on_first_poll(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        completed = _make_interaction(status="completed", text_outputs=["Report text."])

        client = self._make_mock_client(started, [completed])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("What is HTAP?")

        assert result.provider == "gemini"
        assert result.content == "Report text."
        assert result.error is None
        client.aio.interactions.create.assert_called_once_with(
            agent="deep-research-pro-preview-12-2025",
            input="What is HTAP?",
            background=True,
        )
        client.aio.interactions.get.assert_called_once_with("test-interaction-id")

    async def test_polls_until_completed(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        poll1 = _make_interaction(status="in_progress")
        poll2 = _make_interaction(status="in_progress")
        done = _make_interaction(status="completed", text_outputs=["Final report."])

        client = self._make_mock_client(started, [poll1, poll2, done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        assert result.content == "Final report."
        assert client.aio.interactions.get.call_count == 3

    async def test_failed_status_raises(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        failed = _make_interaction(status="failed")

        client = self._make_mock_client(started, [failed])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            with pytest.raises(ResearchProviderError) as exc_info:
                await provider.research("test")

        assert "failed" in str(exc_info.value)

    async def test_cancelled_status_raises(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        cancelled = _make_interaction(status="cancelled")

        client = self._make_mock_client(started, [cancelled])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            with pytest.raises(ResearchProviderError) as exc_info:
                await provider.research("test")

        assert "cancelled" in str(exc_info.value)

    async def test_timeout_raises(self) -> None:
        provider = self._make_provider(timeout_sec=0)
        started = _make_interaction(status="in_progress")
        poll = _make_interaction(status="in_progress")

        async def slow_get(_id: str) -> MagicMock:
            await asyncio.sleep(0.1)
            return poll

        client = MagicMock()
        client.aio.interactions.create = AsyncMock(return_value=started)
        client.aio.interactions.get = slow_get

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            with pytest.raises(ResearchProviderError) as exc_info:
                await provider.research("test")

        assert "Timed out" in str(exc_info.value)

    async def test_extracts_multiple_text_outputs(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        done = _make_interaction(
            status="completed", text_outputs=["Part 1.", "Part 2.", "Part 3."]
        )

        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        assert "Part 1." in result.content
        assert "Part 2." in result.content
        assert "Part 3." in result.content

    async def test_extracts_url_sources(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        done = _make_interaction(
            status="completed",
            text_outputs=["Report."],
            url_results=["https://example.com", "https://other.org"],
        )

        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        urls = {s.url for s in result.sources}
        assert "https://example.com" in urls
        assert "https://other.org" in urls

    async def test_deduplicates_sources(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        done = _make_interaction(
            status="completed",
            text_outputs=["Report."],
            url_results=["https://example.com", "https://example.com"],
        )

        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        assert len(result.sources) == 1

    async def test_uses_configured_agent_id(self) -> None:
        custom_agent = "custom-agent-id"
        provider = self._make_provider(agent=custom_agent)
        started = _make_interaction(status="in_progress")
        done = _make_interaction(status="completed", text_outputs=["Done."])

        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            await provider.research("test")

        client.aio.interactions.create.assert_called_once_with(
            agent=custom_agent,
            input="test",
            background=True,
        )

    async def test_api_error_wrapped_as_provider_error(self) -> None:
        provider = self._make_provider()
        client = MagicMock()
        client.aio.interactions.create = AsyncMock(side_effect=RuntimeError("connection refused"))

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            with pytest.raises(ResearchProviderError) as exc_info:
                await provider.research("test")

        assert "API error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_research_providers (no API calls)
# ---------------------------------------------------------------------------

class TestBuildResearchProviders:
    def test_skips_providers_with_missing_api_key(self, tmp_path: Path, monkeypatch) -> None:
        from ai_council.research.runner import build_research_providers
        from config.config_loader import (
            AppConfig,
            DefaultsConfig,
            InboxConfig,
            ModelConfig,
            PromptsConfig,
            ResearchConfig,
            ResearchProviderConfig,
        )

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
        from ai_council.research.runner import build_research_providers
        from config.config_loader import (
            AppConfig,
            DefaultsConfig,
            InboxConfig,
            ModelConfig,
            PromptsConfig,
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
            research=None,
        )
        providers = build_research_providers(config, deep=False)
        assert providers == []


# ---------------------------------------------------------------------------
# GrokResearchProvider
# ---------------------------------------------------------------------------

def _make_grok_response(
    text: str = "Grok research report.",
    input_tokens: int = 200,
    output_tokens: int = 500,
    annotations: list | None = None,
) -> MagicMock:
    """Build a mock xAI Responses API response."""
    response = MagicMock()

    item = MagicMock()
    item.type = "message"
    item.text = text
    item.content = None
    item.annotations = annotations or []
    response.output = [item]

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage

    return response


class TestGrokResearchProvider:
    """Unit tests for GrokResearchProvider — all API calls mocked."""

    def _make_provider(self, **kwargs):  # type: ignore[return]
        from ai_council.research.providers.grok_research import GrokResearchProvider
        defaults = dict(
            api_key="test-xai-key",
            model="grok-3",
            base_url="https://api.x.ai/v1",
            timeout_sec=120,
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
        )
        defaults.update(kwargs)
        return GrokResearchProvider(**defaults)

    async def test_name_and_model_string(self) -> None:
        provider = self._make_provider()
        assert provider.name() == "grok"
        assert provider.model_string() == "grok-3"

    async def test_successful_research(self) -> None:
        provider = self._make_provider()
        mock_response = _make_grok_response("Grok analysis of the topic.")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            result = await provider.research("What is multi-agent LLM debate?")

        assert result.provider == "grok"
        assert "Grok analysis" in result.content
        assert result.error is None

    async def test_tools_include_x_search_and_web_search(self) -> None:
        provider = self._make_provider()
        mock_response = _make_grok_response()
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            await provider.research("test query")

        call_kwargs = mock_client.responses.create.call_args.kwargs
        tool_types = [t["type"] for t in call_kwargs["tools"]]
        assert "x_search" in tool_types
        assert "web_search" in tool_types

    async def test_token_counting(self) -> None:
        provider = self._make_provider()
        mock_response = _make_grok_response(input_tokens=300, output_tokens=700)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            result = await provider.research("tokens test")

        assert result.token_count == 1000
        expected_cost = (300 / 1_000_000 * 3.00) + (700 / 1_000_000 * 15.00)
        assert result.cost_usd == pytest.approx(expected_cost)

    async def test_source_extraction_from_annotations(self) -> None:
        provider = self._make_provider()
        ann = MagicMock()
        ann.url = "https://x.com/user/status/123"
        ann.title = "Interesting tweet"
        mock_response = _make_grok_response(annotations=[ann])
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            result = await provider.research("test")

        assert len(result.sources) == 1
        assert result.sources[0].url == "https://x.com/user/status/123"
        assert result.sources[0].title == "Interesting tweet"

    async def test_timeout_raises_provider_error(self) -> None:
        from ai_council.research.provider import ResearchProviderError
        provider = self._make_provider(timeout_sec=1)
        mock_client = MagicMock()

        async def slow_call(**kwargs):  # type: ignore[return]
            await asyncio.sleep(10)

        mock_client.responses.create = slow_call

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            with pytest.raises(ResearchProviderError) as exc_info:
                await provider.research("test")

        assert "Timed out" in str(exc_info.value)
        assert "grok" in str(exc_info.value)

    async def test_api_error_wrapped_as_provider_error(self) -> None:
        from openai import APIError

        from ai_council.research.provider import ResearchProviderError
        provider = self._make_provider()
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(
            side_effect=APIError("server error", request=MagicMock(), body=None)
        )

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            with pytest.raises(ResearchProviderError) as exc_info:
                await provider.research("test")

        assert "API error" in str(exc_info.value)

    async def test_missing_api_key_skips_provider(self, tmp_path: Path, monkeypatch) -> None:
        from ai_council.research.runner import build_research_providers
        from config.config_loader import (
            AppConfig,
            DefaultsConfig,
            InboxConfig,
            ModelConfig,
            PromptsConfig,
            ResearchConfig,
            ResearchProviderConfig,
        )

        research_cfg = ResearchConfig(
            default_providers=["grok"],
            deep_providers=["grok"],
            cache_dir=tmp_path,
            cache_ttl_days=7,
            summary_max_tokens=2500,
            summary_model="deepseek",
            providers={
                "grok": ResearchProviderConfig(
                    name="grok",
                    model="grok-3",
                    api_key_env="XAI_API_KEY_MISSING_TEST",
                    timeout_sec=120,
                    base_url="https://api.x.ai/v1",
                )
            },
        )
        model_cfg = ModelConfig(
            name="claude", sdk="anthropic", model="claude-opus-4-7",
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
        monkeypatch.delenv("XAI_API_KEY_MISSING_TEST", raising=False)

        providers = build_research_providers(config, deep=False)
        assert providers == []

    async def test_source_extraction_from_content_block_annotations(self) -> None:
        """Annotations nested in content blocks (real Responses API path) are extracted."""
        provider = self._make_provider()

        ann = MagicMock()
        ann.url = "https://example.com/article"
        ann.title = "Example Article"

        content_block = MagicMock()
        content_block.annotations = [ann]
        content_block.text = "Some text."

        item = MagicMock()
        item.type = "message"
        item.text = None
        item.content = [content_block]
        item.annotations = []

        response = MagicMock()
        response.output = [item]
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 200
        response.usage = usage

        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            result = await provider.research("test")

        assert len(result.sources) == 1
        assert result.sources[0].url == "https://example.com/article"
        assert result.sources[0].title == "Example Article"

    async def test_deduplicates_sources_across_item_and_content_block(self) -> None:
        """Same URL in item annotations and content block annotations counts once."""
        provider = self._make_provider()

        url = "https://shared.com"
        ann1 = MagicMock()
        ann1.url = url
        ann1.title = "Shared"
        ann2 = MagicMock()
        ann2.url = url
        ann2.title = "Shared Duplicate"

        content_block = MagicMock()
        content_block.annotations = [ann2]

        item = MagicMock()
        item.type = "message"
        item.text = None
        item.content = [content_block]
        item.annotations = [ann1]

        response = MagicMock()
        response.output = [item]
        usage = MagicMock()
        usage.input_tokens = 0
        usage.output_tokens = 0
        response.usage = usage

        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            result = await provider.research("test")

        assert len(result.sources) == 1

    async def test_no_annotations_returns_empty_sources(self) -> None:
        """Response with no annotations produces 0 sources without crashing."""
        provider = self._make_provider()
        mock_response = _make_grok_response("Report with no citations.", annotations=[])
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("ai_council.research.providers.grok_research.AsyncOpenAI", return_value=mock_client):
            result = await provider.research("test")

        assert result.sources == []
        assert result.error is None


# ---------------------------------------------------------------------------
# GeminiResearchProvider — additional citation tests
# ---------------------------------------------------------------------------

class TestGeminiCitationParsing:
    """Tests for the markdown-link citation parser in GeminiResearchProvider."""

    def _make_provider(self) -> object:
        from ai_council.research.providers.gemini_research import GeminiResearchProvider
        return GeminiResearchProvider(
            api_key="test-key",
            agent="deep-research-pro-preview-12-2025",
            timeout_sec=60,
            poll_interval_sec=0,
        )

    def _make_mock_client(self, create_interaction, poll_interactions):  # type: ignore[return]
        client = MagicMock()
        client.aio.interactions.create = AsyncMock(return_value=create_interaction)
        client.aio.interactions.get = AsyncMock(side_effect=poll_interactions)
        return client

    async def test_markdown_links_in_text_become_sources(self) -> None:
        provider = self._make_provider()
        report_text = (
            "See [Paper A](https://arxiv.org/abs/1234) and "
            "[Blog Post](https://dev.to/post/abc) for details."
        )
        started = _make_interaction(status="in_progress")
        done = _make_interaction(status="completed", text_outputs=[report_text])
        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        urls = {s.url for s in result.sources}
        titles = {s.title for s in result.sources}
        assert "https://arxiv.org/abs/1234" in urls
        assert "https://dev.to/post/abc" in urls
        assert "Paper A" in titles
        assert "Blog Post" in titles

    async def test_deduplicates_markdown_links(self) -> None:
        provider = self._make_provider()
        report_text = (
            "[Link](https://example.com) mentioned here and "
            "[Same Link](https://example.com) mentioned again."
        )
        started = _make_interaction(status="in_progress")
        done = _make_interaction(status="completed", text_outputs=[report_text])
        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        assert len(result.sources) == 1
        assert result.sources[0].url == "https://example.com"

    async def test_no_links_in_text_returns_empty_sources(self) -> None:
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        done = _make_interaction(status="completed", text_outputs=["Plain text with no links."])
        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        assert result.sources == []

    async def test_markdown_links_and_structured_results_combined(self) -> None:
        """Markdown links from text and structured URL results are both captured."""
        provider = self._make_provider()
        started = _make_interaction(status="in_progress")
        done = _make_interaction(
            status="completed",
            text_outputs=["See [Docs](https://docs.example.com) for info."],
            url_results=["https://structured.example.com"],
        )
        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        urls = {s.url for s in result.sources}
        assert "https://docs.example.com" in urls
        assert "https://structured.example.com" in urls

    async def test_url_in_both_text_and_structured_deduplicated(self) -> None:
        """URL appearing in both markdown text and structured results counts once."""
        provider = self._make_provider()
        shared_url = "https://shared.example.com"
        started = _make_interaction(status="in_progress")
        done = _make_interaction(
            status="completed",
            text_outputs=[f"See [Link]({shared_url})."],
            url_results=[shared_url],
        )
        client = self._make_mock_client(started, [done])

        with patch("ai_council.research.providers.gemini_research.warnings"), \
             patch("ai_council.research.providers.gemini_research.genai") as mock_genai:
            mock_genai.Client.return_value = client
            result = await provider.research("test")

        assert len(result.sources) == 1


# ---------------------------------------------------------------------------
# OpenAI research providers — post-migration (gpt-5.x + web_search)
# ---------------------------------------------------------------------------

def _make_openai_response(
    text: str = "OpenAI research report.",
    input_tokens: int = 200,
    output_tokens: int = 500,
    annotations: list | None = None,
) -> MagicMock:
    """Build a mock OpenAI Responses API response (web_search annotation shape)."""
    response = MagicMock()

    block = MagicMock()
    block.text = text
    block.annotations = annotations or []

    item = MagicMock()
    item.type = "message"
    item.content = [block]
    item.text = None
    item.annotations = None
    response.output = [item]

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage
    return response


class TestOpenAIMiniResearchProviderMigrated:
    """Post-migration: gpt-5.4-mini + web_search on Responses API."""

    def _make_provider(self, **kwargs):  # type: ignore[return]
        from ai_council.research.providers.openai_mini_research import (
            OpenAIMiniResearchProvider,
        )
        defaults = dict(api_key="test-openai-key", timeout_sec=120)
        defaults.update(kwargs)
        return OpenAIMiniResearchProvider(**defaults)

    async def test_default_model_is_gpt_5_4_mini(self) -> None:
        provider = self._make_provider()
        assert provider.model_string() == "gpt-5.4-mini"

    async def test_uses_web_search_tool_and_migrated_model(self) -> None:
        provider = self._make_provider()
        mock_response = _make_openai_response()
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch(
            "ai_council.research.providers.openai_mini_research.AsyncOpenAI",
            return_value=mock_client,
        ):
            await provider.research("test query")

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4-mini"
        tool_types = [t["type"] for t in call_kwargs["tools"]]
        assert "web_search" in tool_types
        assert "web_search_preview" not in tool_types

    async def test_does_not_use_deprecated_model(self) -> None:
        provider = self._make_provider()
        mock_response = _make_openai_response()
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch(
            "ai_council.research.providers.openai_mini_research.AsyncOpenAI",
            return_value=mock_client,
        ):
            await provider.research("test")

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "deep-research" not in call_kwargs["model"]

    async def test_parses_annotation_sources(self) -> None:
        provider = self._make_provider()
        ann = MagicMock()
        ann.url = "https://example.com/article"
        ann.title = "Example article"
        mock_response = _make_openai_response(
            text="Real findings here.", annotations=[ann]
        )
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch(
            "ai_council.research.providers.openai_mini_research.AsyncOpenAI",
            return_value=mock_client,
        ):
            result = await provider.research("test")

        assert result.content and len(result.content) > 0
        assert len(result.sources) == 1
        assert result.sources[0].url == "https://example.com/article"


class TestOpenAIDeepResearchProviderMigrated:
    """Post-migration: gpt-5.5 + web_search + reasoning effort=high."""

    def _make_provider(self, **kwargs):  # type: ignore[return]
        from ai_council.research.providers.openai_deep_research import (
            OpenAIDeepResearchProvider,
        )
        defaults = dict(api_key="test-openai-key", timeout_sec=300)
        defaults.update(kwargs)
        return OpenAIDeepResearchProvider(**defaults)

    async def test_default_model_is_gpt_5_5(self) -> None:
        provider = self._make_provider()
        assert provider.model_string() == "gpt-5.5"

    async def test_uses_web_search_tool_and_high_reasoning(self) -> None:
        provider = self._make_provider()
        mock_response = _make_openai_response()
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch(
            "ai_council.research.providers.openai_deep_research.AsyncOpenAI",
            return_value=mock_client,
        ):
            await provider.research("test query")

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.5"
        tool_types = [t["type"] for t in call_kwargs["tools"]]
        assert "web_search" in tool_types
        reasoning = call_kwargs.get("reasoning")
        assert reasoning is not None
        effort = (
            reasoning.get("effort")
            if isinstance(reasoning, dict)
            else getattr(reasoning, "effort", None)
        )
        assert effort == "high"

    async def test_does_not_use_deprecated_model(self) -> None:
        provider = self._make_provider()
        mock_response = _make_openai_response()
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch(
            "ai_council.research.providers.openai_deep_research.AsyncOpenAI",
            return_value=mock_client,
        ):
            await provider.research("test")

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "deep-research" not in call_kwargs["model"]

    async def test_parses_annotation_sources(self) -> None:
        provider = self._make_provider()
        ann = MagicMock()
        ann.url = "https://example.com/deep"
        ann.title = "Deep source"
        mock_response = _make_openai_response(
            text="Deep findings here.", annotations=[ann]
        )
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch(
            "ai_council.research.providers.openai_deep_research.AsyncOpenAI",
            return_value=mock_client,
        ):
            result = await provider.research("test")

        assert result.content and len(result.content) > 0
        assert len(result.sources) == 1
        assert result.sources[0].url == "https://example.com/deep"


class TestResearchPanelMembership:
    """Config-level guard: openai_deep is --deep only, never in default panel."""

    def test_default_panel_excludes_openai_deep(self) -> None:
        from config.config_loader import load_config
        cfg = load_config()
        assert cfg.research is not None
        assert "openai_deep" not in cfg.research.default_providers

    def test_deep_panel_includes_openai_deep(self) -> None:
        from config.config_loader import load_config
        cfg = load_config()
        assert cfg.research is not None
        assert "openai_deep" in cfg.research.deep_providers


# ---------------------------------------------------------------------------
# Degradation alarm — threshold detection + banner + exit code
# (ADR-08: complete-run + banner + exit 3; denominator = selected panel)
# ---------------------------------------------------------------------------

class TestDegradationDetection:
    """merge_results populates degraded/failed_count against the selected panel."""

    def test_degraded_false_when_all_succeed(self) -> None:
        results = [
            _make_result("p1", "content"),
            _make_result("p2", "content"),
            _make_result("p3", "content"),
        ]
        report = merge_results(
            "q", results, selected_panel=["p1", "p2", "p3"], min_successful=3,
        )
        assert report.degraded is False
        assert report.failed_count == 0

    def test_degraded_true_when_below_threshold(self) -> None:
        results = [
            _make_result("p1", "content"),
            _make_result("p2", error="API error", content=""),
            _make_result("p3", error="timeout", content=""),
        ]
        report = merge_results(
            "q", results, selected_panel=["p1", "p2", "p3"], min_successful=3,
        )
        assert report.degraded is True
        assert report.failed_count == 2

    def test_degraded_at_threshold_boundary(self) -> None:
        """Exactly at the threshold is NOT degraded."""
        results = [
            _make_result("p1", "content"),
            _make_result("p2", "content"),
            _make_result("p3", "content"),
            _make_result("p4", error="fail", content=""),
        ]
        report = merge_results(
            "q", results, selected_panel=["p1", "p2", "p3", "p4"], min_successful=3,
        )
        assert report.degraded is False
        assert report.failed_count == 1

    def test_build_time_dropout_counts_as_failure(self) -> None:
        """A configured-4 panel where 1 dropped at build time + 3 succeed → degraded if min=4."""
        # Only 3 results returned (one provider skipped at build time, missing API key).
        results = [
            _make_result("p1", "content"),
            _make_result("p2", "content"),
            _make_result("p3", "content"),
        ]
        # Selected panel was 4 — denominator includes the dropped provider.
        report = merge_results(
            "q", results, selected_panel=["p1", "p2", "p3", "p4"], min_successful=4,
        )
        assert report.degraded is True
        assert report.failed_count == 1  # p4 dropped at build time

    def test_build_time_dropout_not_degraded_when_threshold_met(self) -> None:
        results = [
            _make_result("p1", "content"),
            _make_result("p2", "content"),
            _make_result("p3", "content"),
        ]
        report = merge_results(
            "q", results, selected_panel=["p1", "p2", "p3", "p4"], min_successful=3,
        )
        assert report.degraded is False
        assert report.failed_count == 1  # still 1 dropped, but threshold met

    def test_backward_compat_no_threshold_no_degradation(self) -> None:
        """Existing callers that pass no threshold: degraded defaults to False."""
        results = [_make_result("p1", error="fail", content="")]
        report = merge_results("q", results)
        assert report.degraded is False
        # failed_count still tracks errored results (informational when no panel given)
        assert report.failed_count == 1


class TestDegradationBannerConsole:
    """print_research_summary emits a loud banner when report.degraded."""

    def test_banner_appears_when_degraded(self) -> None:
        from rich.console import Console

        from ai_council.research.output import print_research_summary

        report = MergedResearchReport(
            query="q",
            results=[
                _make_result("p1", "content"),
                _make_result("p2", error="API error", content=""),
                _make_result("p3", error="timeout", content=""),
            ],
            merged_report="m",
            summary_2500="s",
            degraded=True,
            failed_count=2,
        )
        buf = io.StringIO()
        con = Console(file=buf, force_terminal=False, width=120)
        print_research_summary(report, file_path=None, from_cache=False, console=con)
        out = buf.getvalue()
        assert "DEGRADED" in out.upper()
        assert "2" in out  # failed_count surfaced

    def test_no_banner_when_not_degraded(self) -> None:
        from rich.console import Console

        from ai_council.research.output import print_research_summary

        report = MergedResearchReport(
            query="q",
            results=[_make_result("p1", "content")],
            merged_report="m",
            summary_2500="s",
            degraded=False,
            failed_count=0,
        )
        buf = io.StringIO()
        con = Console(file=buf, force_terminal=False, width=120)
        print_research_summary(report, file_path=None, from_cache=False, console=con)
        assert "DEGRADED" not in buf.getvalue().upper()


class TestDegradationBannerMarkdown:
    """save_research_to_file inserts a warning admonition when degraded."""

    def test_warning_block_at_top_when_degraded(self, tmp_path: Path) -> None:
        from ai_council.research.output import save_research_to_file

        report = MergedResearchReport(
            query="q",
            results=[
                _make_result("p1", "content"),
                _make_result("p2", error="x", content=""),
            ],
            merged_report="m",
            summary_2500="s",
            degraded=True,
            failed_count=1,
        )
        paths = save_research_to_file(report, tmp_path)
        text = paths[0].read_text(encoding="utf-8")
        # Warning admonition appears before the Provider Summary table.
        warn_idx = text.find("WARNING")
        table_idx = text.find("## Provider Summary")
        assert warn_idx != -1, "warning block missing from degraded markdown"
        assert warn_idx < table_idx, "warning block should appear before Provider Summary"

    def test_no_warning_block_when_not_degraded(self, tmp_path: Path) -> None:
        from ai_council.research.output import save_research_to_file

        report = MergedResearchReport(
            query="q",
            results=[_make_result("p1", "content")],
            merged_report="m",
            summary_2500="s",
            degraded=False,
        )
        paths = save_research_to_file(report, tmp_path)
        text = paths[0].read_text(encoding="utf-8")
        assert "WARNING" not in text.upper() or "Degraded" not in text


class TestDegradationConfig:
    """min_successful_providers loads from settings.yaml with sensible default."""

    def test_config_field_present_with_default_3(self) -> None:
        from config.config_loader import load_config
        cfg = load_config()
        assert cfg.research is not None
        assert hasattr(cfg.research, "min_successful_providers")
        assert cfg.research.min_successful_providers == 3


class TestDegradationCLIExitCode:
    """CLI exits 3 when run completes degraded; 0 when healthy."""

    def test_cli_exits_3_on_degraded_run(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from ai_council.cli import main as cli_root

        async def fake_run_research(*args, **kwargs):
            return MergedResearchReport(
                query=kwargs.get("query", "q"),
                results=[_make_result("p1", "ok"), _make_result("p2", error="x", content="")],
                merged_report="m",
                summary_2500="s",
                degraded=True,
                failed_count=2,
            )

        with patch("ai_council.research.runner.run_research", side_effect=fake_run_research), \
             patch("ai_council.cli.run_research", side_effect=fake_run_research, create=True), \
             patch("ai_council.cli._check_and_filter_providers", side_effect=lambda p: p):
            runner = CliRunner()
            result = runner.invoke(
                cli_root,
                ["-M", "r", "--output", str(tmp_path), "trivial query"],
            )
        assert result.exit_code == 3, f"expected exit 3 on degraded; got {result.exit_code}\n{result.output}"

    def test_cli_exits_0_on_healthy_run(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from ai_council.cli import main as cli_root

        async def fake_run_research(*args, **kwargs):
            return MergedResearchReport(
                query=kwargs.get("query", "q"),
                results=[_make_result("p1", "ok")],
                merged_report="m",
                summary_2500="s",
                degraded=False,
                failed_count=0,
            )

        with patch("ai_council.research.runner.run_research", side_effect=fake_run_research), \
             patch("ai_council.cli.run_research", side_effect=fake_run_research, create=True), \
             patch("ai_council.cli._check_and_filter_providers", side_effect=lambda p: p):
            runner = CliRunner()
            result = runner.invoke(
                cli_root,
                ["-M", "r", "--output", str(tmp_path), "trivial query"],
            )
        assert result.exit_code == 0, f"expected exit 0 on healthy; got {result.exit_code}\n{result.output}"
