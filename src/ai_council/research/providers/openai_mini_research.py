"""OpenAI gpt-5.4-mini research provider (Responses API + web_search tool).

Migrated 2026-05-18 off the deprecated `o4-mini-deep-research` deep-research model
onto gpt-5.4-mini with the agentic `web_search` server-side tool. The legacy
provider used a `background=True` + poll loop; the migrated path is synchronous
(`responses.create` returns the completed response in one call).
"""

import asyncio
import logging
import time
from datetime import datetime

from openai import APIError, APITimeoutError, AsyncOpenAI

from ai_council.research.models import ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research assistant with web search. Conduct thorough research on "
    "the given topic. Include specific facts, recent data, expert opinions, and "
    "cite your sources. Structure your report with an executive summary, key "
    "findings, detailed analysis, and a bibliography."
)


class OpenAIMiniResearchProvider(ResearchProvider):
    """Research via OpenAI gpt-5.4-mini + web_search (Responses API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-mini",
        timeout_sec: int = 300,
        cost_per_1m_input: float = 0.75,
        cost_per_1m_output: float = 4.50,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output

    def name(self) -> str:
        return "openai_mini"

    def model_string(self) -> str:
        return self._model

    async def research(self, query: str) -> ResearchResult:
        start = time.monotonic()
        timestamp = datetime.utcnow().isoformat()

        client = AsyncOpenAI(api_key=self._api_key)
        try:
            response = await asyncio.wait_for(
                self._call_with_retry(client, query),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ResearchProviderError(
                "openai_mini", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except APITimeoutError as exc:
            raise ResearchProviderError("openai_mini", f"API timeout: {exc}") from exc
        except APIError as exc:
            raise ResearchProviderError("openai_mini", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        content = self._extract_content(response)
        sources = self._extract_sources(response)

        input_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        output_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        cost = (
            input_tokens / 1_000_000 * self._cost_per_1m_input
            + output_tokens / 1_000_000 * self._cost_per_1m_output
        )

        logger.debug("openai_mini research complete: %.1fs, %d sources", duration, len(sources))
        return ResearchResult(
            provider=self.name(),
            query=query,
            content=content,
            sources=sources,
            token_count=input_tokens + output_tokens,
            cost_usd=cost,
            duration_sec=duration,
            timestamp=timestamp,
        )

    async def _call_with_retry(self, client: AsyncOpenAI, query: str):
        """Single-shot retry around a transient APIError/APITimeoutError."""
        try:
            return await self._call(client, query)
        except (APIError, APITimeoutError) as exc:
            logger.warning("openai_mini: transient failure (%s); retrying once", exc)
            return await self._call(client, query)

    async def _call(self, client: AsyncOpenAI, query: str):
        return await client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=[{"type": "web_search"}],
        )

    def _extract_content(self, response: object) -> str:
        """Extract text content from Responses API output (message items + content blocks)."""
        output = getattr(response, "output", None)
        if not output:
            return ""
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None)
                        if text:
                            parts.append(str(text))
                elif isinstance(content, str):
                    parts.append(content)
                text_attr = getattr(item, "text", None)
                if text_attr and isinstance(text_attr, str):
                    parts.append(text_attr)
            return "\n\n".join(p for p in parts if p)
        return str(output)

    def _extract_sources(self, response: object) -> list[Source]:
        """Extract web_search citations — annotations on content blocks (and item fallback)."""
        sources: list[Source] = []
        seen: set[str] = set()
        output = getattr(response, "output", None)
        if not output or not isinstance(output, list):
            return sources
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                continue
            self._collect_annotations(getattr(item, "annotations", None), sources, seen)
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    self._collect_annotations(getattr(block, "annotations", None), sources, seen)
        return sources

    def _collect_annotations(
        self, annotations: object, sources: list[Source], seen: set[str]
    ) -> None:
        if not annotations:
            return
        for ann in annotations:
            url = getattr(ann, "url", None)
            title = getattr(ann, "title", None)
            if url and url not in seen:
                seen.add(url)
                sources.append(Source(title=str(title or url), url=str(url)))
