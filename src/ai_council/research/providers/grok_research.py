"""Grok research provider (xAI Responses API with x_search + web_search tools)."""

import asyncio
import logging
import time
from datetime import datetime

from openai import APIError, APITimeoutError, AsyncOpenAI

from ai_council.research.models import ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.x.ai/v1"
_SYSTEM_PROMPT = (
    "You are a research assistant with access to X (Twitter) and web search. "
    "Research the given question using both X discussions and web sources. "
    "Prioritize: practitioner experiences, developer opinions, real-world usage "
    "reports, trending discussions, and community sentiment. "
    "Include specific X posts or threads when relevant. "
    "Provide a comprehensive, well-structured report with citations."
)


class GrokResearchProvider(ResearchProvider):
    """Research via xAI Grok using x_search + web_search (Responses API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4.20-0309-reasoning",
        base_url: str = _BASE_URL,
        timeout_sec: int = 120,
        cost_per_1m_input: float = 3.00,
        cost_per_1m_output: float = 15.00,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout_sec = timeout_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output

    def name(self) -> str:
        return "grok"

    def model_string(self) -> str:
        return self._model

    async def research(self, query: str) -> ResearchResult:
        start = time.monotonic()
        timestamp = datetime.utcnow().isoformat()

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            response = await asyncio.wait_for(
                client.responses.create(
                    model=self._model,
                    input=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    tools=[
                        # openai 2.x SDK narrowed tool-param types (gpt-5.2 rollout); the
                        # dict literal no longer matches the ToolParam union. Stopgap ignore
                        # until the 2.x typing migration (BACKLOG #20). Runtime is unaffected.
                        {"type": "x_search"},  # type: ignore[misc, list-item]
                        {"type": "web_search"},
                    ],
                ),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ResearchProviderError(
                "grok", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except APITimeoutError as exc:
            raise ResearchProviderError("grok", f"API timeout: {exc}") from exc
        except APIError as exc:
            raise ResearchProviderError("grok", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        logger.debug("Grok raw response output: %r", getattr(response, "output", None))
        content = self._extract_content(response)
        sources = self._extract_sources(response)

        input_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        output_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        cost = (
            input_tokens / 1_000_000 * self._cost_per_1m_input
            + output_tokens / 1_000_000 * self._cost_per_1m_output
        )

        logger.debug("Grok research complete: %.1fs, %d sources", duration, len(sources))

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

    def _extract_content(self, response: object) -> str:
        """Extract text content from Responses API output."""
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
        """Extract citations from Responses API output items and content blocks."""
        sources: list[Source] = []
        seen: set[str] = set()
        output = getattr(response, "output", None)
        if not output or not isinstance(output, list):
            return sources
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type in ("x_search_call", "web_search_call"):
                continue
            # Annotations may be on the item directly or nested in content blocks
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
        # openai 2.x SDK types the annotations container as `object` (BACKLOG #20).
        for ann in annotations:  # type: ignore[attr-defined]
            url = getattr(ann, "url", None)
            title = getattr(ann, "title", None)
            if url and url not in seen:
                seen.add(url)
                sources.append(Source(title=str(title or url), url=str(url)))
