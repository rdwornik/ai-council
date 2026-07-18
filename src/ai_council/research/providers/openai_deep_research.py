"""OpenAI gpt-5.5 deep research provider (Responses API + web_search, reasoning=high).

Migrated 2026-05-18 off the deprecated `o3-deep-research` model onto gpt-5.5 with
the agentic `web_search` tool and high reasoning effort. Gated behind `--deep`.
"""

import asyncio
import logging
import time

from openai import APIError, APITimeoutError, AsyncOpenAI

from ai_council.research.models import ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError, iso_now

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert deep research analyst with web search. Conduct exhaustive "
    "research on the given topic. Synthesize information from multiple sources, "
    "identify expert consensus and dissenting views, and provide a comprehensive "
    "analysis with full citations. Structure your report with executive summary, "
    "key findings, detailed analysis by subtopic, methodology notes, and a "
    "complete bibliography."
)


class OpenAIDeepResearchProvider(ResearchProvider):
    """Research via OpenAI gpt-5.5 + web_search + high reasoning (Responses API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.5",
        timeout_sec: int = 1800,
        cost_per_1m_input: float = 5.00,
        cost_per_1m_output: float = 30.00,
        reasoning_effort: str = "high",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output
        self._reasoning_effort = reasoning_effort

    def name(self) -> str:
        return "openai_deep"

    def model_string(self) -> str:
        return self._model

    async def research(self, query: str) -> ResearchResult:
        start = time.monotonic()
        timestamp = iso_now()

        client = AsyncOpenAI(
            api_key=self._api_key,
            timeout=float(self._timeout_sec),
            max_retries=1,
        )
        try:
            response = await asyncio.wait_for(
                self._call(client, query),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ResearchProviderError(
                "openai_deep", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except APITimeoutError as exc:
            raise ResearchProviderError("openai_deep", f"API timeout: {exc}") from exc
        except APIError as exc:
            raise ResearchProviderError("openai_deep", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        content = self._extract_content(response)
        sources = self._extract_sources(response)

        input_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        output_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        cost = (
            input_tokens / 1_000_000 * self._cost_per_1m_input
            + output_tokens / 1_000_000 * self._cost_per_1m_output
        )

        logger.debug("openai_deep research complete: %.1fs, %d sources", duration, len(sources))
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

    async def _call(self, client: AsyncOpenAI, query: str):
        # openai 2.x SDK narrowed responses.create overloads (gpt-5.2 rollout); the
        # existing kwargs no longer match a single overload. Stopgap until the 2.x
        # typing migration (BACKLOG #20). Runtime is unaffected.
        return await client.responses.create(  # type: ignore[call-overload]
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=[{"type": "web_search"}],
            reasoning={"effort": self._reasoning_effort},
        )

    def _extract_content(self, response: object) -> str:
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
        # openai 2.x SDK types the annotations container as `object` (BACKLOG #20).
        for ann in annotations:  # type: ignore[attr-defined]
            url = getattr(ann, "url", None)
            title = getattr(ann, "title", None)
            if url and url not in seen:
                seen.add(url)
                sources.append(Source(title=str(title or url), url=str(url)))
