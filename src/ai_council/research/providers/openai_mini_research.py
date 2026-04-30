"""OpenAI o4-mini-deep-research provider (Responses API, background polling)."""

import asyncio
import logging
import time
from datetime import datetime

from openai import APIError, APITimeoutError, AsyncOpenAI

from ai_council.research.models import ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 15
_SYSTEM_PROMPT = (
    "You are a deep research assistant. Conduct thorough research on the given topic. "
    "Include specific facts, recent data, expert opinions, and cite your sources. "
    "Structure your report with an executive summary, key findings, detailed analysis, "
    "and a bibliography."
)


class OpenAIMiniResearchProvider(ResearchProvider):
    """Research via OpenAI o4-mini-deep-research (Responses API with background polling)."""

    def __init__(
        self,
        api_key: str,
        model: str = "o4-mini-deep-research",
        timeout_sec: int = 1200,  # 20 minutes
        cost_per_1m_input: float = 2.00,
        cost_per_1m_output: float = 8.00,
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
            result = await asyncio.wait_for(
                self._run_with_polling(client, query),
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
        result.duration_sec = duration
        result.timestamp = timestamp
        return result

    async def _run_with_polling(self, client: AsyncOpenAI, query: str) -> ResearchResult:
        """Submit background research job and poll until complete."""
        # Submit the research job in background mode
        # Deep research models require at least one search tool
        response = await client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=[{"type": "web_search_preview"}],
            background=True,
        )
        response_id = response.id
        logger.debug("openai_mini: submitted background job %s", response_id)

        # Poll until complete
        while response.status not in ("completed", "failed", "cancelled"):
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            response = await client.responses.retrieve(response_id)
            elapsed = time.monotonic()
            logger.debug("openai_mini: job %s status=%s (%.0fs elapsed)", response_id, response.status, elapsed)

        if response.status != "completed":
            raise ResearchProviderError(
                "openai_mini", f"Job {response_id} ended with status: {response.status}"
            )

        # Extract content and sources from completed response
        content = self._extract_content(response)
        sources = self._extract_sources(response)

        input_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        output_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        cost = (
            input_tokens / 1_000_000 * self._cost_per_1m_input
            + output_tokens / 1_000_000 * self._cost_per_1m_output
        )

        return ResearchResult(
            provider=self.name(),
            query=query,
            content=content,
            sources=sources,
            token_count=input_tokens + output_tokens,
            cost_usd=cost,
        )

    def _extract_content(self, response: object) -> str:
        """Extract text content from a completed Responses API response."""
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
        """Extract web search citations from a Responses API response."""
        sources: list[Source] = []
        output = getattr(response, "output", None)
        if not output or not isinstance(output, list):
            return sources
        for item in output:
            # web_search_call items have result with citations
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                continue
            annotations = getattr(item, "annotations", None)
            if annotations:
                for ann in annotations:
                    url = getattr(ann, "url", None)
                    title = getattr(ann, "title", None)
                    if url:
                        sources.append(Source(title=str(title or url), url=str(url)))
        return sources
