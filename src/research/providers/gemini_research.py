"""Gemini research provider with Google Search grounding."""

# NOTE: Google's "Gemini Deep Research" (the consumer feature) uses an "Interactions API"
# that is not yet exposed in the google-genai SDK as of early 2026. This implementation
# uses the Gemini API with Google Search grounding, which is the closest available API
# equivalent. It enables web search and returns grounded responses with citations.
# When the Interactions API becomes available, this provider can be upgraded.

import asyncio
import logging
import time
from datetime import datetime

from src.research.models import ResearchResult, Source
from src.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_TIMEOUT_DEFAULT = 1800  # 30 minutes
_SYSTEM_PROMPT = (
    "You are a research analyst with access to current web search. "
    "Conduct comprehensive research on the given topic. "
    "Include specific facts, recent data, expert opinions, and cite all sources. "
    "Structure your report with: Executive Summary, Key Findings (numbered), "
    "Detailed Analysis, Competing Perspectives, and Sources."
)


class GeminiResearchProvider(ResearchProvider):
    """Research via Gemini with Google Search grounding."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro-preview-05-06",
        timeout_sec: int = _TIMEOUT_DEFAULT,
        cost_per_1m_input: float = 0.0,
        cost_per_1m_output: float = 0.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output

    def name(self) -> str:
        return "gemini"

    def model_string(self) -> str:
        return self._model

    async def research(self, query: str) -> ResearchResult:
        start = time.monotonic()
        timestamp = datetime.utcnow().isoformat()

        try:
            result = await asyncio.wait_for(
                self._run_research(query),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ResearchProviderError(
                "gemini", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except Exception as exc:
            raise ResearchProviderError("gemini", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        result.duration_sec = duration
        result.timestamp = timestamp
        return result

    async def _run_research(self, query: str) -> ResearchResult:
        # Import inside method: google-genai Client must be created per event loop (see gotchas)
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

        full_prompt = f"{_SYSTEM_PROMPT}\n\nResearch topic: {query}"

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            ),
        )

        content = response.text or ""
        sources = self._extract_grounding_sources(response)

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

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

    def _extract_grounding_sources(self, response: object) -> list[Source]:
        """Extract grounding search entry point citations from Gemini response."""
        sources: list[Source] = []
        try:
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return sources
            for candidate in candidates:
                grounding = getattr(candidate, "grounding_metadata", None)
                if not grounding:
                    continue
                chunks = getattr(grounding, "grounding_chunks", None)
                if chunks:
                    for chunk in chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", None)
                            title = getattr(web, "title", None)
                            if uri:
                                sources.append(Source(
                                    title=str(title or uri),
                                    url=str(uri),
                                ))
        except Exception:
            logger.debug("gemini: could not extract grounding sources", exc_info=True)
        return sources
