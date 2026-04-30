"""Perplexity research provider (sonar-pro, OpenAI-compatible API)."""

import asyncio
import logging
import time
from datetime import datetime

from openai import APIError, APITimeoutError, AsyncOpenAI

from ai_council.research.models import ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.perplexity.ai"
_SYSTEM_PROMPT = (
    "You are a research assistant. Provide a comprehensive, well-structured research "
    "report on the topic. Include specific facts, data points, and cite your sources "
    "inline. Structure your response with clear sections and a sources list at the end."
)


class PerplexityProvider(ResearchProvider):
    """Research via Perplexity sonar-pro with web citations."""

    def __init__(
        self,
        api_key: str,
        model: str = "sonar-pro",
        timeout_sec: int = 60,
        cost_per_1m_input: float = 3.00,
        cost_per_1m_output: float = 15.00,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output

    def name(self) -> str:
        return "perplexity"

    def model_string(self) -> str:
        return self._model

    async def research(self, query: str) -> ResearchResult:
        start = time.monotonic()
        timestamp = datetime.utcnow().isoformat()

        client = AsyncOpenAI(api_key=self._api_key, base_url=_BASE_URL)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                ),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ResearchProviderError(
                "perplexity", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except APITimeoutError as exc:
            raise ResearchProviderError("perplexity", f"API timeout: {exc}") from exc
        except APIError as exc:
            raise ResearchProviderError("perplexity", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract citations from Perplexity response metadata
        sources: list[Source] = []
        citations = getattr(response, "citations", None)
        if citations:
            for i, url in enumerate(citations):
                sources.append(Source(title=f"Source {i + 1}", url=str(url)))

        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        cost = (
            input_tokens / 1_000_000 * self._cost_per_1m_input
            + output_tokens / 1_000_000 * self._cost_per_1m_output
        )

        logger.debug("Perplexity research complete: %.1fs, %d sources", duration, len(sources))

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
