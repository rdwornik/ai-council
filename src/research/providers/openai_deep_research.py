"""OpenAI o3-deep-research provider (Responses API, background polling, --deep only)."""

import asyncio
import logging
import time
from datetime import datetime

from openai import AsyncOpenAI, APIError, APITimeoutError

from src.research.models import ResearchResult, Source
from src.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 20
_SYSTEM_PROMPT = (
    "You are an expert deep research analyst. Conduct exhaustive research on the given "
    "topic. Synthesize information from multiple sources, identify expert consensus and "
    "dissenting views, and provide a comprehensive analysis with full citations. "
    "Structure your report with executive summary, key findings, detailed analysis by "
    "subtopic, methodology notes, and a complete bibliography."
)


class OpenAIDeepResearchProvider(ResearchProvider):
    """Research via OpenAI o3-deep-research (Responses API, --deep flag only)."""

    def __init__(
        self,
        api_key: str,
        model: str = "o3-deep-research",
        timeout_sec: int = 2700,  # 45 minutes
        cost_per_1m_input: float = 10.00,
        cost_per_1m_output: float = 40.00,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output

    def name(self) -> str:
        return "openai_deep"

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
                "openai_deep", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except APITimeoutError as exc:
            raise ResearchProviderError("openai_deep", f"API timeout: {exc}") from exc
        except APIError as exc:
            raise ResearchProviderError("openai_deep", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        result.duration_sec = duration
        result.timestamp = timestamp
        return result

    async def _run_with_polling(self, client: AsyncOpenAI, query: str) -> ResearchResult:
        """Submit background research job and poll until complete."""
        response = await client.responses.create(  # type: ignore[attr-defined]
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            background=True,
        )
        response_id = response.id
        logger.debug("openai_deep: submitted background job %s", response_id)

        while response.status not in ("completed", "failed", "cancelled"):
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            response = await client.responses.retrieve(response_id)  # type: ignore[attr-defined]
            logger.debug("openai_deep: job %s status=%s", response_id, response.status)

        if response.status != "completed":
            raise ResearchProviderError(
                "openai_deep", f"Job {response_id} ended with status: {response.status}"
            )

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
        output = getattr(response, "output", None)
        if not output or not isinstance(output, list):
            return sources
        for item in output:
            annotations = getattr(item, "annotations", None)
            if annotations:
                for ann in annotations:
                    url = getattr(ann, "url", None)
                    title = getattr(ann, "title", None)
                    if url:
                        sources.append(Source(title=str(title or url), url=str(url)))
        return sources
