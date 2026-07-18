"""Gemini Deep Research provider via Interactions API.

Uses client.aio.interactions.create() with an autonomous deep-research agent
that browses the web, reads sources, and writes a cited report (~5-20 min).

The Interactions API is experimental as of google-genai 1.55+. Known agent IDs:
  - "deep-research-pro-preview-12-2025"  (configured in settings.yaml)
"""

import asyncio
import logging
import re
import time
import warnings

from google import genai

from ai_council.research.models import ResearchResult, Source
from ai_council.research.provider import ResearchProvider, ResearchProviderError, iso_now

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete"})
_MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^)\s]+)\)')


class GeminiResearchProvider(ResearchProvider):
    """Deep research via Gemini Interactions API — autonomous agent, real web browsing."""

    def __init__(
        self,
        api_key: str,
        agent: str = "deep-research-pro-preview-12-2025",
        timeout_sec: int = 1800,
        poll_interval_sec: int = 10,
        cost_per_1m_input: float = 0.0,
        cost_per_1m_output: float = 0.0,
    ) -> None:
        self._api_key = api_key
        self._agent = agent
        self._timeout_sec = timeout_sec
        self._poll_interval_sec = poll_interval_sec
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output

    def name(self) -> str:
        return "gemini"

    def model_string(self) -> str:
        return self._agent

    async def research(self, query: str) -> ResearchResult:
        start = time.monotonic()
        timestamp = iso_now()

        try:
            result = await asyncio.wait_for(
                self._run_research(query),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ResearchProviderError(
                "gemini", f"Timed out after {self._timeout_sec}s"
            ) from exc
        except ResearchProviderError:
            raise
        except Exception as exc:
            raise ResearchProviderError("gemini", f"API error: {exc}") from exc

        duration = time.monotonic() - start
        result.duration_sec = duration
        result.timestamp = timestamp
        return result

    async def _run_research(self, query: str) -> ResearchResult:
        # genai.Client() must be created here (inside async method) — not in __init__
        # The nextgen client it spawns binds to the running event loop (gotcha).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            client = genai.Client(api_key=self._api_key)

            interaction = await client.aio.interactions.create(
                agent=self._agent,
                input=query,
                background=True,
            )

        interaction_id = interaction.id
        logger.info("gemini: research started (interaction_id=%s, agent=%s)", interaction_id, self._agent)

        while True:
            await asyncio.sleep(self._poll_interval_sec)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                interaction = await client.aio.interactions.get(interaction_id)

            status = interaction.status
            logger.debug("gemini: poll status=%s (interaction_id=%s)", status, interaction_id)

            if status == "completed":
                break
            if status in _TERMINAL_STATUSES:
                raise ResearchProviderError(
                    "gemini",
                    f"Research {status} (interaction_id={interaction_id})",
                )

        content = self._extract_text(interaction)
        sources = self._extract_sources(interaction)

        input_tokens = 0
        output_tokens = 0
        usage = getattr(interaction, "usage", None)
        if usage:
            input_tokens = getattr(usage, "total_input_tokens", 0) or 0
            output_tokens = getattr(usage, "total_output_tokens", 0) or 0

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

    def _extract_text(self, interaction: object) -> str:
        outputs = getattr(interaction, "outputs", None) or []
        texts = []
        for output in outputs:
            text = getattr(output, "text", None)
            if text:
                texts.append(str(text))
        return "\n\n".join(texts)

    def _extract_sources(self, interaction: object) -> list[Source]:
        """Extract URLs from report text markdown links and structured result items."""
        sources: list[Source] = []
        seen: set[str] = set()
        try:
            outputs = getattr(interaction, "outputs", None) or []
            for output in outputs:
                # Parse markdown links from the report text
                text = getattr(output, "text", None)
                if text:
                    for title, url in _MARKDOWN_LINK_RE.findall(text):
                        if url not in seen:
                            seen.add(url)
                            sources.append(Source(title=title, url=url))
                # Also check structured URL result items
                result_list = getattr(output, "result", None)
                if isinstance(result_list, list):
                    for r in result_list:
                        url = getattr(r, "url", None)
                        if url and url not in seen:
                            seen.add(url)
                            sources.append(Source(title=str(url), url=str(url)))
        except Exception:
            logger.debug("gemini: could not extract sources", exc_info=True)
        return sources
