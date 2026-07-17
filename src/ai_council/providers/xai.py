"""xAI Grok provider using openai SDK (OpenAI-compatible API)."""

import logging
from typing import Any

from openai import AsyncOpenAI

from ai_council.providers.base import AIProvider, ProviderError, _Parsed, parse_openai_chat

logger = logging.getLogger(__name__)


class XAIProvider(AIProvider):
    """xAI Grok provider via OpenAI-compatible API."""

    def _configure(self) -> None:
        if not self._config.base_url:
            raise ProviderError(self._config.name, "base_url is required for xAI provider")
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._config.base_url)

    async def _invoke(self, prompt: str) -> Any:
        return await self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._config.max_tokens,
        )

    def _parse(self, raw: Any) -> _Parsed:
        return parse_openai_chat(raw)
