"""DeepSeek provider using openai SDK (OpenAI-compatible API)."""

import logging
from typing import Any

from openai import AsyncOpenAI

from ai_council.providers.base import AIProvider, ProviderError, _Parsed, parse_openai_chat

logger = logging.getLogger(__name__)


class DeepSeekProvider(AIProvider):
    """DeepSeek provider via OpenAI-compatible API."""

    def _configure(self) -> None:
        # Config validation only — a missing base_url must still fail fast at pool-build time.
        # The client itself is built lazily per event loop in _invoke (see _client_for_loop).
        if not self._config.base_url:
            raise ProviderError(self._config.name, "base_url is required for DeepSeek provider")

    async def _invoke(self, prompt: str) -> Any:
        client = self._client_for_loop(
            lambda: AsyncOpenAI(api_key=self._api_key, base_url=self._config.base_url)
        )
        return await client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._config.max_tokens,
        )

    def _parse(self, raw: Any) -> _Parsed:
        return parse_openai_chat(raw)
