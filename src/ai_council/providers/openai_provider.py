"""OpenAI provider using openai SDK with native async."""

import logging
from typing import Any

from openai import AsyncOpenAI

from ai_council.providers.base import AIProvider, _Parsed, parse_openai_chat

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """OpenAI provider via openai SDK.

    No `_configure` override: the client is built lazily per event loop inside `_invoke`, so its
    httpx connection pool is always bound to the running loop. See `AIProvider._client_for_loop`.
    """

    async def _invoke(self, prompt: str) -> Any:
        client = self._client_for_loop(lambda: AsyncOpenAI(api_key=self._api_key))
        return await client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self._config.max_tokens,
        )

    def _parse(self, raw: Any) -> _Parsed:
        return parse_openai_chat(raw)
