"""Anthropic Claude provider using anthropic SDK with native async."""

import logging
from typing import Any

import anthropic as anthropic_sdk

from ai_council.providers.base import AIProvider, ProviderError, _Parsed

logger = logging.getLogger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider via anthropic SDK.

    No `_configure` override: the client is built lazily per event loop inside `_invoke`, so its
    httpx connection pool is always bound to the running loop. See `AIProvider._client_for_loop`.
    """

    async def _invoke(self, prompt: str) -> Any:
        client = self._client_for_loop(
            lambda: anthropic_sdk.AsyncAnthropic(api_key=self._api_key)
        )
        return await client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

    def _parse(self, raw: Any) -> _Parsed:
        if not raw.content:
            return _Parsed("")  # base raises the generic "Empty response content"
        text_blocks = [b.text for b in raw.content if b.type == "text"]
        if not text_blocks:
            raise ProviderError(self._config.name, "No text blocks in response")
        content = "\n".join(text_blocks)

        input_tokens: int | None = None
        output_tokens: int | None = None
        token_count: int | None = None
        if raw.usage:
            input_tokens = raw.usage.input_tokens
            output_tokens = raw.usage.output_tokens
            token_count = input_tokens + output_tokens
        return _Parsed(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_count=token_count,
        )
