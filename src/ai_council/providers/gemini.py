"""Gemini provider using google-genai SDK with native async."""

import logging
from typing import Any

from google import genai
from google.genai import types as genai_types

from ai_council.providers.base import AIProvider, ProviderError, _Parsed

logger = logging.getLogger(__name__)


class GeminiProvider(AIProvider):
    """Google Gemini provider via google-genai SDK.

    No `_configure` override: the client is built per `_invoke` call so its internal async
    HTTP session is always bound to the active event loop. Caching a client across
    asyncio.run() boundaries (health check -> debate) causes "Event loop is closed" errors.
    """

    async def _invoke(self, prompt: str) -> Any:
        client = genai.Client(api_key=self._api_key)
        return await client.aio.models.generate_content(
            model=self._config.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=self._config.max_tokens,
            ),
        )

    def _parse(self, raw: Any) -> _Parsed:
        if not raw.text:
            raise ProviderError(self._config.name, "Empty response text")

        input_tokens: int | None = None
        output_tokens: int | None = None
        token_count: int | None = None
        if raw.usage_metadata:
            input_tokens = getattr(raw.usage_metadata, "prompt_token_count", None)
            output_tokens = getattr(raw.usage_metadata, "candidates_token_count", None)
            total = raw.usage_metadata.total_token_count
            token_count = total
            # Derive missing split from total if only one side is available
            if input_tokens is not None and output_tokens is None and total is not None:
                output_tokens = total - input_tokens
            elif output_tokens is not None and input_tokens is None and total is not None:
                input_tokens = total - output_tokens
        return _Parsed(
            content=raw.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_count=token_count,
        )
