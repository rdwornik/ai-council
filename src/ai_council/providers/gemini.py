"""Gemini provider using google-genai SDK with native async."""

import asyncio
import logging
import os
import time

from google import genai
from google.genai import types as genai_types

from ai_council.models import ModelResponse
from ai_council.providers.base import AIProvider, ProviderError
from config.config_loader import ModelConfig

logger = logging.getLogger(__name__)


class GeminiProvider(AIProvider):
    """Google Gemini provider via google-genai SDK."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise ProviderError(config.name, f"Missing API key: {config.api_key_env}")
        # Store key only; client is created per generate() call so its internal async
        # HTTP session is always bound to the active event loop. Caching the client
        # across asyncio.run() boundaries (e.g. health check → debate) causes
        # "Event loop is closed" errors because the session is tied to the first loop.
        self._api_key = api_key

    def name(self) -> str:
        return self._config.name

    def model_string(self) -> str:
        return self._config.model

    async def generate(self, prompt: str, round_number: int) -> ModelResponse:
        start = time.monotonic()
        client = genai.Client(api_key=self._api_key)
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self._config.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=self._config.max_tokens,
                    ),
                ),
                timeout=self._config.timeout_sec,
            )
        except TimeoutError as exc:
            raise ProviderError(
                self._config.name,
                f"Request timed out after {self._config.timeout_sec}s",
            ) from exc
        except Exception as exc:
            raise ProviderError(self._config.name, f"API call failed: {exc}") from exc

        latency = time.monotonic() - start

        if not response.text:
            raise ProviderError(self._config.name, "Empty response text")

        input_tokens: int | None = None
        output_tokens: int | None = None
        token_count: int | None = None
        if response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            output_tokens = getattr(
                response.usage_metadata, "candidates_token_count", None
            )
            total = response.usage_metadata.total_token_count
            token_count = total
            # Derive missing split from total if only one side is available
            if input_tokens is not None and output_tokens is None and total is not None:
                output_tokens = total - input_tokens
            elif output_tokens is not None and input_tokens is None and total is not None:
                input_tokens = total - output_tokens

        logger.info(
            "Gemini round %d: %.2fs, %s tokens",
            round_number,
            latency,
            token_count,
        )

        return ModelResponse(
            provider=self._config.name,
            model=self._config.model,
            round_number=round_number,
            content=response.text,
            latency_sec=latency,
            token_count=token_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
