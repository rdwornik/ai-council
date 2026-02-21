"""Gemini provider using google-genai SDK with native async."""

import asyncio
import logging
import os
import time

from google import genai
from google.genai import types as genai_types

from config.config_loader import ModelConfig
from src.models import ModelResponse
from src.providers.base import AIProvider, ProviderError

logger = logging.getLogger(__name__)


class GeminiProvider(AIProvider):
    """Google Gemini provider via google-genai SDK."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise ProviderError(config.name, f"Missing API key: {config.api_key_env}")
        self._client = genai.Client(api_key=api_key)

    def name(self) -> str:
        return self._config.name

    def model_string(self) -> str:
        return self._config.model

    async def generate(self, prompt: str, round_number: int) -> ModelResponse:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._config.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=self._config.max_tokens,
                    ),
                ),
                timeout=self._config.timeout_sec,
            )
        except TimeoutError as exc:
            raise ProviderError(self._config.name, f"Request timed out after {self._config.timeout_sec}s") from exc
        except Exception as exc:
            raise ProviderError(self._config.name, f"API call failed: {exc}") from exc

        latency = time.monotonic() - start

        if not response.text:
            raise ProviderError(self._config.name, "Empty response text")

        token_count: int | None = None
        if response.usage_metadata:
            token_count = response.usage_metadata.total_token_count

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
        )
