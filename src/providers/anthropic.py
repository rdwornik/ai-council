"""Anthropic Claude provider using anthropic SDK with native async."""

import asyncio
import logging
import os
import time

import anthropic as anthropic_sdk

from config.config_loader import ModelConfig
from src.models import ModelResponse
from src.providers.base import AIProvider, ProviderError

logger = logging.getLogger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider via anthropic SDK."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise ProviderError(config.name, f"Missing API key: {config.api_key_env}")
        self._client = anthropic_sdk.AsyncAnthropic(api_key=api_key)

    def name(self) -> str:
        return self._config.name

    def model_string(self) -> str:
        return self._config.model

    async def generate(self, prompt: str, round_number: int) -> ModelResponse:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._config.model,
                    max_tokens=self._config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._config.timeout_sec,
            )
        except TimeoutError as exc:
            raise ProviderError(self._config.name, f"Request timed out after {self._config.timeout_sec}s") from exc
        except Exception as exc:
            raise ProviderError(self._config.name, f"API call failed: {exc}") from exc

        latency = time.monotonic() - start

        if not response.content:
            raise ProviderError(self._config.name, "Empty response content")

        text_blocks = [b.text for b in response.content if b.type == "text"]
        if not text_blocks:
            raise ProviderError(self._config.name, "No text blocks in response")

        content = "\n".join(text_blocks)

        token_count: int | None = None
        if response.usage:
            token_count = response.usage.input_tokens + response.usage.output_tokens

        logger.info(
            "Anthropic round %d: %.2fs, %s tokens",
            round_number,
            latency,
            token_count,
        )

        return ModelResponse(
            provider=self._config.name,
            model=self._config.model,
            round_number=round_number,
            content=content,
            latency_sec=latency,
            token_count=token_count,
        )
