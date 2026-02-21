"""xAI Grok provider using openai SDK (OpenAI-compatible API)."""

import asyncio
import logging
import os
import time

from openai import AsyncOpenAI

from config.config_loader import ModelConfig
from src.models import ModelResponse
from src.providers.base import AIProvider, ProviderError

logger = logging.getLogger(__name__)


class XAIProvider(AIProvider):
    """xAI Grok provider via OpenAI-compatible API."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise ProviderError(config.name, f"Missing API key: {config.api_key_env}")
        if not config.base_url:
            raise ProviderError(config.name, "base_url is required for xAI provider")
        self._client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)

    def name(self) -> str:
        return self._config.name

    def model_string(self) -> str:
        return self._config.model

    async def generate(self, prompt: str, round_number: int) -> ModelResponse:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self._config.max_tokens,
                ),
                timeout=self._config.timeout_sec,
            )
        except TimeoutError as exc:
            raise ProviderError(self._config.name, f"Request timed out after {self._config.timeout_sec}s") from exc
        except Exception as exc:
            raise ProviderError(self._config.name, f"API call failed: {exc}") from exc

        latency = time.monotonic() - start

        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message.content:
            raise ProviderError(self._config.name, "Empty response content")

        token_count: int | None = None
        if response.usage:
            token_count = response.usage.total_tokens

        logger.info(
            "xAI round %d: %.2fs, %s tokens",
            round_number,
            latency,
            token_count,
        )

        return ModelResponse(
            provider=self._config.name,
            model=self._config.model,
            round_number=round_number,
            content=choice.message.content,
            latency_sec=latency,
            token_count=token_count,
        )
