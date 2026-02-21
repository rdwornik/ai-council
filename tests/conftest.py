"""Shared pytest fixtures."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.config_loader import AppConfig, DefaultsConfig, ModelConfig, PromptsConfig
from src.models import ModelResponse, Question, Round
from src.providers.base import AIProvider


@pytest.fixture
def sample_model_config() -> ModelConfig:
    return ModelConfig(
        name="test_model",
        sdk="test",
        model="test-model-1",
        api_key_env="TEST_API_KEY",
        timeout_sec=30,
        max_tokens=1024,
        base_url=None,
    )


@pytest.fixture
def sample_prompts_config() -> PromptsConfig:
    return PromptsConfig(
        initial="Answer this question: {question}",
        critique="Round {round}. Question: {question}\n\nOthers said:\n{previous_responses}\n\nCritique:",
        synthesis="Question: {question}\n\nTranscript ({rounds} rounds):\n{full_transcript}\n\nSynthesize:",
    )


@pytest.fixture
def sample_defaults_config(tmp_path: Path) -> DefaultsConfig:
    return DefaultsConfig(
        rounds=2,
        output_dir=tmp_path / "output",
        synthesizer="claude",
    )


@pytest.fixture
def sample_app_config(
    sample_defaults_config: DefaultsConfig,
    sample_prompts_config: PromptsConfig,
) -> AppConfig:
    model_cfg = ModelConfig(
        name="claude",
        sdk="anthropic",
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        timeout_sec=60,
        max_tokens=4096,
    )
    return AppConfig(
        defaults=sample_defaults_config,
        models={"claude": model_cfg},
        prompts=sample_prompts_config,
        available_providers={"claude"},
    )


@pytest.fixture
def sample_question() -> Question:
    return Question(text="Should we use YAML or JSON for config?", source="cli")


@pytest.fixture
def sample_response() -> ModelResponse:
    return ModelResponse(
        provider="claude",
        model="claude-sonnet-4-20250514",
        round_number=1,
        content="Use YAML for human-editable config, JSON for machine interchange.",
        latency_sec=1.5,
        token_count=42,
    )


@pytest.fixture
def sample_round(sample_response: ModelResponse) -> Round:
    return Round(number=1, responses=[sample_response])


class MockProvider(AIProvider):
    """Test double AIProvider."""

    def __init__(self, provider_name: str = "mock", response_content: str = "Mock response") -> None:
        self._name = provider_name
        self._response_content = response_content
        # Shadow the class method with an AsyncMock at the instance level.
        # ABC check passes because generate is defined in the class body below.
        self.generate = AsyncMock(  # type: ignore[assignment]
            return_value=ModelResponse(
                provider=provider_name,
                model="mock-model",
                round_number=1,
                content=response_content,
                latency_sec=0.1,
                token_count=10,
            )
        )

    def name(self) -> str:
        return self._name

    def model_string(self) -> str:
        return "mock-model"

    async def generate(self, prompt: str, round_number: int) -> ModelResponse:  # type: ignore[override]
        """Default implementation; replaced by AsyncMock in __init__."""
        return ModelResponse(
            provider=self._name,
            model="mock-model",
            round_number=round_number,
            content=self._response_content,
            latency_sec=0.1,
            token_count=10,
        )


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def two_mock_providers() -> list[MockProvider]:
    return [MockProvider("provider_a", "Response from A"), MockProvider("provider_b", "Response from B")]
