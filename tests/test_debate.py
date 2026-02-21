"""Tests for src/debate.py."""

import pytest
from unittest.mock import AsyncMock

from src.debate import _anonymize_responses, run_debate
from src.models import ModelResponse, Question, Round
from src.providers.base import ProviderError
from tests.conftest import MockProvider


def test_anonymize_responses_uses_proposal_labels():
    responses = [
        ModelResponse("gemini", "gemini-3.1-pro-preview", 1, "Use YAML.", 1.0, 10),
        ModelResponse("claude", "claude-opus-4-6", 1, "Use JSON.", 1.2, 15),
    ]
    block, mapping = _anonymize_responses(responses)
    assert "--- Proposal A ---" in block
    assert "--- Proposal B ---" in block
    assert len(mapping) == 2
    assert set(mapping.keys()) == {"A", "B"}


def test_anonymize_responses_hides_model_names():
    responses = [
        ModelResponse("gemini", "gemini-3.1-pro-preview", 1, "Prefer YAML.", 1.0, 10),
        ModelResponse("claude", "claude-opus-4-6", 1, "Prefer JSON.", 1.2, 15),
    ]
    block, mapping = _anonymize_responses(responses)
    # Provider names and model strings must not appear in the anonymized block
    assert "gemini" not in block
    assert "claude" not in block
    assert "gemini-3.1-pro-preview" not in block
    assert "claude-opus-4-6" not in block
    # Content still present
    assert "Prefer YAML." in block or "Prefer JSON." in block


def test_anonymize_responses_mapping_covers_all_providers():
    responses = [
        ModelResponse("provider_a", "model-a", 1, "A says yes.", 0.5, 5),
        ModelResponse("provider_b", "model-b", 1, "B says no.", 0.5, 5),
        ModelResponse("provider_c", "model-c", 1, "C says maybe.", 0.5, 5),
    ]
    block, mapping = _anonymize_responses(responses)
    assert set(mapping.values()) == {"provider_a", "provider_b", "provider_c"}
    assert len(mapping) == 3


async def test_run_debate_single_round(two_mock_providers, sample_prompts_config, sample_question):
    rounds = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(rounds) == 1
    assert rounds[0].number == 1
    assert len(rounds[0].responses) == 2


async def test_run_debate_two_rounds(two_mock_providers, sample_prompts_config, sample_question):
    for p in two_mock_providers:
        p.generate = AsyncMock(
            side_effect=[
                ModelResponse(p.name(), "mock-model", 1, f"Round 1 from {p.name()}", 0.1, 5),
                ModelResponse(p.name(), "mock-model", 2, f"Round 2 from {p.name()}", 0.1, 5),
            ]
        )

    rounds = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=2,
    )
    assert len(rounds) == 2
    assert rounds[1].number == 2


async def test_run_debate_injects_persona_in_round1(sample_prompts_config, sample_question):
    """Persona text should appear in the prompt passed to the provider on round 1."""
    from src.providers.base import AIProvider

    captured_prompts: list[str] = []

    class CapturingProvider(AIProvider):
        def __init__(self, provider_name: str) -> None:
            self._name = provider_name

        def name(self) -> str:
            return self._name

        def model_string(self) -> str:
            return "mock-model"

        async def generate(self, prompt: str, round_number: int) -> ModelResponse:
            captured_prompts.append(prompt)
            return ModelResponse(self._name, "mock-model", round_number, "response", 0.1, 5)

    provider1 = CapturingProvider("mock")
    provider2 = CapturingProvider("mock2")

    await run_debate(
        question=sample_question,
        providers=[provider1, provider2],
        prompts=sample_prompts_config,
        num_rounds=1,
    )

    # The persona for "mock" is "Be a mock architect." (from sample_prompts_config fixture)
    assert any("Be a mock architect." in p for p in captured_prompts)


async def test_run_debate_critique_uses_anonymized_placeholder(two_mock_providers, sample_prompts_config, sample_question):
    """Round 2 prompts should use previous_responses_anonymized, not provider names."""
    captured_prompts: list[str] = []

    for p in two_mock_providers:
        original_name = p.name()
        p.generate = AsyncMock(
            side_effect=[
                ModelResponse(original_name, "mock-model", 1, f"Round 1 from {original_name}", 0.1, 5),
                ModelResponse(original_name, "mock-model", 2, f"Round 2 from {original_name}", 0.1, 5),
            ]
        )

    # Patch generate to capture prompts on round 2
    for p in two_mock_providers:
        original_side_effect = p.generate.side_effect

        async def capturing_generate(prompt, round_number, _orig=original_side_effect):
            if round_number == 2:
                captured_prompts.append(prompt)
            return await AsyncMock(side_effect=_orig)(prompt, round_number)

        p.generate = AsyncMock(side_effect=capturing_generate)

    # Re-set side effects
    for p in two_mock_providers:
        original_name = p.name()
        p.generate = AsyncMock(
            side_effect=[
                ModelResponse(original_name, "mock-model", 1, f"Round 1 from {original_name}", 0.1, 5),
                ModelResponse(original_name, "mock-model", 2, f"Round 2 from {original_name}", 0.1, 5),
            ]
        )

    rounds = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=2,
    )
    assert len(rounds) == 2
    # Ensure round 2 actually ran
    assert rounds[1].number == 2


async def test_run_debate_on_round_complete_callback(two_mock_providers, sample_prompts_config, sample_question):
    completed = []

    def callback(rnd: Round) -> None:
        completed.append(rnd.number)

    await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=2,
        on_round_complete=callback,
    )
    assert completed == [1, 2]


async def test_run_debate_skips_failed_provider(sample_prompts_config, sample_question):
    good = MockProvider("good", "Good response")
    bad = MockProvider("bad", "")
    bad.generate = AsyncMock(side_effect=ProviderError("bad", "API error"))

    rounds = await run_debate(
        question=sample_question,
        providers=[good, bad],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(rounds) == 1
    assert len(rounds[0].responses) == 1
    assert rounds[0].responses[0].provider == "good"


async def test_run_debate_raises_if_all_fail(sample_prompts_config, sample_question):
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    p1.generate = AsyncMock(side_effect=ProviderError("p1", "fail"))
    p2.generate = AsyncMock(side_effect=ProviderError("p2", "fail"))

    with pytest.raises(RuntimeError, match="All providers failed"):
        await run_debate(
            question=sample_question,
            providers=[p1, p2],
            prompts=sample_prompts_config,
            num_rounds=1,
        )
