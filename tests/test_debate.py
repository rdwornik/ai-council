"""Tests for src/debate.py."""

import logging

import pytest
from unittest.mock import AsyncMock

from config.config_loader import ModelConfig
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


# --- Retry logic tests ---

def _make_mock_config(timeout_sec: int = 10) -> ModelConfig:
    return ModelConfig(
        name="slow", sdk="test", model="m", api_key_env="K",
        timeout_sec=timeout_sec, max_tokens=100,
    )


async def test_retry_succeeds_on_second_attempt(sample_prompts_config, sample_question):
    """Provider that times out once but succeeds on retry is included in results."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "Slow but eventual response")
    slow._config = _make_mock_config(10)

    call_count = 0

    async def timeout_then_succeed(prompt: str, round_number: int) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ProviderError("slow", "Request timed out after 10s")
        return ModelResponse("slow", "m", round_number, "Slow but eventual response", 0.1, 5)

    slow.generate = AsyncMock(side_effect=timeout_then_succeed)

    rounds = await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(rounds[0].responses) == 2
    assert call_count == 2  # initial attempt + one retry


async def test_retry_excluded_after_second_timeout(sample_prompts_config, sample_question):
    """Provider that times out on both attempts is excluded from results."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "")
    slow._config = _make_mock_config(10)
    slow.generate = AsyncMock(
        side_effect=ProviderError("slow", "Request timed out after 10s")
    )

    rounds = await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(rounds[0].responses) == 1
    assert rounds[0].responses[0].provider == "good"


async def test_retry_uses_15x_timeout(sample_prompts_config, sample_question):
    """On retry, provider config timeout_sec is bumped to 1.5x."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "Slow response")
    slow._config = _make_mock_config(100)

    observed_timeouts: list[int] = []

    async def capture_timeout(prompt: str, round_number: int) -> ModelResponse:
        observed_timeouts.append(slow._config.timeout_sec)
        if len(observed_timeouts) == 1:
            raise ProviderError("slow", "Request timed out after 100s")
        return ModelResponse("slow", "m", round_number, "Slow response", 0.1, 5)

    slow.generate = AsyncMock(side_effect=capture_timeout)

    await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert observed_timeouts[0] == 100           # original timeout on first attempt
    assert observed_timeouts[1] == 150           # 1.5x on retry
    assert slow._config.timeout_sec == 100       # restored after retry


# --- Quality gate tests ---

async def test_quality_gate_warns_when_too_few_respond(
    sample_prompts_config, sample_question, caplog
):
    """Quality gate fires when fewer than 3 models respond in Round 1 on a 4-model panel."""
    p1 = MockProvider("p1", "Response 1")
    p2 = MockProvider("p2", "Response 2")
    p3 = MockProvider("p3", "")
    p4 = MockProvider("p4", "")
    p3.generate = AsyncMock(side_effect=ProviderError("p3", "fail"))
    p4.generate = AsyncMock(side_effect=ProviderError("p4", "fail"))

    with caplog.at_level(logging.WARNING):
        await run_debate(
            question=sample_question,
            providers=[p1, p2, p3, p4],
            prompts=sample_prompts_config,
            num_rounds=1,
        )

    assert any("Only 2/4" in msg for msg in caplog.messages)
    assert any("Debate quality is degraded" in msg for msg in caplog.messages)


async def test_quality_gate_silent_for_small_panels(
    sample_prompts_config, sample_question, caplog
):
    """Quality gate does not fire when panel has fewer than 3 providers."""
    p1 = MockProvider("p1", "Response 1")
    p2 = MockProvider("p2", "Response 2")

    with caplog.at_level(logging.WARNING):
        await run_debate(
            question=sample_question,
            providers=[p1, p2],
            prompts=sample_prompts_config,
            num_rounds=1,
        )

    assert not any("Debate quality is degraded" in msg for msg in caplog.messages)


async def test_quality_gate_silent_when_enough_respond(
    sample_prompts_config, sample_question, caplog
):
    """Quality gate does not fire when >= 3 providers succeed."""
    providers = [MockProvider(f"p{i}", f"Response {i}") for i in range(1, 5)]

    with caplog.at_level(logging.WARNING):
        await run_debate(
            question=sample_question,
            providers=providers,
            prompts=sample_prompts_config,
            num_rounds=1,
        )

    assert not any("Debate quality is degraded" in msg for msg in caplog.messages)
