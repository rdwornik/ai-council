"""Tests for src/debate.py."""

import pytest
from unittest.mock import AsyncMock

from src.debate import _format_previous_responses, run_debate
from src.models import ModelResponse, Question, Round
from src.providers.base import ProviderError
from tests.conftest import MockProvider


def test_format_previous_responses():
    responses = [
        ModelResponse("gemini", "gemini-2.5-flash", 1, "Use YAML.", 1.0, 10),
        ModelResponse("claude", "claude-sonnet", 1, "Use JSON.", 1.2, 15),
    ]
    result = _format_previous_responses(responses)
    assert "--- gemini (gemini-2.5-flash) ---" in result
    assert "Use YAML." in result
    assert "--- claude (claude-sonnet) ---" in result
    assert "Use JSON." in result


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
    # Each mock needs to return a valid response on each call
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
