"""Tests for src/synthesis.py."""

import time
from unittest.mock import AsyncMock

import pytest

from src.models import DebateResult, ModelResponse, Question, Round
from src.synthesis import _format_full_transcript, synthesize
from tests.conftest import MockProvider


def test_format_full_transcript():
    rounds = [
        Round(
            number=1,
            responses=[
                ModelResponse("gemini", "gemini-2.5-flash", 1, "Use YAML.", 1.0, 10),
                ModelResponse("claude", "claude-sonnet", 1, "Use JSON.", 1.1, 12),
            ],
        ),
        Round(
            number=2,
            responses=[
                ModelResponse("gemini", "gemini-2.5-flash", 2, "Changed mind: JSON.", 0.8, 8),
            ],
        ),
    ]
    transcript = _format_full_transcript(rounds)
    assert "### Round 1" in transcript
    assert "### Round 2" in transcript
    assert "Use YAML." in transcript
    assert "Changed mind: JSON." in transcript


async def test_synthesize_returns_debate_result(sample_prompts_config, sample_question, sample_round):
    synthesizer = MockProvider("claude", "## Consensus\nAll agreed on YAML.")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="claude",
            model="claude-sonnet",
            round_number=2,
            content="## Consensus\nAll agreed on YAML.",
            latency_sec=1.0,
            token_count=20,
        )
    )

    result = await synthesize(
        question=sample_question,
        rounds=[sample_round],
        synthesizer=synthesizer,
        prompts=sample_prompts_config,
        debate_start_time=time.monotonic() - 5.0,
    )

    assert isinstance(result, DebateResult)
    assert "Consensus" in result.synthesis
    assert result.synthesizer == "claude"
    assert result.total_duration_sec >= 5.0


async def test_synthesize_raises_on_empty_content(sample_prompts_config, sample_question, sample_round):
    synthesizer = MockProvider("claude", "")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="claude",
            model="claude-sonnet",
            round_number=2,
            content="",
            latency_sec=1.0,
            token_count=0,
        )
    )

    with pytest.raises(RuntimeError, match="empty content"):
        await synthesize(
            question=sample_question,
            rounds=[sample_round],
            synthesizer=synthesizer,
            prompts=sample_prompts_config,
            debate_start_time=time.monotonic(),
        )
