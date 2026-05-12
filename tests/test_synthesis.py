"""Tests for src/synthesis.py."""

import time
from unittest.mock import AsyncMock

import pytest

from ai_council.models import DebateResult, ModelResponse, Round
from ai_council.providers.base import ProviderError
from ai_council.synthesis import _format_full_transcript, synthesize
from tests.conftest import MockProvider


def test_format_full_transcript():
    rounds = [
        Round(
            number=1,
            responses=[
                ModelResponse(
                    "gemini", "gemini-3.1-pro-preview", 1, "Use YAML.", 1.0, 10
                ),
                ModelResponse("claude", "claude-opus-4-6", 1, "Use JSON.", 1.1, 12),
            ],
        ),
        Round(
            number=2,
            responses=[
                ModelResponse(
                    "gemini", "gemini-3.1-pro-preview", 2, "Changed mind: JSON.", 0.8, 8
                ),
            ],
        ),
    ]
    transcript = _format_full_transcript(rounds)
    assert "### Round 1" in transcript
    assert "### Round 2" in transcript
    assert "Use YAML." in transcript
    assert "Changed mind: JSON." in transcript


async def test_synthesize_returns_debate_result(
    sample_prompts_config, sample_question, sample_round
):
    synthesizer = MockProvider("openai", "## Consensus\nAll agreed on YAML.")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="openai",
            model="gpt-5.2",
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
    assert result.synthesizer == "openai"
    assert result.total_duration_sec >= 5.0


async def test_synthesize_passes_panel_mode_to_result(
    sample_prompts_config, sample_question, sample_round
):
    synthesizer = MockProvider("openai", "## Consensus\nAgreed.")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="openai",
            model="gpt-5.2",
            round_number=2,
            content="## Consensus\nAgreed.",
            latency_sec=0.5,
            token_count=10,
        )
    )

    result = await synthesize(
        question=sample_question,
        rounds=[sample_round],
        synthesizer=synthesizer,
        prompts=sample_prompts_config,
        debate_start_time=time.monotonic(),
        panel_mode="full",
        synthesizer_is_participant=False,
    )

    assert result.panel_mode == "full"
    assert result.synthesizer_is_participant is False


async def test_synthesize_records_is_participant(
    sample_prompts_config, sample_question, sample_round
):
    synthesizer = MockProvider("claude", "## Decision\nUse YAML.")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="claude",
            model="claude-opus-4-6",
            round_number=2,
            content="## Decision\nUse YAML.",
            latency_sec=0.5,
            token_count=10,
        )
    )

    result = await synthesize(
        question=sample_question,
        rounds=[sample_round],
        synthesizer=synthesizer,
        prompts=sample_prompts_config,
        debate_start_time=time.monotonic(),
        panel_mode="default",
        synthesizer_is_participant=True,
    )

    assert result.synthesizer_is_participant is True


async def test_synthesize_raises_on_empty_content(
    sample_prompts_config, sample_question, sample_round
):
    synthesizer = MockProvider("claude", "")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="claude",
            model="claude-opus-4-6",
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


async def test_synthesize_populates_synthesis_metrics(
    sample_prompts_config, sample_question, sample_round
):
    synthesizer = MockProvider("openai", "## Verdict\nYAML wins.")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="openai",
            model="gpt-5.2",
            round_number=2,
            content="## Verdict\nYAML wins.",
            latency_sec=1.2,
            token_count=30,
            input_tokens=200,
            output_tokens=30,
        )
    )

    result = await synthesize(
        question=sample_question,
        rounds=[sample_round],
        synthesizer=synthesizer,
        prompts=sample_prompts_config,
        debate_start_time=time.monotonic() - 2.0,
    )

    assert result.synthesis_metrics is not None
    sm = result.synthesis_metrics
    assert sm.synthesizer_model == "gpt-5.2"
    assert sm.transcript_size_tokens == 200
    assert sm.output_tokens == 30
    assert sm.synth_latency_seconds >= 0  # wall-clock; mock returns instantly
    assert sm.error_class == "none"


async def test_synthesize_timeout_sets_error_class(
    sample_prompts_config, sample_question, sample_round, caplog
):
    synthesizer = MockProvider("gemini", "")
    synthesizer.generate = AsyncMock(
        side_effect=ProviderError("gemini", "Request timed out after 30s")
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="ai_council.synthesis"):
        with pytest.raises(ProviderError):
            await synthesize(
                question=sample_question,
                rounds=[sample_round],
                synthesizer=synthesizer,
                prompts=sample_prompts_config,
                debate_start_time=time.monotonic(),
            )

    assert any("error_class=timeout" in r.message for r in caplog.records)


async def test_synthesize_provider_error_logs_error_class(
    sample_prompts_config, sample_question, sample_round, caplog
):
    synthesizer = MockProvider("deepseek", "")
    synthesizer.generate = AsyncMock(
        side_effect=ProviderError("deepseek", "429 rate limit exceeded")
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="ai_council.synthesis"):
        with pytest.raises(ProviderError):
            await synthesize(
                question=sample_question,
                rounds=[sample_round],
                synthesizer=synthesizer,
                prompts=sample_prompts_config,
                debate_start_time=time.monotonic(),
            )

    assert any("error_class=rate_limit" in r.message for r in caplog.records)
