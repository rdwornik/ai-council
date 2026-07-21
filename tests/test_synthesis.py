"""Tests for src/synthesis.py."""

import time
from unittest.mock import AsyncMock

import pytest

from ai_council.models import DebateResult, ModelResponse, Round
from ai_council.providers.base import ProviderError
from ai_council.synthesis import (
    SYNTHESIS_FAILED_MARKER,
    EmptySynthesisError,
    _format_full_transcript,
    build_failed_synthesis_result,
    synthesize,
)
from tests.conftest import MockProvider


async def test_empty_content_error_carries_the_billed_response(
    sample_prompts_config, sample_question, sample_round
):
    """terra HIGH: the empty-content path had ALREADY received a billable ModelResponse, so
    booking the failed synthesis at zero tokens silently understated real spend. The error
    carries the response so the preservation path can book what was actually charged."""
    synthesizer = MockProvider("openai", "")
    synthesizer.generate = AsyncMock(
        return_value=ModelResponse(
            provider="openai", model="gpt-5.2", round_number=2, content="",
            latency_sec=1.0, token_count=900, input_tokens=800, output_tokens=100,
        )
    )
    with pytest.raises(RuntimeError, match="empty content") as exc_info:
        await synthesize(
            question=sample_question,
            rounds=[sample_round],
            synthesizer=synthesizer,
            prompts=sample_prompts_config,
            debate_start_time=time.monotonic(),
            model_configs={},
        )
    assert isinstance(exc_info.value, EmptySynthesisError)
    assert exc_info.value.response.input_tokens == 800
    assert exc_info.value.response.output_tokens == 100


def test_failed_result_books_the_billed_usage_when_a_response_exists(
    sample_question, sample_round
):
    """A synthesis that returned empty content was still charged — the sidecar must say so."""
    billed = ModelResponse(
        provider="openai", model="gpt-5.2", round_number=2, content="",
        latency_sec=1.0, token_count=900, input_tokens=800, output_tokens=100,
    )
    result = build_failed_synthesis_result(
        question=sample_question,
        rounds=[sample_round],
        synthesizer_name="openai",
        error=EmptySynthesisError("Synthesizer openai returned empty content", billed),
        debate_start_time=time.monotonic() - 3.0,
        model_configs={},
        synth_latency_sec=2.5,
    )
    synth_call = [c for c in result.metrics.calls if c.round_number == 0]
    assert len(synth_call) == 1
    assert synth_call[0].input_tokens == 800  # never rounded down to a fabricated zero
    assert synth_call[0].output_tokens == 100
    assert result.synthesis_metrics.transcript_size_tokens == 800
    assert result.synthesis_metrics.output_tokens == 100


def test_foreign_response_attribute_does_not_defeat_preservation(
    sample_question, sample_round
):
    """terra follow-up HIGH: reading `.response` by duck-typing meant an SDK exception — which
    commonly carries an HTTP response under that exact name — was treated as a ModelResponse.
    build_call_metrics would then raise AttributeError INSIDE the except handler, masking the
    original error and losing every artifact: H1's data-loss path, recreated."""

    class _FakeSdkError(Exception):
        """Shaped like openai.APIStatusError / httpx.HTTPStatusError."""

        def __init__(self) -> None:
            super().__init__("503 upstream unavailable")
            self.response = object()  # an HTTP response, NOT a ModelResponse

    result = build_failed_synthesis_result(
        question=sample_question,
        rounds=[sample_round],
        synthesizer_name="openai",
        error=_FakeSdkError(),
        debate_start_time=time.monotonic() - 1.0,
        model_configs={},
        synth_latency_sec=1.0,
    )
    # Preservation still succeeds, and the foreign attribute is ignored rather than trusted.
    assert result.metrics is not None
    assert result.synthesis_metrics.transcript_size_tokens is None
    assert result.synthesis_metrics.synthesizer_model == "openai"
    assert SYNTHESIS_FAILED_MARKER in result.synthesis


def test_failed_result_records_real_latency_not_zero(sample_question, sample_round):
    """terra HIGH: latency was hardcoded 0.0, contradicting the run's real total_duration."""
    result = build_failed_synthesis_result(
        question=sample_question,
        rounds=[sample_round],
        synthesizer_name="openai",
        error=ProviderError("openai", "API error"),
        debate_start_time=time.monotonic() - 3.0,
        model_configs={},
        synth_latency_sec=2.5,
    )
    assert result.synthesis_metrics.synth_latency_seconds == 2.5
    synth_call = [c for c in result.metrics.calls if c.round_number == 0][0]
    assert synth_call.latency_sec == 2.5
    # No response was ever returned, so zero TOKENS here is observed truth, not a fabrication.
    assert synth_call.input_tokens == 0
    assert result.synthesis_metrics.transcript_size_tokens is None  # unknown, not zero


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
