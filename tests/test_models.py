"""Tests for src/models.py dataclasses."""

from src.models import DebateResult, ModelResponse, Question, Round


def test_question_fields():
    q = Question(text="Should we use YAML?", source="cli")
    assert q.text == "Should we use YAML?"
    assert q.source == "cli"


def test_model_response_fields():
    r = ModelResponse(
        provider="claude",
        model="claude-sonnet-4-20250514",
        round_number=1,
        content="Use YAML.",
        latency_sec=1.2,
        token_count=100,
    )
    assert r.provider == "claude"
    assert r.token_count == 100


def test_model_response_optional_token_count():
    r = ModelResponse(
        provider="gemini",
        model="gemini-2.5-flash",
        round_number=2,
        content="Some answer.",
        latency_sec=0.9,
        token_count=None,
    )
    assert r.token_count is None


def test_round_default_responses():
    rnd = Round(number=1)
    assert rnd.responses == []


def test_round_with_responses(sample_response):
    rnd = Round(number=1, responses=[sample_response])
    assert len(rnd.responses) == 1
    assert rnd.responses[0].provider == "claude"


def test_debate_result_fields(sample_question, sample_round):
    result = DebateResult(
        question=sample_question,
        rounds=[sample_round],
        synthesis="## Consensus\nAll agreed.",
        synthesizer="claude",
        total_duration_sec=5.0,
    )
    assert result.synthesizer == "claude"
    assert len(result.rounds) == 1
    assert result.total_duration_sec == 5.0
