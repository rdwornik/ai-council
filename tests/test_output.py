"""Tests for src/output.py."""

from pathlib import Path

import pytest

from src.models import DebateResult, ModelResponse, Question, Round
from src.output import _slug, save_to_file


def test_slug_basic():
    assert _slug("Should we use YAML or JSON?") == "should-we-use-yaml-or-json"


def test_slug_max_len():
    long_text = "a" * 100
    assert len(_slug(long_text)) <= 40


def test_slug_special_chars():
    result = _slug("API vs. SDK (2024)")
    assert "." not in result
    assert "(" not in result
    assert ")" not in result


@pytest.fixture
def sample_debate_result(sample_question, sample_round) -> DebateResult:
    return DebateResult(
        question=sample_question,
        rounds=[sample_round],
        synthesis="## Consensus\nAll agreed.",
        synthesizer="openai",
        total_duration_sec=10.5,
        panel_mode="default",
        synthesizer_is_participant=False,
    )


def test_save_to_file_creates_file(tmp_path: Path, sample_debate_result: DebateResult):
    saved = save_to_file(sample_debate_result, tmp_path / "output")
    assert saved[0].exists()
    assert saved[0].suffix == ".md"


def test_save_to_file_creates_output_dir(
    tmp_path: Path, sample_debate_result: DebateResult
):
    output_dir = tmp_path / "nested" / "output"
    assert not output_dir.exists()
    save_to_file(sample_debate_result, output_dir)
    assert output_dir.exists()


def test_save_to_file_content(tmp_path: Path, sample_debate_result: DebateResult):
    saved = save_to_file(sample_debate_result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "AI Council Debate" in content
    assert "Round 1" in content
    assert "## Consensus" in content
    assert "Synthesis" in content
    assert "claude" in content  # synthesizer appears in synthesis section


def test_save_to_file_has_panel_header(
    tmp_path: Path, sample_debate_result: DebateResult
):
    saved = save_to_file(sample_debate_result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "**Panel:**" in content


def test_save_to_file_has_mode_header(
    tmp_path: Path, sample_debate_result: DebateResult
):
    saved = save_to_file(sample_debate_result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "**Panel Mode:**" in content
    assert "default" in content


def test_save_to_file_has_synthesizer_header(
    tmp_path: Path, sample_debate_result: DebateResult
):
    saved = save_to_file(sample_debate_result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "**Synthesizer:**" in content
    assert "non-participant" in content


def test_save_to_file_participant_label(tmp_path: Path, sample_question, sample_round):
    result = DebateResult(
        question=sample_question,
        rounds=[sample_round],
        synthesis="## Decision\nUse YAML.",
        synthesizer="claude",
        total_duration_sec=5.0,
        panel_mode="custom",
        synthesizer_is_participant=True,
    )
    saved = save_to_file(result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "participant" in content
    assert "**Panel Mode:** custom" in content


def test_save_to_file_filename_has_slug(
    tmp_path: Path, sample_debate_result: DebateResult
):
    saved = save_to_file(sample_debate_result, tmp_path)
    assert "yaml" in saved[0].name or "should" in saved[0].name  # slug from question


def test_provider_notes_retried_provider(tmp_path, sample_question):
    """Provider Notes line appears when a response has was_retry=True."""
    retried_response = ModelResponse(
        provider="claude",
        model="claude-opus-4-6",
        round_number=1,
        content="Retried response",
        latency_sec=2.0,
        token_count=10,
        was_retry=True,
    )
    result = DebateResult(
        question=sample_question,
        rounds=[Round(number=1, responses=[retried_response])],
        synthesis="## Decision",
        synthesizer="openai",
        total_duration_sec=5.0,
        provider_statuses={"claude": "ok"},
    )
    saved = save_to_file(result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "**Provider Notes:**" in content
    assert "claude retried" in content
    assert "recovered" in content


def test_provider_notes_skipped_provider(tmp_path, sample_question, sample_round):
    """Provider Notes line appears when a provider is in provider_statuses as 'failed'."""
    result = DebateResult(
        question=sample_question,
        rounds=[sample_round],
        synthesis="## Decision",
        synthesizer="openai",
        total_duration_sec=5.0,
        provider_statuses={"claude": "ok", "deepseek": "failed"},
    )
    saved = save_to_file(result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "**Provider Notes:**" in content
    assert "deepseek skipped" in content


def test_provider_notes_absent_when_all_ok(tmp_path, sample_debate_result):
    """No Provider Notes line when all providers succeeded without retry."""
    saved = save_to_file(sample_debate_result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert "**Provider Notes:**" not in content
