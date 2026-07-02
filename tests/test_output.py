"""Tests for src/output.py."""

from pathlib import Path

import pytest

from ai_council.models import DebateResult, ModelResponse, Round
from ai_council.output import _slug, extract_dissent, save_minority_report, save_to_file


def _result_with_synthesis(synthesis: str, sample_question, sample_round, mode: str = "pick") -> DebateResult:
    return DebateResult(
        question=sample_question,
        rounds=[sample_round],
        synthesis=synthesis,
        synthesizer="openai",
        total_duration_sec=5.0,
        panel_mode="default",
        synthesizer_is_participant=False,
        mode=mode,
    )


_DISSENT_SYNTHESIS = (
    "## Consensus\nBoth options are viable.\n\n"
    "## Unresolved Disagreements\n"
    "The crux: Claude argued Postgres is sufficient at this scale, while Grok held that "
    "Cassandra is required for the projected write volume. Grok's write-amplification "
    "estimate was the stronger argument.\n\n"
    "## Recommended Decision\nStart with Postgres.\n"
)

_CONSENSUS_SYNTHESIS = (
    "## Consensus\nEveryone agreed.\n\n"
    "## Unresolved Disagreements\nNone - the panel reached consensus.\n\n"
    "## Recommended Decision\nProceed.\n"
)


# ---------------------------------------------------------------------------
# Minority report (Rama 4, #15)
# ---------------------------------------------------------------------------


def test_extract_dissent_returns_body_on_genuine_dissent() -> None:
    body = extract_dissent(_DISSENT_SYNTHESIS)
    assert body is not None
    assert "Unresolved Disagreements" in body
    assert "Cassandra" in body
    assert "Recommended Decision" not in body  # only the dissent section is extracted


def test_extract_dissent_none_on_consensus() -> None:
    assert extract_dissent(_CONSENSUS_SYNTHESIS) is None


def test_extract_dissent_none_when_no_dissent_section() -> None:
    assert extract_dissent("## Consensus\nAll agreed.\n\n## Recommended Decision\nDo it.") is None


def test_extract_dissent_judge_contested_points() -> None:
    synthesis = (
        "## Overall Verdict\nSolid.\n\n"
        "## Contested Points\nThe reviewers split on whether the retry logic is safe "
        "under partition; one flagged a data-loss window.\n"
    )
    body = extract_dissent(synthesis)
    assert body is not None
    assert "Contested Points" in body


def test_save_minority_report_writes_artifact(tmp_path: Path, sample_question, sample_round) -> None:
    result = _result_with_synthesis(_DISSENT_SYNTHESIS, sample_question, sample_round)
    saved = save_minority_report(result, tmp_path / "output")
    assert len(saved) == 1
    assert saved[0].exists()
    assert saved[0].name.startswith("council-minority-")
    content = saved[0].read_text(encoding="utf-8")
    assert "Minority Report" in content
    assert "Cassandra" in content
    assert "NOT unanimous" in content


def test_save_minority_report_empty_on_consensus(tmp_path: Path, sample_question, sample_round) -> None:
    result = _result_with_synthesis(_CONSENSUS_SYNTHESIS, sample_question, sample_round)
    out_dir = tmp_path / "output"
    saved = save_minority_report(result, out_dir)
    assert saved == []
    # no council-minority file was created
    assert not (out_dir.exists() and list(out_dir.glob("council-minority-*.md")))


def test_save_minority_report_routes_to_return_dir(tmp_path: Path, sample_question, sample_round) -> None:
    result = _result_with_synthesis(_DISSENT_SYNTHESIS, sample_question, sample_round)
    ret = tmp_path / "commissioned"
    saved = save_minority_report(result, tmp_path / "output", return_dir=ret)
    assert len(saved) == 2
    assert saved[0].parent == tmp_path / "output"  # canonical first
    assert saved[1].parent == ret
    assert saved[0].name == saved[1].name


def test_minority_and_verdict_share_slug(tmp_path: Path, sample_question, sample_round) -> None:
    """The minority artifact sits alongside the verdict with the matching slug."""
    result = _result_with_synthesis(_DISSENT_SYNTHESIS, sample_question, sample_round)
    verdict = save_to_file(result, tmp_path / "output", slug_override="my-question")
    minority = save_minority_report(result, tmp_path / "output", slug_override="my-question")
    assert verdict[0].name.startswith("council-out-")
    assert minority[0].name.startswith("council-minority-")
    assert verdict[0].name.endswith("-pick-my-question.md")
    assert minority[0].name.endswith("-pick-my-question.md")


def test_slug_basic():
    assert _slug("Should we use YAML or JSON?") == "should-we-use-yaml-or-json"


def test_slug_max_len():
    long_text = "a" * 100
    assert len(_slug(long_text)) <= 50


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


_LONG_QUESTION = (
    "Should we migrate the entire ingestion pipeline from Kafka to Pulsar this "
    "quarter, given the operational cost concerns raised by the platform team "
    "and the open question about exactly-once semantics under partition rebalance?"
)


def _result_with_question(question_text: str, mode: str, sample_round) -> DebateResult:
    from ai_council.models import Question
    return DebateResult(
        question=Question(text=question_text, source="cli"),
        rounds=[sample_round],
        synthesis="## Decision\nProceed.",
        synthesizer="openai",
        total_duration_sec=5.0,
        panel_mode="default",
        synthesizer_is_participant=False,
        mode=mode,
    )


@pytest.mark.parametrize("mode", ["pick", "judge", "ideas"])
def test_transcript_preserves_full_question(tmp_path: Path, sample_round, mode: str):
    """Debate transcripts must contain the full submitted question, not just the truncated title."""
    assert len(_LONG_QUESTION) > 80, "test question must exceed the 80-char title cap"
    result = _result_with_question(_LONG_QUESTION, mode, sample_round)
    saved = save_to_file(result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    assert _LONG_QUESTION in content


@pytest.mark.parametrize("mode", ["pick", "judge", "ideas"])
def test_transcript_question_section_position(tmp_path: Path, sample_round, mode: str):
    """The `## Question` block must sit after the metadata `Source:` line and before `## Round 1`."""
    result = _result_with_question(_LONG_QUESTION, mode, sample_round)
    saved = save_to_file(result, tmp_path)
    content = saved[0].read_text(encoding="utf-8")
    src_idx = content.index("**Source:**")
    q_idx = content.index("## Question")
    r1_idx = content.index("## Round 1")
    assert src_idx < q_idx < r1_idx
    assert content.index(_LONG_QUESTION) > q_idx
    assert content.index(_LONG_QUESTION) < r1_idx


def test_save_to_file_filename_convention(
    tmp_path: Path, sample_debate_result: DebateResult
):
    saved = save_to_file(sample_debate_result, tmp_path)
    name = saved[0].name
    assert name.startswith("council-out-")
    assert "-pick-" in name  # mode in filename
    assert "yaml" in name or "should" in name  # slug from question
    assert name.endswith(".md")


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


# ---------------------------------------------------------------------------
# target_paths mirroring
# ---------------------------------------------------------------------------


def test_target_paths_canonical_only(tmp_path: Path, sample_debate_result) -> None:
    saved = save_to_file(sample_debate_result, tmp_path / "primary")
    assert len(saved) == 1


def test_target_paths_single_target(tmp_path: Path, sample_debate_result) -> None:
    target = tmp_path / "target_root" / "docs" / "decisions" / "transcripts"
    target.mkdir(parents=True)
    saved = save_to_file(sample_debate_result, tmp_path / "primary", target_paths=[target])
    assert len(saved) == 2
    assert saved[1].parent == target
    assert saved[1].exists()


def test_target_paths_two_targets(tmp_path: Path, sample_debate_result) -> None:
    t1 = tmp_path / "target1" / "docs" / "decisions" / "transcripts"
    t2 = tmp_path / "target2" / "docs" / "decisions" / "transcripts"
    t1.mkdir(parents=True)
    t2.mkdir(parents=True)
    saved = save_to_file(sample_debate_result, tmp_path / "primary", target_paths=[t1, t2])
    assert len(saved) == 3


def test_target_paths_auto_mkdir(tmp_path: Path, sample_debate_result) -> None:
    target = tmp_path / "new_project" / "docs" / "decisions" / "transcripts"
    assert not target.exists()
    saved = save_to_file(sample_debate_result, tmp_path / "primary", target_paths=[target])
    assert target.exists()
    assert len(saved) == 2


def test_target_paths_content_identical(tmp_path: Path, sample_debate_result) -> None:
    target = tmp_path / "target" / "docs" / "decisions" / "transcripts"
    saved = save_to_file(sample_debate_result, tmp_path / "primary", target_paths=[target])
    assert saved[0].read_text(encoding="utf-8") == saved[1].read_text(encoding="utf-8")


def test_target_paths_mirror_failure_logs_warning(tmp_path: Path, sample_debate_result, caplog, monkeypatch) -> None:
    import logging
    target = tmp_path / "target"
    original_mkdir = Path.mkdir

    def _selective_fail(self: Path, *args, **kwargs):
        if self == target:
            raise PermissionError("blocked")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _selective_fail)
    with caplog.at_level(logging.WARNING, logger="ai_council.output"):
        saved = save_to_file(sample_debate_result, tmp_path / "primary", target_paths=[target])
    assert len(saved) == 1  # canonical written, mirror skipped
    assert any("Mirror write failed" in r.message for r in caplog.records)


def test_target_paths_mirror_failure_canonical_still_written(tmp_path: Path, sample_debate_result, monkeypatch) -> None:
    target = tmp_path / "target"
    original_mkdir = Path.mkdir

    def _selective_fail(self: Path, *args, **kwargs):
        if self == target:
            raise PermissionError("blocked")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _selective_fail)
    saved = save_to_file(sample_debate_result, tmp_path / "primary", target_paths=[target])
    assert saved[0].exists()  # canonical always written


# ---------------------------------------------------------------------------
# return_dir routing (ADR-10, #13)
# ---------------------------------------------------------------------------


def test_return_dir_unset_canonical_only(tmp_path: Path, sample_debate_result) -> None:
    """Unset return_dir → output lands in the canonical dir only."""
    saved = save_to_file(sample_debate_result, tmp_path / "output", return_dir=None)
    assert len(saved) == 1
    assert saved[0].parent == tmp_path / "output"


def test_return_dir_routes_and_keeps_canonical(tmp_path: Path, sample_debate_result) -> None:
    """--return-dir routes a copy while the canonical ./output write still fires."""
    canonical = tmp_path / "output"
    ret = tmp_path / "commissioned" / "return"
    saved = save_to_file(sample_debate_result, canonical, return_dir=ret)
    assert len(saved) == 2
    assert saved[0].parent == canonical  # canonical always first
    assert saved[1].parent == ret
    assert saved[0].exists() and saved[1].exists()


def test_return_dir_auto_mkdir(tmp_path: Path, sample_debate_result) -> None:
    ret = tmp_path / "does" / "not" / "exist" / "yet"
    assert not ret.exists()
    saved = save_to_file(sample_debate_result, tmp_path / "output", return_dir=ret)
    assert ret.exists()
    assert len(saved) == 2


def test_return_dir_content_identical_to_canonical(tmp_path: Path, sample_debate_result) -> None:
    ret = tmp_path / "return"
    saved = save_to_file(sample_debate_result, tmp_path / "output", return_dir=ret)
    assert saved[0].read_text(encoding="utf-8") == saved[1].read_text(encoding="utf-8")
    assert saved[0].name == saved[1].name


def test_return_dir_failure_canonical_still_written(tmp_path: Path, sample_debate_result, monkeypatch, caplog) -> None:
    """A return_dir write failure is best-effort: canonical is still written, warning logged."""
    import logging
    ret = tmp_path / "return"
    original_mkdir = Path.mkdir

    def _selective_fail(self: Path, *args, **kwargs):
        if self == ret:
            raise PermissionError("blocked")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _selective_fail)
    with caplog.at_level(logging.WARNING, logger="ai_council.output"):
        saved = save_to_file(sample_debate_result, tmp_path / "output", return_dir=ret)
    assert len(saved) == 1
    assert saved[0].exists()
    assert any("Return-dir write failed" in r.message for r in caplog.records)
