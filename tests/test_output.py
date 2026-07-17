"""Tests for src/output.py."""

import json
from pathlib import Path

import pytest

from ai_council.models import (
    DebateMetrics,
    DebateResult,
    FallbackEvent,
    ModelResponse,
    Round,
    SeatMetrics,
    SynthesisMetrics,
)
from ai_council.output import (
    _slug,
    extract_dissent,
    save_minority_report,
    save_to_file,
    save_verdict_package,
)


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


def test_save_metrics_json_emits_seats(tmp_path, sample_question, sample_round):
    """seats[] is emitted as an additive top-level namespace, keys/models by name only."""
    import json as _json

    from ai_council.models import (
        DebateMetrics,
        DebateResult,
        FallbackEvent,
        SeatMetrics,
        SynthesisMetrics,
    )
    from ai_council.output import _save_metrics_json

    seats = [
        SeatMetrics(seat="claude", requested_backend="cli", actual_backend="cli",
                    requested_model="opus", actual_model="claude-opus-4-8",
                    identity_channel="modelUsage", identity_readable=True,
                    cli={"name": "claude", "version": "2.1.212"}),
        SeatMetrics(seat="openai", requested_backend="cli", actual_backend="api",
                    requested_model="gpt-5.6-sol", actual_model="gpt-x",
                    identity_channel="api-echo", identity_readable=True,
                    fallback_events=[FallbackEvent(round=1, from_backend="cli",
                                                   to_backend="api", cause="timeout",
                                                   detail="CLI timed out after 30s")]),
    ]
    result = DebateResult(
        question=sample_question, rounds=[sample_round], synthesis="s", synthesizer="gemini",
        total_duration_sec=1.0, metrics=DebateMetrics(seats=seats),
        synthesis_metrics=SynthesisMetrics("gemini", 10, 5, 0.5, "none"),
    )
    transcript = tmp_path / "council-out-test.md"
    transcript.write_text("x", encoding="utf-8")
    _save_metrics_json(result, transcript)

    data = _json.loads((tmp_path / "council-out-test_metrics.json").read_text(encoding="utf-8"))
    assert len(data["seats"]) == 2
    by_seat = {s["seat"]: s for s in data["seats"]}
    assert by_seat["claude"]["actual_backend"] == "cli"
    assert by_seat["claude"]["identity_channel"] == "modelUsage"
    assert by_seat["claude"]["cli"] == {"name": "claude", "version": "2.1.212"}
    assert by_seat["openai"]["fallback_events"][0]["cause"] == "timeout"
    # additive namespace: calls[] still present, seats[] alongside (not nested)
    assert "calls" in data and "seats" in data


# ---------------------------------------------------------------------------
# Verdict package (DRAFT-INT-1, #26) — the transcript-free caller deliverable
# ---------------------------------------------------------------------------

_PICK_SYNTHESIS = (
    "## Position\nUse YAML.\n\n"
    "## Recommendation\nAdopt YAML for config.\n\n"
    "## Rationale\n- Human-editable\n- Comments supported\n\n"
    "## Alternatives Considered\n- JSON: no comments\n- TOML: less familiar\n"
)


def _pick_result(sample_question, sample_round, **overrides) -> DebateResult:
    base = dict(
        question=sample_question,
        rounds=[sample_round],
        synthesis=_PICK_SYNTHESIS,
        synthesizer="gemini",
        total_duration_sec=5.0,
        panel_mode="default",
        mode="pick",
    )
    base.update(overrides)
    return DebateResult(**base)


def _load_verdict(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(result: DebateResult, out: Path, **routes) -> dict:
    """save_to_file then save_verdict_package (the orchestrator sequence); return the payload."""
    transcript = save_to_file(result, out, **routes)[0]
    verdict = save_verdict_package(result, out, transcript, **routes)
    return _load_verdict(verdict[0])


def test_verdict_package_emits_full_field_set(tmp_path, sample_question, sample_round):
    result = _pick_result(sample_question, sample_round)
    transcript = save_to_file(result, tmp_path / "out")[0]
    saved = save_verdict_package(result, tmp_path / "out", transcript)
    assert len(saved) == 1
    assert saved[0].name.startswith("council-verdict-")
    assert saved[0].name.endswith(".json")
    data = _load_verdict(saved[0])
    for field in (
        "run_id", "timestamp", "contract_version", "question", "mode", "exit_semantics",
        "decision", "rationale", "options_considered", "dissent", "panel",
        "verdict_author", "degradation", "artifacts",
    ):
        assert field in data, f"missing DRAFT-INT-1 field: {field}"


def test_verdict_ts_matches_transcript_deterministically(tmp_path, sample_question, sample_round):
    """The verdict shares the transcript's <ts>-<mode>-<slug> stem (single source, B3)."""
    result = _pick_result(sample_question, sample_round)
    transcript = save_to_file(result, tmp_path / "out")[0]
    verdict = save_verdict_package(result, tmp_path / "out", transcript)[0]
    t_base = transcript.name[len("council-out-"):-len(".md")]
    v_base = verdict.name[len("council-verdict-"):-len(".json")]
    assert t_base == v_base
    assert _load_verdict(verdict)["run_id"] == transcript.stem


def test_verdict_routes_to_every_destination(tmp_path, sample_question, sample_round):
    result = _pick_result(sample_question, sample_round)
    ret = tmp_path / "return"
    target = tmp_path / "mirror"
    transcript = save_to_file(result, tmp_path / "out", return_dir=ret, target_paths=[target])[0]
    saved = save_verdict_package(
        result, tmp_path / "out", transcript, return_dir=ret, target_paths=[target]
    )
    assert len(saved) == 3
    assert saved[0].parent == tmp_path / "out"  # canonical first
    assert {p.parent for p in saved} == {tmp_path / "out", ret, target}
    assert len({p.name for p in saved}) == 1  # identical filename in every destination


def test_verdict_transcript_free_decision_extraction(tmp_path, sample_question, sample_round):
    """decision/rationale/options are extracted from synthesis and annotated source='extraction'."""
    data = _emit(_pick_result(sample_question, sample_round), tmp_path / "out")
    assert data["decision"]["value"] == "Adopt YAML for config."
    assert data["decision"]["source"] == "extraction"
    assert data["decision"]["heading"] == "Recommendation"
    assert "Human-editable" in data["rationale"]["value"]
    assert data["rationale"]["source"] == "extraction"
    assert data["options_considered"]["items"] == ["JSON: no comments", "TOML: less familiar"]
    assert data["options_considered"]["source"] == "extraction"


def test_verdict_decision_strips_wrapping_emphasis(tmp_path, sample_question, sample_round):
    """A bold-wrapped one-line decision is extracted clean (no leading/trailing ** )."""
    synthesis = "## Recommended Decision\n**Default to a monorepo with lightweight tooling.**\n"
    data = _emit(_pick_result(sample_question, sample_round, synthesis=synthesis), tmp_path / "out")
    assert data["decision"]["value"] == "Default to a monorepo with lightweight tooling."


def test_verdict_dissent_unanimous(tmp_path, sample_question, sample_round):
    data = _emit(_pick_result(sample_question, sample_round), tmp_path / "out")
    assert data["dissent"]["status"] == "unanimous"
    assert data["dissent"]["minority_artifact"] is None
    assert data["dissent"]["source"] == "extraction"


def test_verdict_dissent_non_unanimous_points_to_minority(tmp_path, sample_question, sample_round):
    result = _pick_result(sample_question, sample_round, synthesis=_DISSENT_SYNTHESIS)
    data = _emit(result, tmp_path / "out")
    assert data["dissent"]["status"] == "non-unanimous"
    assert data["dissent"]["minority_artifact"].startswith("council-minority-")
    assert data["dissent"]["minority_artifact"].endswith(".md")
    # gist is the dissent CONTENT, not an echo of the section heading
    assert "crux" in data["dissent"]["gist"].lower()
    assert data["dissent"]["gist"] != "Unresolved Disagreements"


def test_verdict_contract_version_null_and_exit_zero(tmp_path, sample_question, sample_round):
    """Gap 2 (no invented version) + Gap 3 (completed debate exits 0)."""
    data = _emit(_pick_result(sample_question, sample_round), tmp_path / "out")
    assert data["contract_version"] is None
    assert data["exit_semantics"] == 0


def test_verdict_panel_and_degradation_record(tmp_path, sample_question, sample_round):
    """Shrunk-panel truth is caller-visible (two-signal rule); alarm text persisted (G3)."""
    result = _pick_result(
        sample_question,
        sample_round,
        provider_statuses={"claude": "ok", "deepseek": "failed"},
        degraded=True,
        degradation_summary="deepseek failed: timeout after 120s",
    )
    data = _emit(result, tmp_path / "out")
    assert "claude" in data["panel"]["seated"]
    assert data["panel"]["dropped"] == ["deepseek"]
    assert "deepseek" in data["panel"]["requested"]
    assert data["panel"]["source"] == "record"
    assert data["degradation"]["degraded"] is True
    assert "timeout" in data["degradation"]["summary"]
    assert data["degradation"]["failed_providers"] == ["deepseek"]


def test_verdict_fallback_events_persisted_by_reference(tmp_path, sample_question, sample_round):
    """G4: per-seat classified CLI fallback causes land in the package (seats[] consumed by ref)."""
    seat = SeatMetrics(
        seat="claude",
        requested_backend="cli",
        actual_backend="api",
        requested_model="claude-opus-4-6",
        actual_model="claude-opus-4-6",
        identity_channel="stderr-banner",
        identity_readable=True,
        cli={"name": "claude", "version": "1.2.3"},
        fallback_events=[
            FallbackEvent(
                round=1,
                from_backend="cli",
                to_backend="api",
                cause="process-error",
                detail="cli exited 1",
            )
        ],
    )
    result = _pick_result(sample_question, sample_round, metrics=DebateMetrics(seats=[seat]))
    data = _emit(result, tmp_path / "out")
    events = data["degradation"]["fallback_events"]
    assert len(events) == 1
    assert events[0]["seat"] == "claude"
    assert events[0]["cause"] == "process-error"
    assert data["panel"]["seats"][0]["requested_backend"] == "cli"
    assert data["panel"]["seats"][0]["actual_backend"] == "api"


def test_verdict_author_sourced_from_synthesis_metrics(tmp_path, sample_question, sample_round):
    sm = SynthesisMetrics(
        synthesizer_model="gemini-2.5-pro",
        transcript_size_tokens=1000,
        output_tokens=500,
        synth_latency_seconds=3.2,
        error_class="none",
    )
    result = _pick_result(sample_question, sample_round, synthesis_metrics=sm)
    data = _emit(result, tmp_path / "out")
    assert data["verdict_author"]["actual"] == "gemini"
    assert data["verdict_author"]["model"] == "gemini-2.5-pro"
    assert data["verdict_author"]["source"] == "record"


def test_verdict_mirror_block_atop_transcript(tmp_path, sample_question, sample_round):
    """DRAFT-INT-1 human-readable mirror block lands in council-out (Gap 1 ruling)."""
    result = _pick_result(sample_question, sample_round)
    transcript = save_to_file(result, tmp_path / "out")[0]
    content = transcript.read_text(encoding="utf-8")
    assert "## Verdict Summary" in content
    assert "**Decision:** Adopt YAML for config." in content
    assert "council-verdict-*.json" in content


def test_verdict_mirror_block_does_not_echo_question(tmp_path, sample_round):
    """The mirror must not duplicate the question (protects the ## Question single-site invariant)."""
    result = _result_with_question(_LONG_QUESTION, "pick", sample_round)
    transcript = save_to_file(result, tmp_path / "out")[0]
    content = transcript.read_text(encoding="utf-8")
    # first occurrence of the full question is in the body's ## Question section, not the mirror
    assert content.index(_LONG_QUESTION) > content.index("## Question")


def test_verdict_artifacts_manifest_lists_run_outputs(tmp_path, sample_question, sample_round):
    result = _pick_result(sample_question, sample_round)
    saved_paths = save_to_file(result, tmp_path / "out")
    written = {"transcript": saved_paths}
    verdict = save_verdict_package(
        result, tmp_path / "out", saved_paths[0], written=written
    )
    data = _load_verdict(verdict[0])
    kinds = {a["kind"] for a in data["artifacts"]}
    assert "transcript" in kinds
    assert "verdict" in kinds
    transcript_entry = next(a for a in data["artifacts"] if a["kind"] == "transcript")
    assert transcript_entry["filename"] == saved_paths[0].name
