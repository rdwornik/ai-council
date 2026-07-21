"""Tests for src/output.py."""

import json
from pathlib import Path

import pytest

from ai_council.models import (
    SEAT_STATUS_FAILED,
    SEAT_STATUS_LOST,
    SEAT_STATUS_OK,
    DebateMetrics,
    DebateResult,
    FallbackEvent,
    ModelResponse,
    Question,
    Round,
    SeatMetrics,
    SynthesisMetrics,
)
from ai_council.output import (
    _build_verdict_payload,
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


def test_minority_report_fails_loud_when_required_return_dir_unwritten(
    tmp_path: Path, sample_question, sample_round
) -> None:
    """#35: the minority report now really carries the verdict's R4 guarantee.

    Its docstring claimed "so a --return-dir also receives it" while the write was in fact
    best-effort and a miss was swallowed. This pins the claim to the behaviour.
    """
    from ai_council.output import OutputRoutingError

    result = _result_with_synthesis(_DISSENT_SYNTHESIS, sample_question, sample_round)
    canonical = tmp_path / "output"
    # point return_dir at an existing FILE so its mkdir fails inside _write_routed
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")

    with pytest.raises(OutputRoutingError) as excinfo:
        save_minority_report(result, canonical, return_dir=blocker)

    assert [f.artifact for f in excinfo.value.failures] == ["minority report"]
    assert len(list(canonical.glob("council-minority-*.md"))) == 1


def test_routing_failures_aggregate_names_every_missed_deliverable(
    tmp_path: Path, sample_question, sample_round
) -> None:
    """#35: a common-mode return-dir fault reports ALL deliverables, not just the first.

    The writers share one --return-dir, so reporting only the first would understate what
    the caller did not receive.
    """
    from ai_council.output import OutputRoutingError, RoutingFailure, raise_for_routing_failures

    result = _result_with_synthesis(_DISSENT_SYNTHESIS, sample_question, sample_round)
    canonical = tmp_path / "output"
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")

    failures: list[RoutingFailure] = []
    transcript = save_to_file(result, canonical, return_dir=blocker, routing_failures=failures)
    minority = save_minority_report(
        result, canonical, return_dir=blocker, stem_base=transcript[0].stem[len("council-out-"):],
        routing_failures=failures,
    )
    save_verdict_package(
        result, canonical, transcript[0], written={"minority": minority},
        return_dir=blocker, routing_failures=failures,
    )

    # every canonical artifact landed despite the common-mode return-dir fault
    assert transcript[0].exists() and minority[0].exists()
    assert len(list(canonical.glob("council-verdict-*.json"))) == 1

    with pytest.raises(OutputRoutingError) as excinfo:
        raise_for_routing_failures(failures)
    assert [f.artifact for f in excinfo.value.failures] == [
        "transcript", "minority report", "verdict package",
    ]
    assert "3 deliverables not delivered" in str(excinfo.value)


def test_raise_for_routing_failures_is_noop_when_empty() -> None:
    from ai_council.output import raise_for_routing_failures

    raise_for_routing_failures([])  # must not raise


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


def _fail_mkdir_for(monkeypatch, doomed: Path) -> None:
    """Make mkdir raise for exactly one dir, so only that destination fails."""
    original_mkdir = Path.mkdir

    def _selective_fail(self: Path, *args, **kwargs):
        if self == doomed:
            raise PermissionError("blocked")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _selective_fail)


def test_return_dir_failure_raises_and_canonical_still_written(
    tmp_path: Path, sample_debate_result, monkeypatch
) -> None:
    """#35/R4: a required return_dir miss raises rather than logging a warning and exiting 0.

    Polarity inverted from the pre-#35 contract, which asserted this was best-effort. The
    canonical write still lands first — R4 buys a loud failure, never a lost artifact.
    """
    from ai_council.output import OutputRoutingError

    ret = tmp_path / "return"
    canonical = tmp_path / "output"
    _fail_mkdir_for(monkeypatch, ret)

    with pytest.raises(OutputRoutingError) as excinfo:
        save_to_file(sample_debate_result, canonical, return_dir=ret)

    assert "transcript" in str(excinfo.value)
    assert [f.artifact for f in excinfo.value.failures] == ["transcript"]
    # canonical is written before the return-dir attempt, so it survives the raise
    assert len(list(canonical.glob("council-out-*.md"))) == 1


def test_return_dir_failure_recorded_when_caller_accumulates(
    tmp_path: Path, sample_debate_result, monkeypatch
) -> None:
    """#35: with an accumulator the writer records and returns, so the caller can raise once."""
    from ai_council.output import RoutingFailure

    ret = tmp_path / "return"
    canonical = tmp_path / "output"
    _fail_mkdir_for(monkeypatch, ret)

    failures: list[RoutingFailure] = []
    saved = save_to_file(sample_debate_result, canonical, return_dir=ret, routing_failures=failures)

    assert len(saved) == 1 and saved[0].exists()  # canonical only
    assert [f.artifact for f in failures] == ["transcript"]
    assert failures[0].destination == ret
    assert "PermissionError" in failures[0].cause


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
# #63: a metrics-sidecar failure degrades, it does not suppress the deliverable
# ---------------------------------------------------------------------------

def _fail_sidecar_write(monkeypatch) -> None:
    """Break only the *_metrics.json write, harness-side. No provider code is touched."""
    original_write = Path.write_text

    def _selective(self: Path, *args, **kwargs):
        if self.name.endswith("_metrics.json"):
            raise PermissionError("sidecar blocked")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _selective)


def _result_with_metrics(sample_question, sample_round, **overrides) -> DebateResult:
    return _pick_result(sample_question, sample_round, metrics=DebateMetrics(), **overrides)


def test_sidecar_failure_does_not_abort_the_transcript(
    tmp_path, sample_question, sample_round, monkeypatch
):
    """#63: save_to_file absorbs the sidecar failure and still returns its paths.

    Previously the exception propagated, so the orchestrator never reached
    save_verdict_package and no contract_version package was emitted at all.
    """
    result = _result_with_metrics(sample_question, sample_round)
    _fail_sidecar_write(monkeypatch)

    saved = save_to_file(result, tmp_path / "out")

    assert saved[0].exists()
    assert not list((tmp_path / "out").glob("*_metrics.json"))


def test_sidecar_failure_is_machine_readable_not_log_only(
    tmp_path, sample_question, sample_round, monkeypatch
):
    """#63: the failure rides the existing #26 degradation two-signal, not a log line alone.

    A consumer must be able to tell "metrics failed" from "metrics never produced". No new
    field or flag — degraded/degradation_summary are what _build_verdict_payload already
    serializes, and exit_semantics stays 0.
    """
    result = _result_with_metrics(sample_question, sample_round)
    assert result.degraded is False
    _fail_sidecar_write(monkeypatch)

    save_to_file(result, tmp_path / "out")

    assert result.degraded is True
    assert "metrics sidecar not written" in (result.degradation_summary or "")


def test_sidecar_failure_preserves_an_existing_degradation_summary(
    tmp_path, sample_question, sample_round, monkeypatch
):
    """#63: the sidecar note is appended, never clobbering a provider-degradation summary."""
    result = _result_with_metrics(
        sample_question, sample_round,
        degraded=True, degradation_summary="2 of 5 providers failed",
    )
    _fail_sidecar_write(monkeypatch)

    save_to_file(result, tmp_path / "out")

    assert "2 of 5 providers failed" in (result.degradation_summary or "")
    assert "metrics sidecar not written" in (result.degradation_summary or "")


def test_sidecar_failure_still_emits_verdict_package_carrying_the_degradation(
    tmp_path, sample_question, sample_round, monkeypatch
):
    """#63 end to end at the writer seam: package present AND the degradation is visible."""
    import json as _json

    result = _result_with_metrics(sample_question, sample_round)
    out = tmp_path / "out"
    _fail_sidecar_write(monkeypatch)

    transcript = save_to_file(result, out)[0]
    verdict = save_verdict_package(result, out, transcript, written={"transcript": [transcript]})

    data = _json.loads(verdict[0].read_text(encoding="utf-8"))
    assert data["contract_version"] == "1.0"
    assert data["exit_semantics"] == 0  # two-signal: exit 0 + degradation flag
    assert data["degradation"]["degraded"] is True
    assert "metrics sidecar not written" in data["degradation"]["summary"]


def test_verdict_manifest_omits_paths_not_on_disk(
    tmp_path, sample_question, sample_round
):
    """#63: the manifest never advertises a file a consumer would then fail to find."""
    import json as _json

    result = _pick_result(sample_question, sample_round)
    out = tmp_path / "out"
    transcript = save_to_file(result, out)[0]
    phantom = out / "council-out-phantom_metrics.json"  # never written

    verdict = save_verdict_package(
        result, out, transcript,
        written={"transcript": [transcript], "metrics": [phantom]},
    )

    data = _json.loads(verdict[0].read_text(encoding="utf-8"))
    kinds = {a["kind"] for a in data["artifacts"]}
    assert "transcript" in kinds
    assert "metrics" not in kinds  # dropped: the path does not exist
    for artifact in data["artifacts"]:
        for p in artifact["paths"]:
            if artifact["kind"] != "verdict":  # verdict lists its own not-yet-written paths
                assert Path(p).exists()


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


# ---------------------------------------------------------------------------
# #40 — options_considered extractor, verified against the REAL 2026-07-17
# night-batch artifacts (docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md
# §3.2 H2). Synthetic fixtures do NOT satisfy the #40 done-contract; these read the
# genuine transcripts copied verbatim into tests/fixtures/night_batch/.
# ---------------------------------------------------------------------------

_NIGHT_BATCH = Path(__file__).parent / "fixtures" / "night_batch"


def _slice_transcript(md: str) -> tuple[str, str]:
    """Recover (question_text, synthesis) from a real council-out transcript.

    Mirrors what _build_body wrote: question.text sits under the ``## Question`` wrapper
    (up to the first ``## Round``), and result.synthesis is the body under the
    ``## Synthesis (by …)`` wrapper — neither wrapper heading is part of the model field.
    """
    lines = md.splitlines()
    q_start = next(i for i, ln in enumerate(lines) if ln.strip() == "## Question")
    q_end = next(i for i, ln in enumerate(lines) if ln.startswith("## Round "))
    s_start = next(i for i, ln in enumerate(lines) if ln.startswith("## Synthesis"))
    question_text = "\n".join(lines[q_start + 1 : q_end]).strip()
    synthesis = "\n".join(lines[s_start + 1 :]).strip()
    return question_text, synthesis


def _real_result(fixture: str, sample_round, mode: str) -> DebateResult:
    md = (_NIGHT_BATCH / fixture).read_text(encoding="utf-8")
    question_text, synthesis = _slice_transcript(md)
    return DebateResult(
        question=Question(text=question_text, source=str(_NIGHT_BATCH / fixture)),
        rounds=[sample_round],
        synthesis=synthesis,
        synthesizer="openai",
        total_duration_sec=5.0,
        panel_mode="custom",
        synthesizer_is_participant=False,
        mode=mode,
    )


_PICK_FIXTURES = [
    "council-out-20260717_230406-pick-uc1-rama1-crux-grounding.md",
    "council-out-20260717_230852-pick-uc2-rama3-framing-defense.md",
    "council-out-20260717_231220-pick-uc3-deepseek-panel-disposition.md",
]


@pytest.mark.parametrize("fixture", _PICK_FIXTURES)
def test_options_pick_falls_back_to_question_alternatives(fixture, tmp_path, sample_round):
    """#40 empty-on-pick: the pick synthesis prescribes no options heading, so the field
    was []. Options now fall back to the debate QUESTION's own ``## Options`` section —
    the three (a)/(b)/(c) alternatives the panel actually chose among."""
    data = _emit(_real_result(fixture, sample_round, mode="pick"), tmp_path / "out")
    opts = data["options_considered"]
    assert opts["heading"] == "Options"          # sourced from the question, not the synthesis
    assert opts["source"] == "extraction"
    items = opts["items"]
    assert len(items) == 3                        # each UC pick question enumerates (a)/(b)/(c)
    assert items[0].startswith("(a)")
    assert all("**" not in it for it in items)    # no leaked emphasis


def test_options_ideas_top_tier_top_level_only_and_clean(tmp_path, sample_round):
    """#40 polluted-on-ideas: the extractor scooped indented sub-bullets ('Who endorsed it')
    and leaked trailing **. It now keeps only top-level Top-Tier ideas, emphasis stripped
    (verified against the REAL uc4 ideas transcript)."""
    data = _emit(
        _real_result(
            "council-out-20260717_231341-ideas-uc4-model-currency-detection.md",
            sample_round,
            mode="ideas",
        ),
        tmp_path / "out",
    )
    opts = data["options_considered"]
    assert opts["heading"] == "Top Tier (Implement Soon)"
    items = opts["items"]
    assert len(items) == 5                                       # the five top-level ideas only
    assert "Provider-specific API/model listing adapters" in items   # clean, no trailing **
    assert all("**" not in it for it in items)                  # emphasis stripped
    assert all("Who endorsed it" not in it for it in items)     # nested sub-bullet dropped


# ---------------------------------------------------------------------------
# #60 — a synthesis options heading with no bullets must not suppress the fallback
# ---------------------------------------------------------------------------

_PROSE_ONLY_OPTIONS = (
    "## Recommendation\nAdopt YAML.\n\n"
    "## Alternatives Considered\n"
    "The panel weighed the alternatives at length and converged quickly.\n"
)


def test_options_prose_only_heading_falls_back_to_question():
    """#60: the gate is 'no options extracted', not 'no section present'.

    A synthesis heading whose body is prose used to short-circuit the fallback, yielding
    items=[] under a heading that listed nothing.
    """
    from ai_council.output import _extracted_options, _split_sections

    question = "Which config format?\n\n## Options\n- (a) Keep YAML\n- (b) Move to TOML\n"
    opts = _extracted_options(
        _split_sections(_PROSE_ONLY_OPTIONS), question_sections=_split_sections(question)
    )
    assert opts["items"] == ["(a) Keep YAML", "(b) Move to TOML"]
    assert opts["heading"] == "Options"  # adopted from the question
    assert opts["source"] == "extraction"


def test_options_fallback_never_clobbers_a_real_synthesis_heading():
    """#60 clobber guard: a fallback that finds nothing must not overwrite the heading.

    Gating on items alone — without checking the fallback actually produced any — would
    rebind heading to None whenever the question carries no ## Options of its own.
    """
    from ai_council.output import _extracted_options, _split_sections

    question = "Which config format? No options section here.\n"
    opts = _extracted_options(
        _split_sections(_PROSE_ONLY_OPTIONS), question_sections=_split_sections(question)
    )
    assert opts["items"] == []
    assert opts["heading"] == "Alternatives Considered"  # the synthesis heading, not None


def test_options_prose_heading_does_not_hide_a_later_synthesis_section():
    """#60: a prose options heading must not skip PAST bulleted synthesis options.

    _first_by_priority returns the first matching section, so a prose
    ## Alternatives Considered followed by a bulleted ## Options used to fall straight
    through to the question's staler list. Raised by terra in adversarial review.
    """
    from ai_council.output import _extracted_options, _split_sections

    synthesis = (
        "## Alternatives Considered\nThe panel weighed them at length.\n\n"
        "## Options\n- Ship the shim\n- Rewrite the adapter\n"
    )
    question = "## Options\n- (a) Stale one\n- (b) Stale two\n"
    opts = _extracted_options(
        _split_sections(synthesis), question_sections=_split_sections(question)
    )
    assert opts["items"] == ["Ship the shim", "Rewrite the adapter"]  # synthesis, not question
    assert opts["heading"] == "Options"


def test_routing_failure_aggregate_chains_the_underlying_cause():
    """The accumulator path must chain a real traceback, like the direct path does."""
    from ai_council.output import OutputRoutingError, RoutingFailure, raise_for_routing_failures

    root = PermissionError("blocked")
    failures = [RoutingFailure("transcript", Path("x"), "PermissionError: blocked", root)]
    with pytest.raises(OutputRoutingError) as excinfo:
        raise_for_routing_failures(failures)
    assert excinfo.value.__cause__ is root


def test_options_synthesis_bullets_still_win_over_the_question():
    """#60 must not change the happy path: a synthesis WITH options is never overridden."""
    from ai_council.output import _extracted_options, _split_sections

    synthesis = "## Alternatives Considered\n- JSON: no comments\n- TOML: less familiar\n"
    question = "## Options\n- (a) Keep YAML\n- (b) Move to TOML\n"
    opts = _extracted_options(
        _split_sections(synthesis), question_sections=_split_sections(question)
    )
    assert opts["items"] == ["JSON: no comments", "TOML: less familiar"]
    assert opts["heading"] == "Alternatives Considered"


def test_verdict_decision_strips_wrapping_emphasis(tmp_path, sample_question, sample_round):
    """A bold-wrapped one-line decision is extracted clean (no leading/trailing ** )."""
    synthesis = "## Recommended Decision\n**Default to a monorepo with lightweight tooling.**\n"
    data = _emit(_pick_result(sample_question, sample_round, synthesis=synthesis), tmp_path / "out")
    assert data["decision"]["value"] == "Default to a monorepo with lightweight tooling."


def test_verdict_judge_decision_is_overall_verdict_not_recommendations(
    tmp_path, sample_question, sample_round
):
    """Judge mode: the decision must be ## Overall Verdict, not the first ## Recommendations item."""
    synthesis = (
        "## Overall Verdict\nThe design is sound and ready to ship.\n\n"
        "## Recommendations\n- Add a retry budget\n- Document the timeout\n"
    )
    result = _pick_result(sample_question, sample_round, synthesis=synthesis, mode="judge")
    data = _emit(result, tmp_path / "out")
    assert data["decision"]["heading"] == "Overall Verdict"
    assert data["decision"]["value"] == "The design is sound and ready to ship."


def test_verdict_minority_pointer_uses_actual_emitted_filename(
    tmp_path, sample_question, sample_round
):
    """Fix C: the minority pointer references the ACTUAL emitted file, even if its <ts> differs."""
    result = _pick_result(sample_question, sample_round, synthesis=_DISSENT_SYNTHESIS)
    out = tmp_path / "out"
    transcript = save_to_file(result, out)[0]
    # a minority file whose <ts> deliberately differs from the transcript stem
    minority = save_minority_report(result, out, stem_base="19990101_000000-pick-drift")
    data = _load_verdict(
        save_verdict_package(
            result, out, transcript, written={"minority": minority}
        )[0]
    )
    assert data["dissent"]["minority_artifact"] == minority[0].name
    assert "19990101_000000-pick-drift" in data["dissent"]["minority_artifact"]


def test_verdict_fails_loud_when_required_return_dir_unwritten(
    tmp_path, sample_question, sample_round
):
    """Fix D / R4: a required --return-dir that cannot be written raises, not a silent exit 0."""
    from ai_council.output import OutputRoutingError

    result = _pick_result(sample_question, sample_round)
    out = tmp_path / "out"
    transcript = save_to_file(result, out)[0]
    # point return_dir at an existing FILE so its mkdir fails inside _write_routed
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    with pytest.raises(OutputRoutingError):
        save_verdict_package(result, out, transcript, return_dir=blocker)


def test_verdict_dissent_unanimous(tmp_path, sample_question, sample_round):
    data = _emit(_pick_result(sample_question, sample_round), tmp_path / "out")
    assert data["dissent"]["status"] == "unanimous"
    assert data["dissent"]["minority_artifact"] is None
    assert data["dissent"]["source"] == "extraction"


def test_verdict_dissent_non_unanimous_points_to_minority(tmp_path, sample_question, sample_round):
    """Orchestrator path: minority is emitted and passed, so the pointer resolves to it."""
    result = _pick_result(sample_question, sample_round, synthesis=_DISSENT_SYNTHESIS)
    out = tmp_path / "out"
    transcript = save_to_file(result, out)[0]
    run_base = transcript.stem[len("council-out-"):]
    minority = save_minority_report(result, out, stem_base=run_base)
    data = _load_verdict(
        save_verdict_package(result, out, transcript, written={"minority": minority})[0]
    )
    assert data["dissent"]["status"] == "non-unanimous"
    assert data["dissent"]["minority_artifact"] == minority[0].name
    # gist is the dissent CONTENT, not an echo of the section heading
    assert "crux" in data["dissent"]["gist"].lower()
    assert data["dissent"]["gist"] != "Unresolved Disagreements"


def test_verdict_dissent_pointer_null_when_no_minority_emitted(tmp_path, sample_question, sample_round):
    """A direct caller that emits no minority gets a null pointer, never a fabricated dangling name."""
    result = _pick_result(sample_question, sample_round, synthesis=_DISSENT_SYNTHESIS)
    data = _emit(result, tmp_path / "out")  # no written["minority"] supplied
    assert data["dissent"]["status"] == "non-unanimous"
    assert data["dissent"]["minority_artifact"] is None
    assert data["dissent"]["gist"]  # dissent is still conveyed via the gist


def test_verdict_contract_version_stamped_and_exit_zero(tmp_path, sample_question, sample_round):
    """Contract-Version 1.0 stamped once §7 emptied (#22/#23 shipped) + completed debate exits 0."""
    data = _emit(_pick_result(sample_question, sample_round), tmp_path / "out")
    assert data["contract_version"] == "1.0"
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


def test_verdict_surfaces_a_seat_lost_mid_debate(tmp_path, sample_question, sample_round):
    """P1-8: a seat that answered Round 1 then died read as "ok", so it stayed out of
    panel.dropped and degradation.failed_providers entirely — mid-debate loss was invisible in
    every field a foreign repo consumes as binding input."""
    result = _pick_result(
        sample_question,
        sample_round,
        provider_statuses={"claude": SEAT_STATUS_OK, "deepseek": SEAT_STATUS_LOST},
        degraded=True,
        degradation_summary="1 seat(s) responded in an earlier round but were absent from the final round: deepseek.",
    )
    data = _emit(result, tmp_path / "out")
    assert data["panel"]["dropped"] == ["deepseek"]
    assert data["degradation"]["failed_providers"] == ["deepseek"]
    assert data["degradation"]["degraded"] is True
    assert "deepseek" in data["degradation"]["summary"]
    # seated stays ROUND-1 derived and unchanged — the seat did take its seat.
    assert "deepseek" in data["panel"]["requested"]


def test_verdict_lost_and_failed_both_count_as_dropped(tmp_path, sample_question, sample_round):
    """Both non-ok statuses reach the caller's dropped list; the ok seat never does."""
    result = _pick_result(
        sample_question,
        sample_round,
        provider_statuses={
            "claude": SEAT_STATUS_OK,
            "deepseek": SEAT_STATUS_LOST,
            "grok": SEAT_STATUS_FAILED,
        },
        degraded=True,
    )
    data = _emit(result, tmp_path / "out")
    assert data["panel"]["dropped"] == ["deepseek", "grok"]
    assert data["degradation"]["failed_providers"] == ["deepseek", "grok"]


def test_verdict_unknown_status_counts_as_dropped(tmp_path, sample_question, sample_round):
    """Drift guard: an unrecognized value must fall to DROPPED, never be silently seated.
    Matches the pessimistic seeding the producer has always used."""
    result = _pick_result(
        sample_question,
        sample_round,
        provider_statuses={"claude": SEAT_STATUS_OK, "deepseek": "some-future-status"},
    )
    data = _emit(result, tmp_path / "out")
    assert data["panel"]["dropped"] == ["deepseek"]


def test_verdict_field_set_identical_with_and_without_a_lost_seat(
    tmp_path, sample_question, sample_round
):
    """CONTRACT GUARD (P1-8): the "lost" status is a VALUE correction inside the existing field
    set — it must add, remove or rename nothing. Mirrors the crux Phase-A freeze test below."""
    ok_only = _pick_result(
        sample_question, sample_round, provider_statuses={"claude": SEAT_STATUS_OK}
    )
    with_lost = _pick_result(
        sample_question,
        sample_round,
        provider_statuses={"claude": SEAT_STATUS_OK, "deepseek": SEAT_STATUS_LOST},
        degraded=True,
    )
    a = _emit(ok_only, tmp_path / "a")
    b = _emit(with_lost, tmp_path / "b")

    assert a.keys() == b.keys()
    assert a["panel"].keys() == b["panel"].keys()
    assert a["degradation"].keys() == b["degradation"].keys()
    assert a["contract_version"] == b["contract_version"] == "1.0"
    assert a["exit_semantics"] == b["exit_semantics"] == 0
    # The status vocabulary itself must never leak into the package as a value.
    assert SEAT_STATUS_LOST not in json.dumps(b)


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
    # panel.seats carries the FULL canonical L-CLI shape (no parallel subset schema): the
    # same keys the _metrics.json sidecar emits via the shared _seat_payload serializer.
    pseat = data["panel"]["seats"][0]
    assert pseat["requested_backend"] == "cli"
    assert pseat["actual_backend"] == "api"
    assert pseat["cli"] == {"name": "claude", "version": "1.2.3"}
    assert pseat["identity_channel"] == "stderr-banner"
    assert pseat["fallback_events"][0]["cause"] == "process-error"


def test_verdict_panel_seats_shape_matches_metrics_sidecar(tmp_path, sample_question, sample_round):
    """The verdict panel.seats keys are identical to the _metrics.json seats[] keys (one source)."""
    seat = SeatMetrics(
        seat="openai",
        requested_backend="api",
        actual_backend="api",
        requested_model="gpt-5.4",
        actual_model="gpt-5.4",
        identity_channel="api-echo",
        identity_readable=True,
    )
    result = _pick_result(sample_question, sample_round, metrics=DebateMetrics(seats=[seat]))
    out = tmp_path / "out"
    transcript = save_to_file(result, out)[0]  # also emits _metrics.json (seats[])
    data = _load_verdict(save_verdict_package(result, out, transcript)[0])
    metrics = json.loads(
        (out / (transcript.stem + "_metrics.json")).read_text(encoding="utf-8")
    )
    assert data["panel"]["seats"][0].keys() == metrics["seats"][0].keys()


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
    out = tmp_path / "out"
    ret = tmp_path / "return"
    saved_paths = save_to_file(result, out, return_dir=ret)
    written = {"transcript": saved_paths}
    verdict = save_verdict_package(
        result, out, saved_paths[0], written=written, return_dir=ret
    )
    data = _load_verdict(verdict[0])
    kinds = {a["kind"] for a in data["artifacts"]}
    assert "transcript" in kinds
    assert "verdict" in kinds
    transcript_entry = next(a for a in data["artifacts"] if a["kind"] == "transcript")
    assert transcript_entry["filename"] == saved_paths[0].name
    # the verdict entry lists its GUARANTEED destinations (canonical + verified return_dir),
    # and every listed path actually exists on disk (never a planned-but-failed claim)
    verdict_entry = next(a for a in data["artifacts"] if a["kind"] == "verdict")
    assert len(verdict_entry["paths"]) == 2
    assert all(Path(p).exists() for p in verdict_entry["paths"])


def test_verdict_manifest_excludes_best_effort_target_mirrors(tmp_path, sample_question, sample_round):
    """A best-effort --target mirror is NOT claimed in the verdict's own paths (no overclaim)."""
    result = _pick_result(sample_question, sample_round)
    out = tmp_path / "out"
    target = tmp_path / "mirror"
    transcript = save_to_file(result, out, target_paths=[target])[0]
    data = _load_verdict(
        save_verdict_package(result, out, transcript, target_paths=[target])[0]
    )
    verdict_entry = next(a for a in data["artifacts"] if a["kind"] == "verdict")
    assert verdict_entry["paths"] == [str(out / verdict_entry["filename"])]  # canonical only


# ---------------------------------------------------------------------------
# OutputRoutingError input guard — regression for the A1/A2 integration seam
# ---------------------------------------------------------------------------

def test_output_routing_error_rejects_wrong_input_types():
    """A wrong type must raise TypeError, never be shredded into fake deliverables.

    Caught at integration: str/bytes are iterable, so the original ``list(failures)`` turned
    a message string into one "deliverable" per character. Four call sites still passed a
    string and two of their assertions survived the mangling, so the lane gates stayed green
    while the operator-facing message was nonsense.
    """
    import pytest

    from ai_council.output import OutputRoutingError, RoutingFailure

    good = RoutingFailure(artifact="transcript", destination=Path("/nope"), cause="denied")
    for bad in ("a message string", b"bytes", good, ["transcript"], None, 42):
        with pytest.raises(TypeError, match="takes a list of RoutingFailure"):
            OutputRoutingError(bad)


def test_output_routing_error_message_reports_true_count():
    """The count in the message is the number of failures passed in — 2 is 2, not 58."""
    from ai_council.output import OutputRoutingError, RoutingFailure

    dest = Path("/nope")
    two = OutputRoutingError([
        RoutingFailure(artifact="transcript", destination=dest, cause="denied"),
        RoutingFailure(artifact="verdict package", destination=dest, cause="denied"),
    ])
    assert len(two.failures) == 2
    assert "2 deliverables not delivered" in str(two)

    one = OutputRoutingError([RoutingFailure(artifact="research report", destination=dest, cause="denied")])
    assert "1 deliverable not delivered" in str(one)


# ---------------------------------------------------------------------------
# F8 / F2 — regressions introduced by A1's merged diff, caught by the sol
# adversarial pass and confirmed by differential run against 27a45d1.
# ---------------------------------------------------------------------------

def test_later_considered_section_does_not_beat_question_options():
    """F8: the exact document that regressed. 27a45d1 -> items=[]; A1's merged diff ->
    ['Risk one'] under heading 'Risks Considered'. Risks must never surface as options.
    """
    from ai_council.output import _extracted_options, _split_sections

    synthesis = (
        "## Alternatives Considered\n\nProse only, no bullets.\n\n"
        "## Risks Considered\n\n- Risk one\n- Risk two\n"
    )
    question = "## Options\n\n- Real option A\n- Real option B\n"
    got = _extracted_options(_split_sections(synthesis), _split_sections(question))

    assert got["items"] == ["Real option A", "Real option B"], (
        "a later '...Considered' section was promoted over the question fallback"
    )
    assert "Risk one" not in got["items"]


def test_risks_considered_alone_yields_no_options():
    """F8, the case that proves narrowing the SCAN would not have been enough.

    When '## Risks Considered' is the ONLY options-ish heading it is also the FIRST
    match, so any fix that merely restricts continue-scanning still emits risks as
    options. Only removing the bare 'considered' marker fixes this.
    """
    from ai_council.output import _extracted_options, _split_sections

    synthesis = "## Risks Considered\n\n- Risk one\n- Risk two\n"
    got = _extracted_options(_split_sections(synthesis), None)

    assert got["items"] == [], f"risks emitted as options_considered: {got['items']}"


def test_direct_mode_writes_canonical_metrics_before_raising(
    tmp_path, sample_question, sample_round
):
    """F2: a direct-mode caller must not lose a CANONICAL artifact to a return-dir failure.

    This lane's invariant is that every canonical write lands before any raise. The
    orchestrator's accumulator mode honoured it; direct mode raised first and skipped the
    metrics sidecar. The invariant is the contract, not 'production is unaffected'.
    """
    import pytest

    from ai_council.output import OutputRoutingError, save_to_file

    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    out = tmp_path / "canonical"

    result = _result_with_metrics(sample_question, sample_round)
    with pytest.raises(OutputRoutingError):
        save_to_file(result, out, return_dir=blocker)

    transcripts = list(out.glob("council-out-*.md"))
    sidecars = list(out.glob("*_metrics.json"))
    assert len(transcripts) == 1, "canonical transcript lost"
    assert len(sidecars) == 1, "canonical metrics sidecar lost to the raise (F2)"


# ---------------------------------------------------------------------------
# #77 -- options_considered as ONE contract (sol adversarial pass, F6+F7).
# These tests are the ex-ante contract: written BEFORE the fix, one named test
# per frozen acceptance rule. The extractor feeds the DELEGATION SURFACE, so a
# corrupted option string is read by a consuming repo as the council's own words.
# ---------------------------------------------------------------------------

def test_options_bullet_grammar_accepts_every_marker():
    """Rule 1: `-`, `*`, `+`, `1.`, `1)` all parse to clean items.

    `+ item` and `1) item` used to yield [] -- the character-class test only knew
    `-`/`*` and a bare-digit first token.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- dash\n* star\n+ plus\n1. dot\n2) paren\n") == [
        "dash",
        "star",
        "plus",
        "dot",
        "paren",
    ]


def test_options_marker_removal_never_eats_payload():
    """Rule 2: only the EXACT list marker is removed, never a payload character.

    `lstrip("-*0123456789. ")` is a character-set strip, so it chewed through the
    option's own leading digits: `- 3D printing` -> `D printing`.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- 3D printing\n- 2026 roadmap\n- 401k match\n") == [
        "3D printing",
        "2026 roadmap",
        "401k match",
    ]


def test_options_numbered_marker_removal_keeps_numeric_payload():
    """Rule 2, numbered form: `1. 2026 roadmap` keeps its 2026."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("1. 2026 roadmap\n2) 3D printing\n") == [
        "2026 roadmap",
        "3D printing",
    ]


def test_options_emphasis_unwrapped_as_markdown_delimiters():
    """Rule 3: emphasis is unwrapped as real paired delimiters, not edge-stripped.

    `.strip("*`_")` only touched the ends, so `- **Alpha** - fast` kept an interior
    `**`: `Alpha** - fast`.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets(
        "- **Alpha** - fast\n- *Beta* - cheap\n- `Gamma` - proven\n- __Delta__ - safe\n"
    ) == ["Alpha - fast", "Beta - cheap", "Gamma - proven", "Delta - safe"]


def test_options_emphasis_unwrapping_spares_intra_word_underscores():
    """Rule 3 guard: `_` inside an identifier is payload, not a delimiter.

    A naive paired-delimiter regex turns `snake_case_name` into `snakecasename`.
    Unwrapping must never fire mid-word.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- keep snake_case_name intact\n- 3 * 4 is not emphasis\n") == [
        "keep snake_case_name intact",
        "3 * 4 is not emphasis",
    ]


def test_options_honest_empty_when_no_bullets_anywhere():
    """Rule 5: no options -> [], never a plausible-wrong single item.

    An honest [] is readable as 'none extracted'; a fabricated item is not
    detectable by the consumer.
    """
    from ai_council.output import _extracted_options, _split_sections

    synthesis = "## Alternatives Considered\n\nThe panel converged without listing any.\n"
    got = _extracted_options(_split_sections(synthesis), _split_sections("No options here.\n"))
    assert got["items"] == []


def test_options_value_shape_is_unchanged():
    """Rule 6: the {items, source, heading} triple the :974 caller depends on."""
    from ai_council.output import _extracted_options, _split_sections

    got = _extracted_options(_split_sections("## Options\n- One\n- Two\n"), None)
    assert set(got) == {"items", "source", "heading"}
    assert got["items"] == ["One", "Two"]
    assert got["source"] == "extraction"
    assert got["heading"] == "Options"


def test_options_nested_sub_bullets_still_dropped():
    """Regression guard: indented annotations are not scooped as their own options."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- Alpha\n  - Who endorsed it: gemini\n- Beta\n") == ["Alpha", "Beta"]


def test_options_thematic_break_is_not_an_option():
    """Rule 5 guard: a horizontal rule must not surface as a junk option.

    `---` fails the bullet grammar outright, but the spaced form `* * *` parses as a
    `*` bullet carrying `* *`. Honest-empty beats a junk item on the delegation surface.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("---\n* * *\n___\n- Real option\n") == ["Real option"]


# --- #77, terra pre-merge review: three HIGH findings against the first fix. ---

def test_options_code_span_is_atomic():
    """terra HIGH: a backtick span must survive whole.

    Unwrapping the backticks and re-scanning the result ate the dunder --
    `__init__` became `init`. Payload loss on the delegation surface.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- Refactor `__init__` wiring\n- Use `a * b` form\n") == [
        "Refactor __init__ wiring",
        "Use a * b form",
    ]


def test_options_escaped_delimiters_are_literal_not_emphasis():
    r"""terra HIGH: `\*literal\*` is an author asking for asterisks, not emphasis."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets(r"- keep \*literal\* stars" "\n" r"- a \_b\_ c" "\n") == [
        "keep *literal* stars",
        "a _b_ c",
    ]


def test_options_unpaired_and_unequal_delimiters_are_left_alone():
    """terra HIGH: `x *** y` lost an asterisk. Only equal-length runs pair."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- x *** y\n- unmatched ** here\n- **uneven*\n") == [
        "x *** y",
        "unmatched ** here",
        "**uneven*",
    ]


def test_options_list_wrapped_thematic_break_is_dropped():
    """terra HIGH: `- * * *` is a list-wrapped separator, not an option.

    The line-level guard cannot see it -- the break test must also run on the
    payload after marker removal.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- * * *\n* - - -\n- Real option\n") == ["Real option"]


def test_options_emphasis_unwrapping_is_linear_on_pathological_input():
    """terra HIGH: the lazy-quantifier regex was quadratic (~4.4s on 30k chars).

    A model may emit a very long single-line option; unwrapping must not stall
    verdict generation. Generous bound -- it fails on quadratic, passes on linear.
    """
    import time

    from ai_council.output import _top_level_bullets

    started = time.perf_counter()
    items = _top_level_bullets("- " + " *a" * 30_000 + "\n")
    elapsed = time.perf_counter() - started

    assert len(items) == 1
    assert elapsed < 1.0, f"emphasis unwrapping took {elapsed:.2f}s -- superlinear"


def test_options_nested_emphasis_still_unwraps():
    """Regression guard for the scanner rewrite: nesting must still resolve."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- **_Alpha_** wins\n- ***Bold italic*** option\n") == [
        "Alpha wins",
        "Bold italic option",
    ]


# --- #77, terra pass 2: three HIGH findings against the scanner itself. ---

def test_options_longer_backtick_run_is_not_a_shorter_fence_closer():
    """terra pass-2 HIGH: a 1-backtick opener must not pair with the third
    backtick of a ``` run and swallow the other two.

    Searching for the fence SUBSTRING restarted inside the longer run; pairing
    maximal runs by equal length is what makes it unusable as a closer.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- a `b ``` c\n") == ["a `b ``` c"]
    assert _top_level_bullets("- a ``b ````` c\n") == ["a ``b ````` c"]


def test_options_code_span_pairing_is_equal_length_only():
    """Equal-length runs pair; the inner longer run is content, not a delimiter."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- `x ``y`` z`\n") == ["x ``y`` z"]
    assert _top_level_bullets("- ``a ` b``\n") == ["a ` b"]


def test_options_unmatched_flanked_delimiters_stay_linear():
    """terra pass-2 HIGH: the depth cap was bypassed on the close-and-open path.

    `!_!*!` runs can both close and open and never match each other, so they grew
    the opener stack without bound and each reverse search walked all of it. The
    earlier timing test used ` *a`, whose delimiters have close=False, so it did
    NOT exercise this path -- the gap terra identified.
    """
    import time

    from ai_council.output import _top_level_bullets

    started = time.perf_counter()
    items = _top_level_bullets("- " + "!_!*" * 20_000 + "\n")
    elapsed = time.perf_counter() - started

    assert len(items) == 1
    assert elapsed < 1.0, f"unmatched flanked delimiters took {elapsed:.2f}s -- superlinear"


def test_options_descending_backtick_fences_stay_linear():
    """terra pass-2 HIGH: every unmatched run re-walked the suffix via find()."""
    import time

    from ai_council.output import _top_level_bullets

    payload = "".join("`" * length + "x" for length in range(40, 0, -1)) * 60
    started = time.perf_counter()
    _top_level_bullets("- " + payload + "\n")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"descending fences took {elapsed:.2f}s -- superlinear"


# --- #77, terra pass 3: three HIGH findings. ---

def test_options_escaped_backtick_does_not_steal_a_code_span():
    r"""terra pass-3 HIGH: the code-span pre-pass was escape-blind.

    An escaped backtick opened a span that swallowed the real `__init__` span
    after it, and the dunder was then eaten as emphasis. The pre-pass and the
    main scan must agree on which characters are delimiters at all.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets(r"- \` `__init__` wiring" "\n") == ["` __init__ wiring"]


def test_options_punctuation_flanked_delimiters_are_literal():
    """terra pass-3 HIGH: `a*.*b` has no emphasis but collapsed to `a.b`.

    A whitespace-only flanking test deleted literal payload; CommonMark also
    requires the punctuation-flanking conditions.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- a*.*b\n- x*,*y\n") == ["a*.*b", "x*,*y"]


def test_options_ordered_marker_is_one_to_nine_ascii_digits():
    r"""terra pass-3 HIGH: `\d+` fabricated an option out of long-number prose.

    A CommonMark ordered marker is 1-9 ASCII digits, and `\d` also matched
    non-ASCII digits.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("1234567890. ordinary prose\n") == []
    assert _top_level_bullets("٢. prose with an arabic-indic digit\n") == []
    assert _top_level_bullets("123456789. still a real item\n") == ["still a real item"]


def test_options_emphasis_happy_paths_survive_the_flanking_rules():
    """Regression guard: tightening flanking must not stop real emphasis unwrapping."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets(
        "- **Alpha** - fast\n- *Beta* - cheap\n- __Delta__ - safe\n- 50% *off*\n"
    ) == ["Alpha - fast", "Beta - cheap", "Delta - safe", "50% off"]


# --- #77, terra pass 4: the scanner could HANG, plus three payload defects. ---

def test_options_backslash_that_escapes_nothing_terminates():
    r"""terra pass-4: `- C:\Users\rob` HUNG the extractor forever.

    A backslash not followed by escapable punctuation fell through to the
    fallback branch, whose scan stops AT a backslash without advancing -- so the
    loop spun on the same index. On a Windows-first repo an option naming an
    ordinary path would have hung verdict generation. Every branch must consume
    at least one character.
    """
    import time

    from ai_council.output import _top_level_bullets

    started = time.perf_counter()
    got = _top_level_bullets("- C:\\Users\\rob\n- trailing\\\n")
    elapsed = time.perf_counter() - started

    assert got == ["C:\\Users\\rob", "trailing\\"]
    assert elapsed < 1.0, f"took {elapsed:.2f}s -- the scanner is not terminating promptly"


def test_options_escaped_backtick_can_still_close_a_code_span():
    r"""terra pass-4: backslash escapes do NOT apply inside a code span.

    `` `C:\` `` is the path `C:\`. Skipping the escaped closing backtick left the
    span open and dropped the trailing backslash from the payload. An escaped run
    cannot OPEN a span (pass 3) but must still be able to CLOSE one.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- `C:\\`\n") == ["C:\\"]


def test_options_unicode_symbols_count_as_flanking_punctuation():
    """terra pass-4: `a*EURO*b` lost both asterisks.

    CommonMark flanking treats Unicode symbol categories (S*) as punctuation, not
    just P*. Without S* a currency or math symbol looked like an ordinary letter,
    so the run appeared to open real emphasis and literal payload was deleted.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- a*€*b\n- x*±*y\n") == ["a*€*b", "x*±*y"]


def test_options_marker_separator_must_be_ascii_space_or_tab():
    """terra pass-4: `\\s+` accepted a non-breaking space after the marker.

    A hyphen followed by U+00A0 is not a Markdown list item, so treating it as
    one fabricated an option out of ordinary prose.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("-\u00a0ordinary prose\n") == []
    assert _top_level_bullets("-\u2003em-space prose\n") == []
    assert _top_level_bullets("-\treal tab item\n") == ["real tab item"]


def test_options_extraction_is_total_over_random_markdown():
    r"""Fuzz guard: the extractor must TERMINATE and never raise, on any input.

    Pass 4 found a scanner that looped forever on `- C:\Users\rob`. Every named
    test in this file probes a case someone already thought of; the hang survived
    three review passes because nobody thought of a bare Windows path. This test
    is the systemic guard -- it does not know what the right answer is, only that
    there must BE one, promptly.

    Deterministic seed so a failure is reproducible.
    """
    import random
    import time

    from ai_council.output import _top_level_bullets

    rng = random.Random(20260720)
    # Weighted toward the characters that carry parsing meaning.
    alphabet = "\\`*_-+#.)( abc123" + "\t\u00a0\u2003\u20ac\u00b1\u2014\u00a7"
    for _ in range(3000):
        body = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        started = time.perf_counter()
        items = _top_level_bullets(body)
        elapsed = time.perf_counter() - started
        assert isinstance(items, list)
        assert all(isinstance(item, str) and item for item in items)
        assert elapsed < 0.5, f"slow/hanging on {body!r}: {elapsed:.2f}s"


def test_options_extraction_never_invents_characters():
    """Fuzz guard: extraction only REMOVES markup, it never adds content.

    Every character of an extracted item must appear in the source line, in
    order. This is the property the original defect violated -- `- 3D printing`
    yielding `D printing` is a subsequence, but `2026 roadmap` -> `roadmap`
    losing payload is what the ordering check makes visible when combined with
    the named tests above. Here it guards against the inverse: fabrication.
    """
    import random

    from ai_council.output import _top_level_bullets

    rng = random.Random(20260721)
    alphabet = "\\`*_-+123 abc.()"
    for _ in range(2000):
        line = "- " + "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        for item in _top_level_bullets(line):
            pos = 0
            for ch in item:
                pos = line.find(ch, pos)
                assert pos != -1, f"extraction invented {ch!r} for {line!r} -> {item!r}"
                pos += 1


# --- #77, terra pass 5: Unicode whitespace as false structure, plus fence merging. ---

def test_options_escaped_backtick_does_not_merge_with_the_following_fence():
    r"""terra pass-5: only ONE backtick is escaped.

    Absorbing the whole following run merged the literal backtick with the real
    fence after it, so in `\```x`` the fence was mis-sized and the payload was
    exposed to emphasis removal.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- \\```__init__``\n") == ["`__init__"]


def test_options_unicode_line_separator_does_not_create_a_line():
    """terra pass-5: U+2028 is not a Markdown line ending.

    `str.splitlines()` breaks on it, so inline payload containing one was split
    into a fresh line whose tail then parsed as a list item -- an option
    fabricated out of prose.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("ordinary prose\u2028- fabricated option\n") == []
    assert _top_level_bullets("prose\u2029- also fabricated\n") == []


def test_options_leading_unicode_whitespace_does_not_expose_a_marker():
    """terra pass-5: `raw.strip()` removed leading Unicode whitespace.

    A line opening with U+00A0 then a hyphen had the NBSP stripped, exposing the
    hyphen as a top-level marker where the Markdown line has none.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("\u00a0- ordinary prose\n") == []
    assert _top_level_bullets("\u2003- em-space prose\n") == []


def test_options_nbsp_separated_asterisks_are_payload_not_a_break():
    """terra pass-5: the break regex still used `\\s*` after _BULLET_RE was narrowed.

    NBSP-separated asterisks are option payload, so classifying them as a
    thematic break DELETED the whole option. Narrowing one regex and not the
    other was an internal inconsistency.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- *\u00a0*\u00a0*\n") == ["*\u00a0*\u00a0*"]
    assert _top_level_bullets("- * * *\n") == []  # ASCII-separated: still a real break


def test_options_handles_every_commonmark_line_ending():
    """Regression guard for the line-splitting change: LF, CRLF and bare CR."""
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- one\n- two") == ["one", "two"]
    assert _top_level_bullets("- one\r\n- two") == ["one", "two"]
    assert _top_level_bullets("- one\r- two") == ["one", "two"]


# --- #77, terra pass 6. ---

def test_options_multi_backtick_span_keeps_a_trailing_backslash():
    r"""terra pass-6: ``C:\`` lost its backslash and kept its fences.

    Splitting a backslash-adjacent run to model the escape mis-sized the CLOSING
    fence, so the span never closed and the backslash was then read as an escape.
    Escapes do not apply inside a code span, so a closer always uses its full run
    length; the escape only ever affects the OPENING side.

    This was a regression against main, which preserved the text (its edge-only
    strip never touched interior markup).
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- Use ``C:\\`` path\n") == ["Use C:\\ path"]
    assert _top_level_bullets("- ```a\\``` b\n") == ["a\\ b"]


def test_options_escaped_backtick_opener_rules_hold_together():
    r"""The three escape/code-span rules from passes 3-6 must all hold at once.

    An escaped run cannot OPEN a span; it can still CLOSE one; and only its FIRST
    backtick is escaped, so any remainder can open. Each was fixed in a different
    pass, and each fix risked reintroducing the previous one.
    """
    from ai_council.output import _top_level_bullets

    assert _top_level_bullets("- \\` `x` y\n") == ["` x y"]        # cannot open (pass 3)
    assert _top_level_bullets("- `C:\\`\n") == ["C:\\"]            # can still close (pass 4)
    assert _top_level_bullets("- \\```x``\n") == ["`x"]            # remainder opens (pass 5)
    assert _top_level_bullets("- ``C:\\`` z\n") == ["C:\\ z"]      # closer keeps length (pass 6)


# --- #18 Phase A: the crux artifact must stay OUT of the verdict package ---


def _result_with_crux(sample_question, sample_round):
    """A DebateResult carrying a fully-populated crux artifact."""
    from ai_council.models import CruxArtifact, CruxStatus

    result = _result_with_synthesis(
        "## Decision\nDo not deploy on Fridays.\n", sample_question, sample_round
    )
    result.crux = CruxArtifact(
        status=CruxStatus.GROUNDED,
        crux_claim="Deploys fail more on Fridays.",
        evidence_block="--- Evidence ---\nIncident rate is 12% higher.",
        sources_count=3,
        providers_succeeded=1,
        providers_attempted=1,
    )
    return result


def test_verdict_payload_contract_version_unchanged_by_crux(tmp_path, sample_question, sample_round):
    """Phase A freeze: a crux-carrying result still stamps contract_version 1.0."""
    payload = _build_verdict_payload(
        _result_with_crux(sample_question, sample_round), "run-1", "f.md", {}, [tmp_path]
    )
    assert payload["contract_version"] == "1.0"


def test_verdict_payload_has_no_crux_key(tmp_path, sample_question, sample_round):
    """The artifact rides on DebateResult only — it must not reach the sealed package."""
    payload = _build_verdict_payload(
        _result_with_crux(sample_question, sample_round), "run-1", "f.md", {}, [tmp_path]
    )
    assert "crux" not in payload
    for key in payload:
        assert "crux" not in key.lower()


def test_verdict_payload_body_carries_no_crux_evidence(tmp_path, sample_question, sample_round):
    """Deep check: the evidence text must not leak into ANY nested payload value."""
    import json

    payload = _build_verdict_payload(
        _result_with_crux(sample_question, sample_round), "run-1", "f.md", {}, [tmp_path]
    )
    blob = json.dumps(payload, default=str)
    assert "Incident rate is 12% higher." not in blob
    assert "Deploys fail more on Fridays." not in blob


def test_verdict_payload_identical_with_and_without_crux(tmp_path, sample_question, sample_round):
    """The strongest form: setting .crux changes the package by exactly nothing.

    ``timestamp`` is excluded because each _build_verdict_payload call stamps its own
    _iso_now(). Comparing it made the assertion depend on the platform clock: it passes on
    Windows only because datetime.now() has ~15.6ms resolution, and would be flaky on a
    microsecond-resolution clock. Every other field is compared.
    """
    import json

    with_crux = _result_with_crux(sample_question, sample_round)
    without = _result_with_crux(sample_question, sample_round)
    without.crux = None

    a = _build_verdict_payload(with_crux, "run-1", "f.md", {}, [tmp_path])
    b = _build_verdict_payload(without, "run-1", "f.md", {}, [tmp_path])

    assert a.keys() == b.keys()
    a.pop("timestamp", None)
    b.pop("timestamp", None)
    assert json.dumps(a, default=str, sort_keys=True) == json.dumps(
        b, default=str, sort_keys=True
    )
