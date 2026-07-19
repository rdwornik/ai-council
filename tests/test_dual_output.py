"""Tests for dual output path behavior (primary + optional secondary)."""

import logging

import pytest

from ai_council.models import DebateResult
from ai_council.output import save_to_file
from ai_council.research.models import MergedResearchReport
from ai_council.research.output import save_research_to_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result(sample_question, sample_round) -> DebateResult:
    return DebateResult(
        question=sample_question,
        rounds=[sample_round],
        synthesis="## Consensus\nAll agreed.",
        synthesizer="openai",
        total_duration_sec=10.0,
        panel_mode="default",
        synthesizer_is_participant=False,
    )


@pytest.fixture
def sample_research_report() -> MergedResearchReport:
    return MergedResearchReport(
        query="Best databases 2026",
        results=[],
        merged_report="merged content",
        summary_2500="summary",
        total_cost_usd=0.01,
        total_duration_sec=5.0,
        total_sources=3,
        cache_key="abc123",
    )


# ---------------------------------------------------------------------------
# save_to_file — debate transcripts
# ---------------------------------------------------------------------------

def test_primary_always_written(tmp_path, sample_result):
    saved = save_to_file(sample_result, tmp_path / "primary")
    assert len(saved) == 1
    assert saved[0].exists()
    assert saved[0].suffix == ".md"


def test_secondary_written_when_dir_exists(tmp_path, sample_result):
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    saved = save_to_file(sample_result, tmp_path / "primary", secondary_dir=secondary)
    assert len(saved) == 2
    assert saved[1].parent == secondary
    assert saved[1].exists()


def test_secondary_skipped_when_dir_missing(tmp_path, sample_result):
    missing = tmp_path / "does_not_exist"
    saved = save_to_file(sample_result, tmp_path / "primary", secondary_dir=missing)
    assert len(saved) == 1


def test_secondary_disabled_when_none(tmp_path, sample_result):
    saved = save_to_file(sample_result, tmp_path / "primary", secondary_dir=None)
    assert len(saved) == 1


def test_both_files_have_identical_content(tmp_path, sample_result):
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    saved = save_to_file(sample_result, tmp_path / "primary", secondary_dir=secondary)
    assert saved[0].read_text(encoding="utf-8") == saved[1].read_text(encoding="utf-8")


def test_both_files_have_same_filename(tmp_path, sample_result):
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    saved = save_to_file(sample_result, tmp_path / "primary", secondary_dir=secondary)
    assert saved[0].name == saved[1].name


def test_secondary_missing_logs_warning(tmp_path, sample_result, caplog):
    missing = tmp_path / "does_not_exist"
    with caplog.at_level(logging.WARNING, logger="ai_council.output"):
        save_to_file(sample_result, tmp_path / "primary", secondary_dir=missing)
    assert any("not found" in r.message for r in caplog.records)


def test_primary_dir_created_if_needed(tmp_path, sample_result):
    primary = tmp_path / "nested" / "new" / "dir"
    assert not primary.exists()
    save_to_file(sample_result, primary)
    assert primary.exists()


# ---------------------------------------------------------------------------
# save_research_to_file — research reports
# ---------------------------------------------------------------------------

def test_research_primary_always_written(tmp_path, sample_research_report):
    saved = save_research_to_file(sample_research_report, tmp_path / "primary")
    assert len(saved) == 1
    assert saved[0].exists()
    assert saved[0].name.startswith("council-out-")
    assert "-research-" in saved[0].name


def test_research_secondary_written_when_dir_exists(tmp_path, sample_research_report):
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    saved = save_research_to_file(
        sample_research_report, tmp_path / "primary", secondary_dir=secondary
    )
    assert len(saved) == 2
    assert saved[1].parent == secondary
    assert saved[1].exists()


def test_research_secondary_skipped_when_missing(tmp_path, sample_research_report):
    missing = tmp_path / "does_not_exist"
    saved = save_research_to_file(
        sample_research_report, tmp_path / "primary", secondary_dir=missing
    )
    assert len(saved) == 1


def test_research_both_files_identical(tmp_path, sample_research_report):
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    saved = save_research_to_file(
        sample_research_report, tmp_path / "primary", secondary_dir=secondary
    )
    assert saved[0].read_text(encoding="utf-8") == saved[1].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# #23: research honors --return-dir (canonical ./output/ always written)
# ---------------------------------------------------------------------------

def test_research_return_dir_written(tmp_path, sample_research_report):
    """#23: a research commission lands a copy in the caller's return dir (auto-mkdir)."""
    primary = tmp_path / "primary"
    return_dir = tmp_path / "caller_return"  # does not exist yet -> auto-mkdir
    saved = save_research_to_file(sample_research_report, primary, return_dir=return_dir)

    # canonical is always written first and present
    assert saved[0].parent == primary
    assert saved[0].exists()

    # the return-dir copy is present, same filename, identical content
    return_paths = [p for p in saved if p.parent == return_dir]
    assert len(return_paths) == 1
    assert return_paths[0].exists()
    assert return_paths[0].name == saved[0].name
    assert return_paths[0].read_text(encoding="utf-8") == saved[0].read_text(encoding="utf-8")


def test_research_return_dir_is_additive_canonical_unchanged(tmp_path, sample_research_report):
    """#23: return-dir is a copy, never a replacement — canonical ./output/ is unchanged."""
    primary = tmp_path / "primary"
    return_dir = tmp_path / "caller_return"
    saved = save_research_to_file(sample_research_report, primary, return_dir=return_dir)

    # canonical file exists in the primary dir regardless of return_dir
    canonical = [p for p in saved if p.parent == primary]
    assert len(canonical) == 1
    assert canonical[0].exists()
    # exactly canonical + return-dir here (no secondary/targets configured)
    assert len(saved) == 2


# ---------------------------------------------------------------------------
# #62: research R4 parity — a required return-dir miss is never swallowed
# ---------------------------------------------------------------------------

def test_research_required_return_dir_failure_raises(tmp_path, sample_research_report):
    """#62: before this, the research path had no required-destination check at all.

    A mkdir/write failure was logged and swallowed, so run_research returned success with
    the commissioned report absent. Now it raises, and the canonical copy still lands.
    """
    from ai_council.output import OutputRoutingError

    primary = tmp_path / "primary"
    # point return_dir at an existing FILE so its mkdir fails inside _write_routed
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")

    with pytest.raises(OutputRoutingError) as excinfo:
        save_research_to_file(sample_research_report, primary, return_dir=blocker)

    assert [f.artifact for f in excinfo.value.failures] == ["research report"]
    assert len(list(primary.glob("council-out-*research*.md"))) == 1


def test_research_required_return_dir_failure_recorded_when_accumulating(
    tmp_path, sample_research_report
):
    """#62: with an accumulator the writer records and returns, matching the debate path."""
    from ai_council.output import RoutingFailure

    primary = tmp_path / "primary"
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")

    failures: list[RoutingFailure] = []
    saved = save_research_to_file(
        sample_research_report, primary, return_dir=blocker, routing_failures=failures
    )

    assert len(saved) == 1 and saved[0].exists()  # canonical only
    assert [f.artifact for f in failures] == ["research report"]
    assert failures[0].destination == blocker


# ---------------------------------------------------------------------------
# #42: research filename must not double the `research-` mode token
# ---------------------------------------------------------------------------

def test_research_filename_no_double_prefix_when_query_begins_research(tmp_path):
    """#42: a query beginning "Research…" must not yield council-out-…-research-research-…"""
    report = MergedResearchReport(
        query="Research: sycophantic convergence and blind-vote integrity",
        results=[],
        merged_report="merged content",
        summary_2500="summary",
        total_cost_usd=0.01,
        total_duration_sec=5.0,
        total_sources=0,
        cache_key="k",
    )
    saved = save_research_to_file(report, tmp_path / "primary")
    name = saved[0].name
    assert "research-research-" not in name
    assert name.count("research") == 1  # only the single mode token
    assert name.startswith("council-out-")
    assert "-research-sycophantic" in name


def test_research_filename_preserves_researcher_word(tmp_path):
    """#42: only a leading "research" *token* is stripped — "researcher…" is preserved."""
    report = MergedResearchReport(
        query="Researcher salaries in 2026",
        results=[],
        merged_report="merged content",
        summary_2500="summary",
        total_cost_usd=0.01,
        total_duration_sec=5.0,
        total_sources=0,
        cache_key="k",
    )
    saved = save_research_to_file(report, tmp_path / "primary")
    assert "-research-researcher-salaries" in saved[0].name
