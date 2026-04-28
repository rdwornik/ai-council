"""Tests for dual output path behavior (primary + optional secondary)."""

import logging
from pathlib import Path

import pytest

from src.models import DebateResult, ModelResponse, Question, Round
from src.output import save_to_file
from src.research.models import MergedResearchReport, ResearchResult
from src.research.output import save_research_to_file


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
    with caplog.at_level(logging.WARNING, logger="src.output"):
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
    assert "_research.md" in saved[0].name


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
