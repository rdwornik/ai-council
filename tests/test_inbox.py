"""Unit tests for src/inbox.py — no API calls."""

import textwrap
from pathlib import Path

import pytest

from src.inbox import archive_file, clean_slug, parse_file, scan_inbox


def test_parse_file_no_frontmatter(tmp_path: Path) -> None:
    """File without frontmatter returns full content and empty metadata."""
    f = tmp_path / "question.md"
    f.write_text("Should we use Redis or Memcached?", encoding="utf-8")
    content, metadata = parse_file(f)
    assert content == "Should we use Redis or Memcached?"
    assert metadata == {}


def test_parse_file_with_frontmatter(tmp_path: Path) -> None:
    """File with frontmatter returns metadata keys and body content."""
    f = tmp_path / "question.md"
    f.write_text(
        textwrap.dedent("""\
            ---
            models: claude,openai
            rounds: 1
            full: false
            ---
            REST or GraphQL for a public API?
        """),
        encoding="utf-8",
    )
    content, metadata = parse_file(f)
    assert content == "REST or GraphQL for a public API?"
    assert metadata["models"] == "claude,openai"
    assert metadata["rounds"] == 1
    assert metadata["full"] is False


def test_archive_file_success(tmp_path: Path) -> None:
    """archive_file() moves file to archive dir with timestamp prefix."""
    inbox = tmp_path / "inbox"
    archive = tmp_path / "archive"
    inbox.mkdir()
    archive.mkdir()

    src = inbox / "my-question.md"
    src.write_text("A question", encoding="utf-8")

    dest = archive_file(src, archive)

    assert not src.exists(), "Source should be moved"
    assert dest.exists(), "Destination should exist"
    assert dest.parent == archive
    # Timestamp prefix: YYYY-MM-DDTHHMM_my-question.md
    assert dest.name.endswith("_my-question.md")
    assert not dest.name.startswith("FAILED_")


def test_archive_file_failed(tmp_path: Path) -> None:
    """archive_file(failed=True) prefixes filename with FAILED_."""
    inbox = tmp_path / "inbox"
    archive = tmp_path / "archive"
    inbox.mkdir()
    archive.mkdir()

    src = inbox / "broken.md"
    src.write_text("Bad question", encoding="utf-8")

    dest = archive_file(src, archive, failed=True)

    assert not src.exists()
    assert dest.name.startswith("FAILED_")
    assert "broken.md" in dest.name


def test_scan_inbox_empty(tmp_path: Path) -> None:
    """scan_inbox() on an empty directory returns an empty list."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    result = scan_inbox(inbox)
    assert result == []


# ---------------------------------------------------------------------------
# clean_slug
# ---------------------------------------------------------------------------


def test_clean_slug_strips_failed_and_timestamp() -> None:
    stem = "FAILED_2026-03-26T1200_council_mode_system_design"
    assert clean_slug(stem) == "council_mode_system_design"


def test_clean_slug_strips_timestamp_only() -> None:
    stem = "2026-03-26T1200_council_mode_system_design"
    assert clean_slug(stem) == "council_mode_system_design"


def test_clean_slug_plain_filename() -> None:
    assert clean_slug("question") == "question"


def test_clean_slug_does_not_strip_failed_mid_filename() -> None:
    """'failure_analysis' should NOT be treated as a FAILED_ prefix."""
    assert clean_slug("failure_analysis") == "failure_analysis"


def test_clean_slug_failed_without_timestamp() -> None:
    """FAILED_ prefix alone (no timestamp) is stripped."""
    assert clean_slug("FAILED_my_question") == "my_question"


# ---------------------------------------------------------------------------
# scan_inbox — archive exclusion
# ---------------------------------------------------------------------------


def test_scan_inbox_excludes_archive_subdir(tmp_path: Path) -> None:
    """Files inside archive/ subdirectory are not returned by scan_inbox."""
    inbox = tmp_path / "inbox"
    archive = inbox / "archive"
    inbox.mkdir()
    archive.mkdir()

    (inbox / "valid_question.md").write_text("Q", encoding="utf-8")
    (archive / "FAILED_2026-03-26T1200_old_question.md").write_text("old", encoding="utf-8")

    result = scan_inbox(inbox)
    names = [p.name for p in result]
    assert names == ["valid_question.md"]
    assert not any("FAILED_" in n for n in names)
