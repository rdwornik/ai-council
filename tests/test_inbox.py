"""Unit tests for src/inbox.py — no API calls."""

import textwrap
from pathlib import Path

import pytest

from src.inbox import archive_file, parse_file, scan_inbox


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
