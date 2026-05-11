"""Unit tests for src/inbox.py — no API calls."""

import textwrap
from pathlib import Path

import pytest

from ai_council.inbox import archive_file, clean_slug, parse_file, scan_downloads_folder, scan_inbox
from ai_council.routing import RoutingError, TargetResolver


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


# ---------------------------------------------------------------------------
# scan_downloads_folder
# ---------------------------------------------------------------------------

_COUNCIL_KEYS = ["mode", "rounds", "models", "synthesizer", "full"]

_FM_TEMPLATE = "---\n{}\n---\nIs REST or GraphQL better?\n"


def _write_md(path: Path, name: str, frontmatter: str) -> Path:
    f = path / name
    f.write_text(_FM_TEMPLATE.format(frontmatter), encoding="utf-8")
    return f


def test_scan_downloads_detects_mode_key(tmp_path: Path) -> None:
    """Frontmatter with 'mode: pick' is detected as a council question."""
    _write_md(tmp_path, "question.md", "mode: pick")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1
    assert result[0].name == "question.md"


def test_scan_downloads_detects_mixed_case_key(tmp_path: Path) -> None:
    """Frontmatter key 'Mode: pick' (mixed case) is normalized and detected."""
    _write_md(tmp_path, "question.md", "Mode: pick")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1


def test_scan_downloads_detects_uppercase_key(tmp_path: Path) -> None:
    """Frontmatter key 'ROUNDS: 2' (uppercase) is normalized and detected."""
    _write_md(tmp_path, "question.md", "ROUNDS: 2")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1


def test_scan_downloads_single_key_enough(tmp_path: Path) -> None:
    """A single matching key (synthesizer) is sufficient for detection."""
    _write_md(tmp_path, "question.md", "synthesizer: gemini")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1


def test_scan_downloads_no_frontmatter_skipped(tmp_path: Path) -> None:
    """Plain .md file without frontmatter is silently skipped."""
    f = tmp_path / "plain.md"
    f.write_text("No frontmatter here.", encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert result == []


def test_scan_downloads_noncouncil_keys_skipped(tmp_path: Path) -> None:
    """File with frontmatter but no council keys (e.g. 'title:') is skipped."""
    _write_md(tmp_path, "notes.md", "title: My Notes\nauthor: Rob")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert result == []


def test_scan_downloads_malformed_yaml_skipped(tmp_path: Path) -> None:
    """File with malformed YAML frontmatter is skipped without crashing."""
    f = tmp_path / "broken.md"
    f.write_text("---\n: bad: yaml: [\n---\nContent\n", encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert result == []


def test_scan_downloads_missing_dir_returns_empty(tmp_path: Path) -> None:
    """Non-existent downloads dir returns empty list without error."""
    missing = tmp_path / "does_not_exist"
    result = scan_downloads_folder(missing, _COUNCIL_KEYS)
    assert result == []


def test_scan_downloads_case_insensitive_extension(tmp_path: Path) -> None:
    """Files with .MD extension are also scanned."""
    f = tmp_path / "question.MD"
    f.write_text(_FM_TEMPLATE.format("mode: pick"), encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1
    assert result[0].name == "question.MD"


# ---------------------------------------------------------------------------
# Filename-based detection
# ---------------------------------------------------------------------------


def test_scan_downloads_council_filename_no_frontmatter(tmp_path: Path) -> None:
    """File named council_xyz.md with no frontmatter is detected by filename."""
    f = tmp_path / "council_prompt_adr33.md"
    f.write_text("Should we use REST or GraphQL?", encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1
    assert result[0].name == "council_prompt_adr33.md"


def test_scan_downloads_council_filename_uppercase(tmp_path: Path) -> None:
    """File named COUNCIL_ABC.MD (uppercase stem) is detected case-insensitively."""
    f = tmp_path / "COUNCIL_ABC.md"
    f.write_text("Some question.", encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1


def test_scan_downloads_council_anywhere_in_stem(tmp_path: Path) -> None:
    """'council' anywhere in the filename stem triggers detection."""
    f = tmp_path / "my_council_question.md"
    f.write_text("Some question.", encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1


def test_scan_downloads_non_council_no_frontmatter_skipped(tmp_path: Path) -> None:
    """Plain file without 'council' in name and no frontmatter is skipped."""
    f = tmp_path / "random.md"
    f.write_text("Just a random note.", encoding="utf-8")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert result == []


def test_scan_downloads_frontmatter_wins_without_council_name(tmp_path: Path) -> None:
    """File named random.md but with council frontmatter is still detected."""
    _write_md(tmp_path, "random.md", "rounds: 2")
    result = scan_downloads_folder(tmp_path, _COUNCIL_KEYS)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_file target-project routing
# ---------------------------------------------------------------------------

_RESOLVER_PROJECTS = {
    ".dev-knowledge": "C:/Dev/.dev-knowledge",
    "foo": "C:/Dev/foo",
}
_TRANSCRIPTS = Path("docs") / "decisions" / "transcripts"


@pytest.fixture
def resolver() -> TargetResolver:
    return TargetResolver(_RESOLVER_PROJECTS)


def _write_fm(tmp_path: Path, name: str, frontmatter_body: str, content: str = "Question?") -> Path:
    f = tmp_path / name
    f.write_text(f"---\n{frontmatter_body}\n---\n{content}", encoding="utf-8")
    return f


def test_parse_file_no_target_project_no_resolver(tmp_path: Path) -> None:
    f = _write_fm(tmp_path, "q.md", "rounds: 1")
    _, meta = parse_file(f)
    assert meta.get("target_paths", []) == []


def test_parse_file_no_target_project_with_resolver(tmp_path: Path, resolver: TargetResolver) -> None:
    f = _write_fm(tmp_path, "q.md", "rounds: 1")
    _, meta = parse_file(f, resolver=resolver)
    assert meta["target_paths"] == []


def test_parse_file_target_project_single_string(tmp_path: Path, resolver: TargetResolver) -> None:
    f = _write_fm(tmp_path, "q.md", "target-project: .dev-knowledge")
    _, meta = parse_file(f, resolver=resolver)
    assert meta["target_paths"] == [Path("C:/Dev/.dev-knowledge") / _TRANSCRIPTS]


def test_parse_file_target_project_list(tmp_path: Path, resolver: TargetResolver) -> None:
    f = _write_fm(tmp_path, "q.md", "target-project:\n  - .dev-knowledge\n  - foo")
    _, meta = parse_file(f, resolver=resolver)
    assert len(meta["target_paths"]) == 2
    assert meta["target_paths"][0] == Path("C:/Dev/.dev-knowledge") / _TRANSCRIPTS
    assert meta["target_paths"][1] == Path("C:/Dev/foo") / _TRANSCRIPTS


def test_parse_file_unknown_target_raises_routing_error(tmp_path: Path, resolver: TargetResolver) -> None:
    f = _write_fm(tmp_path, "q.md", "target-project: unknown-project")
    with pytest.raises(RoutingError, match="Unknown target-project"):
        parse_file(f, resolver=resolver)


def test_parse_file_target_project_ignored_without_resolver(tmp_path: Path) -> None:
    f = _write_fm(tmp_path, "q.md", "target-project: .dev-knowledge")
    _, meta = parse_file(f)
    # Without resolver, target-project is in raw metadata but NOT resolved
    assert meta.get("target-project") == ".dev-knowledge"
    assert "target_paths" not in meta
