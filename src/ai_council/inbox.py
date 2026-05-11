"""Inbox folder scanning, frontmatter parsing, and archive logic."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

if TYPE_CHECKING:
    from ai_council.routing import TargetResolver

logger = logging.getLogger(__name__)

# Matches optional FAILED_ prefix and optional YYYY-MM-DDTHHMM_ timestamp prefix.
# Group 1 captures everything after those prefixes.
_SLUG_PREFIX_RE = re.compile(r"^(?:FAILED_)?(?:\d{4}-\d{2}-\d{2}T\d{4}_)?(.+)$")


def clean_slug(filename_stem: str) -> str:
    """Strip FAILED_ and timestamp prefixes from a filename stem.

    Only strips FAILED_ when it is a leading prefix (not mid-filename).
    Only strips a timestamp when it directly follows the FAILED_ prefix or starts the name.

    Examples:
        FAILED_2026-03-26T1200_question  -> question
        2026-03-26T1200_question         -> question
        question                         -> question
        failure_analysis                 -> failure_analysis  (not stripped)
    """
    m = _SLUG_PREFIX_RE.match(filename_stem)
    return m.group(1) if m else filename_stem


def ensure_dirs(inbox_dir: Path, archive_dir: Path) -> None:
    """Create inbox and archive directories if they don't exist."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)


def scan_inbox(inbox_dir: Path) -> list[Path]:
    """Return all .md files in inbox_dir, sorted by mtime ascending (oldest first)."""
    files = list(inbox_dir.glob("*.md"))
    return sorted(files, key=lambda p: p.stat().st_mtime)


def scan_downloads_folder(downloads_dir: Path, council_keys: list[str]) -> list[Path]:
    """Return .md files in downloads_dir that look like council files.

    Detection (either condition is sufficient):
    1. Filename stem contains "council" (case-insensitive).
    2. File has frontmatter with a key matching council_keys (case-insensitive).

    Malformed YAML frontmatter is logged as a warning and skipped only if the
    file doesn't qualify by filename.
    """
    if not downloads_dir.exists():
        logger.info("Downloads folder not found, skipping: %s", downloads_dir)
        return []

    # Case-insensitive extension match: *.md and *.MD etc.
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in ("*.md", "*.MD", "*.Md", "*.mD"):
        for p in downloads_dir.glob(pattern):
            if p not in seen:
                seen.add(p)
                candidates.append(p)

    council_keys_lower = {k.lower() for k in council_keys}
    detected: list[Path] = []

    for file_path in candidates:
        if "council" in file_path.stem.lower():
            detected.append(file_path)
            continue

        try:
            post = frontmatter.load(str(file_path))
        except Exception:
            logger.warning("Skipping malformed frontmatter: %s", file_path.name)
            continue

        if not post.metadata:
            continue

        file_keys = {k.lower() for k in post.metadata}
        if file_keys & council_keys_lower:
            detected.append(file_path)

    return sorted(detected, key=lambda p: p.stat().st_mtime)


def parse_file(
    file_path: Path,
    resolver: TargetResolver | None = None,
) -> tuple[str, dict]:
    """Parse a markdown file with optional YAML frontmatter.

    If resolver is provided, the 'target-project' frontmatter field (single string
    or list of strings) is resolved to a list[Path] stored as 'target_paths' in
    the returned metadata dict.  RoutingError is raised on unknown target names
    so the caller fails before debate logic runs.

    Returns:
        (content, metadata) where content is the body text and metadata
        is a dict with keys: models (str), rounds (int), full (bool), and
        optionally target_paths (list[Path]) when resolver is provided.
        If no frontmatter, metadata is {}.
    """
    post = frontmatter.load(str(file_path))
    content = post.content.strip()
    metadata = dict(post.metadata)

    if resolver is not None:
        raw_target = metadata.get("target-project")
        if isinstance(raw_target, str):
            raw_target = [raw_target]
        metadata["target_paths"] = resolver.resolve(raw_target)

    return content, metadata


def archive_file(file_path: Path, archive_dir: Path, *, failed: bool = False) -> Path:
    """Move file to archive_dir with a timestamp prefix.

    Args:
        file_path: Source file to archive.
        archive_dir: Destination directory.
        failed: If True, prefix filename with "FAILED_".

    Returns:
        Path to the archived file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M")
    prefix = "FAILED_" if failed else ""
    dest_name = f"{prefix}{timestamp}_{file_path.name}"
    dest = archive_dir / dest_name
    shutil.move(str(file_path), str(dest))
    return dest
