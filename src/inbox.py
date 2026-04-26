"""Inbox folder scanning, frontmatter parsing, and archive logic."""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import frontmatter

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
    """Return .md files in downloads_dir that contain council frontmatter keys.

    Detection: any frontmatter key (case-insensitive) must match an entry in council_keys.
    Files without frontmatter or with no matching keys are silently skipped.
    Malformed YAML frontmatter is logged as a warning and skipped.
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


def parse_file(file_path: Path) -> tuple[str, dict]:
    """Parse a markdown file with optional YAML frontmatter.

    Returns:
        (content, metadata) where content is the body text and metadata
        is a dict with keys: models (str), rounds (int), full (bool).
        If no frontmatter, metadata is {}.
    """
    post = frontmatter.load(str(file_path))
    content = post.content.strip()
    metadata = dict(post.metadata)
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
