"""Inbox folder scanning, frontmatter parsing, and archive logic."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import frontmatter

if TYPE_CHECKING:
    from ai_council.routing import TargetResolver

logger = logging.getLogger(__name__)

# Matches optional FAILED_ prefix and optional YYYY-MM-DDTHHMM_ timestamp prefix.
# Group 1 captures everything after those prefixes.
_SLUG_PREFIX_RE = re.compile(r"^(?:FAILED_)?(?:\d{4}-\d{2}-\d{2}T\d{4}_)?(.+)$")

# Matches a leading "council" token followed by a separator and remaining content.
# Case-insensitive; only fires when real content follows so a bare "council" stem survives.
_LEADING_COUNCIL_RE = re.compile(r"^council[-_ ]+(.+)$", re.IGNORECASE)


def clean_slug(filename_stem: str) -> str:
    """Strip FAILED_/timestamp prefixes AND a leading "council" token from a filename stem.

    Only strips FAILED_ when it is a leading prefix (not mid-filename).
    Only strips a timestamp when it directly follows the FAILED_ prefix or starts the name.
    A leading "council" token is stripped last (after the prefixes) so the emitted
    `council-out-...-<slug>.md` filename never carries "council" twice (#14). The strip
    fires only when a separator + content follow, so "council" alone and words like
    "councillor" are preserved.

    Examples:
        FAILED_2026-03-26T1200_council_mode_design -> mode_design
        2026-03-26T1200_council-question-foo       -> question-foo
        council_mode_system_design                 -> mode_system_design
        question                                   -> question
        council                                    -> council  (no trailing content)
        councillor_notes                           -> councillor_notes  (not a token)
        failure_analysis                           -> failure_analysis  (not stripped)
    """
    m = _SLUG_PREFIX_RE.match(filename_stem)
    slug = m.group(1) if m else filename_stem
    council_m = _LEADING_COUNCIL_RE.match(slug)
    return council_m.group(1) if council_m else slug


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
        # YAML frontmatter is untyped, so this value is `object` to a type checker. The cast
        # asserts nothing about the shape: `TargetResolver.resolve` is itself the validator --
        # its else-branch raises RoutingError naming the type it received, which is exactly the
        # fail-loud path a malformed `target-project` should take. Casting here keeps the
        # resolver's signature honest for the CLI call site, which really does pass a tuple.
        metadata["target_paths"] = resolver.resolve(
            cast("str | list[str] | tuple[str, ...] | None", raw_target))

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
