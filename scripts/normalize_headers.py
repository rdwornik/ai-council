"""normalize_headers.py — dated-log entry header normalizer.

Deterministic, idempotent rewriter for dated-log entry headers in LESSONS.md
and JOURNAL.md (and any markdown file with ISO-date headings). Wired as an
auto-format pre-commit hook: it rewrites; it never fails-and-asks.

Rules:
  - `## YYYY-MM-DD`              -> `### YYYY-MM-DD`
  - `## YYYY-MM-DD — Topic`      -> `### YYYY-MM-DD — Topic`  (separator + topic preserved verbatim)
  - `### YYYY-MM-DD`             -> unchanged
  - `### YYYY-MM-DD — Topic`     -> unchanged
  - `### YYYY-MM-DD | a | b ...` -> unchanged   (LESSONS 6-field pipe schema)
  - `## YYYY-MM-DD | a | b ...`  -> unchanged   (non-canonical pipe form left alone)
  - Non-date headings, prose, bullets, fenced code blocks -> unchanged

Usage:
    python scripts/normalize_headers.py LESSONS.md JOURNAL.md
    python scripts/normalize_headers.py --check LESSONS.md   # exit 0 always; rewrites in place
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_DATE = r"\d{4}-\d{2}-\d{2}"

_DATE_ONLY = re.compile(rf"^(##|###)\s+({_DATE})\s*$")
_DATE_WITH_PIPE = re.compile(rf"^(##|###)\s+({_DATE})\s*\|")
_DATE_WITH_SEP = re.compile(rf"^(##|###)\s+({_DATE})(\s+[—–\-]\s+.+)$")

_FENCE = re.compile(r"^```")


def normalize_line(line: str) -> str:
    """Rewrite a single line per the rules. Lines outside fenced blocks only."""
    # LESSONS pipe schema — leave alone regardless of header level
    if _DATE_WITH_PIPE.match(line):
        return line
    m_only = _DATE_ONLY.match(line)
    if m_only:
        return f"### {m_only.group(2)}"
    m_sep = _DATE_WITH_SEP.match(line)
    if m_sep:
        # Preserve separator + topic verbatim; only the heading level changes
        return f"### {m_sep.group(2)}{m_sep.group(3)}"
    return line


def normalize_text(text: str) -> str:
    """Normalize all dated-log entry headers in a markdown document.

    Lines inside fenced code blocks are passed through verbatim.
    """
    out: list[str] = []
    in_fence = False
    # Preserve original line endings by splitting on '\n' and rejoining.
    # splitlines(True) keeps the trailing newline characters per line.
    for raw in text.splitlines(keepends=True):
        # Strip the trailing newline for matching, then re-attach
        if raw.endswith("\r\n"):
            body, eol = raw[:-2], "\r\n"
        elif raw.endswith("\n"):
            body, eol = raw[:-1], "\n"
        elif raw.endswith("\r"):
            body, eol = raw[:-1], "\r"
        else:
            body, eol = raw, ""
        if _FENCE.match(body):
            in_fence = not in_fence
            out.append(body + eol)
            continue
        if in_fence:
            out.append(body + eol)
            continue
        out.append(normalize_line(body) + eol)
    return "".join(out)


def normalize_file(path: Path) -> bool:
    """Normalize one file in place. Returns True iff content changed."""
    original = path.read_text(encoding="utf-8")
    rewritten = normalize_text(original)
    if rewritten == original:
        return False
    path.write_text(rewritten, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("usage: normalize_headers.py FILE [FILE ...]\n")
        return 0  # auto-format hooks should not fail
    for arg in argv:
        p = Path(arg)
        if not p.exists() or p.is_dir():
            continue
        normalize_file(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
