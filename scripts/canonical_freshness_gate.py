"""Canonical living-file freshness gate — the portable (Group A) enforcement organ.

Single-sourced from what used to live inline in `scripts/audit.py::check_canonical_freshness`,
so ONE module serves both:
  - the hub **audit leg** — `audit.py` imports `evaluate` + the helpers and wraps the result in
    its `Finding` envelope (leg behaviour unchanged); and
  - a **consumer-local pre-commit gate** — this module's `__main__` runs against the consumer
    root and exits 1 on a FAIL (blocks the commit), 0 otherwise.

Deployed into a consumer verbatim by the enforcement-mesh carrier (Fable-consult mesh, #236);
its `canonical_freshness` name/path is exactly what the Informant's `locate` scans for.

A2 (FAIL): `last_reviewed` predates the file's last git-commit date (edited but not re-reviewed).
A1 (WARN): `last_reviewed` older than the calendar cadence (a loose nudge even when unchanged).
Missing `last_reviewed` -> WARN (child-repo-safe: adopt the convention without a hard failure).
A2 is COMMIT-based (not working-tree): an uncommitted edit is not flagged until it lands, so a
mid-edit tree does not FAIL before the reviewer bumps the stamp. Read-only; degrades gracefully
without git (A2 skipped). JOURNAL/LESSONS + the per-session BACKLOG are deliberately EXCLUDED
from the default set (their freshness is intrinsic to how they are written).
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_FRESHNESS_FILES = ["VISION.md", "ARCHITECTURE.md", "CLAUDE.md", "CONTRIBUTING.md",
                           "docs/handoffs/README.md", "protocols/ESSENTIALS.md"]
# Calendar-age backstop (A1): WARN — not FAIL — past this many days even if unchanged. The
# load-bearing signal is A2 (edited-since-review), which is the FAIL.
FRESHNESS_CADENCE_DAYS = 30


def parse_last_reviewed(text: str) -> Optional[date]:
    """Extract `last_reviewed` from a file's YAML frontmatter, or None if absent.

    Returns None when the file has no frontmatter, the frontmatter is unclosed or not a
    mapping, the key is missing, or its value is not a parseable ISO date. YAML parses an
    unquoted ISO date to a date (or datetime); quoted/string forms are parsed explicitly.
    """
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict):
        return None
    val = fm.get("last_reviewed")
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return date.fromisoformat(val.strip())
        except ValueError:
            return None
    return None


def git_last_commit_date(repo_path: Path, filename: str) -> Optional[date]:
    """Author date (short ISO) of the most recent commit touching `filename`.

    Uses author date (`%as`), not committer date: author date survives rebase / cherry-pick /
    amend, so A2 keys off when the content was actually edited, not when history was rewritten.
    Read-only. Returns None when git is absent, the path is not a git repo, or the file has no
    history — callers then skip A2 and fall back to the A1 calendar backstop (non-git-safe).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log", "-1", "--format=%as", "--", filename],
            capture_output=True, text=True, encoding="utf-8",
        )
    except OSError:
        return None
    out = result.stdout.strip()
    if result.returncode != 0 or not out:
        return None
    try:
        return date.fromisoformat(out)
    except ValueError:
        return None


def evaluate(repo_path: Path, freshness_files: Optional[list[str]] = None, *,
             parse_fn=parse_last_reviewed, git_date_fn=git_last_commit_date,
             today: Optional[date] = None) -> tuple[list[str], list[str]]:
    """Pure freshness evaluation -> (fails, warns) of per-file detail strings.

    `parse_fn` / `git_date_fn` are injectable so the audit leg keeps its monkeypatch seam
    (tests set `audit._git_last_commit_date`); `today` defaults to `date.today()`.
    """
    files = DEFAULT_FRESHNESS_FILES if freshness_files is None else freshness_files
    today = today or date.today()
    fails: list[str] = []
    warns: list[str] = []
    for fname in files:
        fpath = repo_path / fname
        if not fpath.exists():
            continue  # presence enforced elsewhere — don't double-report
        reviewed = parse_fn(fpath.read_text(encoding="utf-8"))
        if reviewed is None:
            warns.append(f"{fname}: no parseable last_reviewed frontmatter")
            continue
        git_date = git_date_fn(repo_path, fname)
        if git_date is not None and reviewed < git_date:
            fails.append(
                f"{fname}: last_reviewed {reviewed.isoformat()} predates last edit "
                f"{git_date.isoformat()} - edited but not re-reviewed")
            continue  # A2 dominates; don't also calendar-warn a file already failing
        age = (today - reviewed).days
        if age > FRESHNESS_CADENCE_DAYS:
            warns.append(
                f"{fname}: last_reviewed {reviewed.isoformat()} is {age}d old "
                f"(> {FRESHNESS_CADENCE_DAYS}d cadence)")
    return fails, warns


def _resolve_repo_root() -> Path:
    """Consumer-local root: git-toplevel (from cwd) -> $CLAUDE_PROJECT_DIR -> the deployed copy's
    location. Same git-toplevel-first rationale as session_end_backpressure (#237): a deployed
    copy audits the repo it RUNS in, and a clone-based harness (cwd=<clone>) resolves correctly.
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except Exception:
        pass
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def main() -> int:
    """Consumer-local pre-commit gate: A2 FAIL -> exit 1 (blocks the commit); WARN -> print,
    exit 0. Read-only. Resolution of a real A2 is a GENUINE re-read + honest stamp bump — never
    a date faked to green the gate.
    """
    root = _resolve_repo_root()
    # Resolve the present-file set ONCE and hand it to evaluate(), so the count printed below is
    # literally the list that was evaluated rather than a second implementation of the same
    # existence predicate (evaluate() skips absent files, so this is behaviour-identical). A
    # recomputed count is the duplicate-predicate shape LESSONS 2026-07-26 warns about.
    present = [f for f in DEFAULT_FRESHNESS_FILES if (root / f).exists()]
    fails, warns = evaluate(root, present)
    for w in warns:
        print(f"canonical_freshness WARN: {w}")
    for f in fails:
        print(f"canonical_freshness FAIL: {f}")
    if fails:
        print(f"canonical_freshness: {len(fails)} canonical doc(s) stale (edited since review) — "
              "re-read end-to-end and bump last_reviewed to the GENUINE review date "
              "(never a fake stamp). Bypass in good faith with --no-verify if wrong.")
        return 1
    # #126 output contract (local convention, ruled 2026-07-27): success is a POSITIVE
    # assertion — name, verdict, predicate, item count — never exit-0 silence; a zero-item
    # run must be distinguishable from a clean one. Superseded by the hub fleet-intake
    # (2026-07-26, commissions A–J) gate-output ruling when it lands.
    print(f"canonical_freshness: OK ({len(present)} canonical doc(s) checked -- last_reviewed not "
          f"predating each doc's last commit (A2), {FRESHNESS_CADENCE_DAYS}d cadence (A1, "
          f"warn-only); {len(warns)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
