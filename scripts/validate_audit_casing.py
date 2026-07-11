#!/usr/bin/env python
"""validate_audit_casing.py -- ADR-101 R4 casing gate (fleet ruling d1; CASING-ONLY carry).

A scoped **extract** of the hub gate `.dev-knowledge/scripts/validate_hermetization.py`
(#306): this repo carries ONLY the ADR-101 R4 casing branch of hub Rule B. Two hub-local
branches are deliberately NOT carried (recorded machine-readably in the repo-root
`.methodology.yaml` `sanctioned_divergences`):

  * hub Rule A (top-level tree seal) -- keyed to the HUB tree (its SANCTIONED_TIER1_DIRS
    omit `src/`), so carrying it verbatim would BLOCK every new file added under
    `src/ai_council/`.
  * hub Rule B grammar (the leading `<YYYY-MM-DD>-` date-shape + the CLOSED 11-class enum) --
    ai-council audits use free-form leading tokens (`code-quality`, `fable-*`, `council-*`),
    already casing-conformant but not enum-conformant. The enum and date-shape grammar remain
    hub-local per fleet ruling d1; any future carry of the full gate requires a new operator
    ruling.

`language: system` local pre-commit hook, **prospective-only**: it inspects only ADDED paths
(`git diff --cached --diff-filter=A --no-renames`), so every existing file -- and the whole
`docs/audits/archive/legacy/` quarantine (5 path parts, structurally out of Rule B's 3-part
scope) -- is grandfathered and never checked. Name-SHAPE only, never date-accuracy.

Read-only: reads the staged name-status; writes NOTHING. Fail-OPEN but LOUD on any git error
(a convention/hygiene gate must not brick every commit on a near-impossible git failure).
Bypass parity with peer hooks: `git commit --no-verify`.

Authority: hub ADR-101 R4; fleet ruling d1; ai-council ADR-34 filename lineage.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional

# R4 casing: all-lowercase kebab-case + digits; `.` carve-out (repo/version tokens like
# `.dev-knowledge` / `v3.4`). No UPPERCASE, no underscore, no other charset. Applied to the
# FULL filename (incl. the `.md` extension) so an uppercase `.MD` extension is caught too.
_LOWER_KEBAB_DOT = re.compile(r"^[a-z0-9.-]+$")


def _posix_parts(path: str) -> list[str]:
    """Repo-relative path -> its components, normalized to forward slashes."""
    return path.replace("\\", "/").strip("/").split("/")


def audit_casing_violation(path: str) -> Optional[str]:
    """R4 casing check. Applies ONLY to an added `docs/audits/*.md` (3 path parts exactly, so
    `docs/audits/archive/legacy/*` is out of scope). Return a BLOCK reason, or None."""
    parts = _posix_parts(path)
    # Applicability is EXTENSION-CASE-INSENSITIVE so an uppercase `.MD` cannot dodge the check
    # by escaping the `.md` match; the casing rule below then rejects the uppercase extension.
    if not (len(parts) == 3 and parts[0] == "docs" and parts[1] == "audits"
            and parts[2].lower().endswith(".md")):
        return None  # not an audit file -> silent
    fname = parts[2]
    if fname.lower() == "readme.md":
        return None  # the generated index, not an audit artifact
    if not _LOWER_KEBAB_DOT.match(fname):
        return (f"casing: '{fname}' must be all-lowercase kebab-case everywhere incl. the "
                f".md extension (no UPPERCASE, no _underscore_, no CamelCase; only a `.` "
                f"inside the slug for a repo/version token) -- ADR-101 R4")
    return None


def check(added_paths: list[str]) -> list[str]:
    """Every added path -> the list of `path: reason` BLOCK strings (empty == clean)."""
    reasons: list[str] = []
    for p in added_paths:
        r = audit_casing_violation(p)
        if r is not None:
            reasons.append(f"{p}: {r}")
    return reasons


def staged_added_paths() -> list[str]:
    """Paths staged with status A (added). Prospective-only: MODIFIED existing files are
    grandfathered. `--no-renames` forces a rename to surface as delete+ADD so a rename that
    introduces a NEW off-casing pathname is policed too. Raises on git error."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--diff-filter=A", "--no-renames", "--name-only"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"git exited {out.returncode}")
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def main() -> int:
    try:
        added = staged_added_paths()
    except (OSError, RuntimeError) as exc:
        # Fail OPEN but LOUD: a casing convention gate, not a safety control.
        print(f"validate_audit_casing: WARNING -- could not read staged adds ({exc}); "
              f"casing check skipped", file=sys.stderr)
        return 0
    reasons = check(added)
    if reasons:
        print("validate_audit_casing: refused -- ADR-101 R4 casing violation(s):",
              file=sys.stderr)
        for r in reasons:
            print(f"  {r}", file=sys.stderr)
        print("  docs/audits/*.md names are all-lowercase kebab-case (ADR-101 R4). "
              "Bypass (peer-hook parity): git commit --no-verify.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
