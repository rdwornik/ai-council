#!/usr/bin/env python
"""validate_sealed_keys.py -- refuse to let a sealed key enter the repo (#67).

Two sealed keys exist under `docs/audits/`, both protected ONLY by a `.gitignore` rule
(`.gitignore:61` and `:65`). The 2026-07-18 consolidation had a real near-miss -- a `git mv`
nearly broke the ignore rule and staged a key. This gate is **defence in depth**: it does not
replace the ignore rules, it catches the case where they have been broken or bypassed.

Match rule (operator ruling 2026-07-19, deliberately WIDER than #67's literal
`SEALED-KEY*.json`): any `.json` whose FILENAME contains both `SEALED` and `KEY`,
case-insensitive, in EITHER order, at any position. The two real keys are named
inconsistently --

  * `docs/audits/2026-07-18-cli4-parity/SEALED-KEY.json`
  * `docs/audits/2026-07-17-epi1-archaeology-KEY-SEALED.json`

-- so a literal `SEALED-KEY*.json` pattern would miss the second one. A guard that misses one
of the two actual keys is worse than no guard: it converts vigilance into false confidence.

Verified at authoring time: this pattern matches ZERO tracked files repo-wide
(`git ls-files` -> only `.claude/settings.json` and `.vscode/settings.json` are tracked `.json`),
so the gate cannot brick commits on an already-tracked file.

**Prospective-only**: inspects only ADDED paths (`git diff --cached --diff-filter=A
--no-renames`), so `--no-renames` forces a rename to surface as delete+ADD and a `git mv` that
drags a key into a tracked location is policed.

Override: `AICOUNCIL_SEALED_KEY_ALLOW` must name the EXACT repo-relative path(s) being
authorized, `;`-separated. A bare truthy value does NOT work -- the operator must state what
they are authorizing, so an override can never blanket-disarm the gate. `--no-verify` is NOT
the sanctioned bypass here: it disarms every other hook at the same time. When the override
fires it emits a loud stderr banner, because an env var leaves no trace in `git log` and the
terminal transcript is its only audit trail.

Fails CLOSED: any git error blocks the commit and is labelled a GUARD MALFUNCTION. Peer gates
(`validate_audit_casing.py`) fail open because they police a naming convention; this one
polices a secret, so the failure directions are opposite.

Read-only: reads the staged name-status and one env var; writes NOTHING.

Authority: BACKLOG #67; near-miss recorded in
`docs/audits/2026-07-19-night-consolidation-verification.md`; CLAUDE.md §9 consumer-local gates.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

# Filename (not full path) must contain SEALED and KEY in either order, and end `.json`.
# Case-insensitive so `sealed-key.json` cannot dodge it by lowercasing.
_SEALED_KEY_NAME = re.compile(r"^.*(?:sealed.*key|key.*sealed).*\.json$", re.IGNORECASE)

_ALLOW_ENV = "AICOUNCIL_SEALED_KEY_ALLOW"


def _basename(path: str) -> str:
    """Repo-relative path -> its final component, normalized to forward slashes."""
    return path.replace("\\", "/").strip("/").split("/")[-1]


def is_sealed_key(path: str) -> bool:
    """True if this path's FILENAME is sealed-key-shaped. Path position is irrelevant --
    a key is a key wherever it is staged."""
    return bool(_SEALED_KEY_NAME.match(_basename(path)))


def allowed_paths() -> set[str]:
    """Paths the operator has explicitly authorized via the override env var. Compared as
    exact repo-relative strings (slash-normalized); a bare truthy value authorizes nothing."""
    raw = os.environ.get(_ALLOW_ENV, "")
    return {p.replace("\\", "/").strip().strip("/") for p in raw.split(";") if p.strip()}


def check(added_paths: list[str], allowed: set[str]) -> tuple[list[str], list[str]]:
    """Split the staged adds into (blocked, overridden) sealed-key paths."""
    blocked: list[str] = []
    overridden: list[str] = []
    for p in added_paths:
        if not is_sealed_key(p):
            continue
        if p.replace("\\", "/").strip("/") in allowed:
            overridden.append(p)
        else:
            blocked.append(p)
    return blocked, overridden


def staged_added_paths() -> list[str]:
    """Paths staged with status A (added). `--no-renames` forces a rename to surface as
    delete+ADD, so a `git mv` of a key into a tracked path is caught. Raises on git error."""
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
        # Fail CLOSED -- opposite of the peer casing gate. A secret-protection guard that
        # silently passes on error is worse than no guard.
        print("validate_sealed_keys: GUARD MALFUNCTION -- this is NOT a policy violation.",
              file=sys.stderr)
        print(f"  Could not read the staged file list: {exc}", file=sys.stderr)
        print("  Blocking the commit because this gate protects a secret and cannot verify "
              "that no sealed key is staged.", file=sys.stderr)
        print("  Fix the git error and retry; do not bypass.", file=sys.stderr)
        return 1

    blocked, overridden = check(added, allowed_paths())

    if overridden:
        # An env var leaves NO trace in git log. This banner is the only audit trail.
        print("=" * 72, file=sys.stderr)
        print("validate_sealed_keys: SEALED-KEY GUARD DELIBERATELY BYPASSED", file=sys.stderr)
        for p in overridden:
            print(f"  ALLOWED: {p}", file=sys.stderr)
        print(f"  Authorized by the {_ALLOW_ENV} environment variable.", file=sys.stderr)
        print("  A sealed key is being committed on purpose. This bypass leaves NO record "
              "in git log -- this terminal transcript is its only audit trail.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

    if blocked:
        print("validate_sealed_keys: refused -- sealed key staged (#67):", file=sys.stderr)
        for p in blocked:
            print(f"  {p}", file=sys.stderr)
        print("  A sealed key must never enter the repo: it holds the blind-trial identity "
              "mapping, and committing it destroys the seal.", file=sys.stderr)
        print("  These files are meant to stay gitignored and untracked -- if one is staged, "
              "an ignore rule was broken or bypassed (e.g. by a `git mv`).", file=sys.stderr)
        print(f"  Unstage it (`git restore --staged <path>`) and check `.gitignore`. "
              f"If committing it is genuinely intended, authorize the exact path: "
              f"{_ALLOW_ENV}='<repo-relative-path>' (`;`-separated for several). "
              f"Do NOT use --no-verify -- it disarms every other hook too.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
