#!/usr/bin/env python
"""validate_docs_registry.py -- a new `docs/` directory must be registered (#68).

`docs/audits/README.md`: *"Registration is what makes a corpus legible. An unregistered folder
is indistinguishable from a leftover."* That rule is currently enforced by memory alone. #71
(`--no-persist` leaving an untracked scratch dir) is the live instance that motivated this.

**A registry check, NEVER a blanket directory ban.** A blanket ban would reject
`docs/audits/archive/` itself and both live corpora that legitimately sit in `docs/audits/`
today. A new directory under `docs/` is admissible if it is:

  * **grandfathered** -- already tracked in HEAD (prospective-only, like the peer casing gate);
  * an **`archive/`** child of a sanctioned taxonomy folder (invariant class b, read from the
    README's Directory-invariant table); or
  * directly under `docs/audits/` and **named in the README's registry** -- the Live-corpora
    table's Path column.

The registry is READ AT RUNTIME from `docs/audits/README.md` and is deliberately NOT restated
here: a copy in code is a second source of truth that rots silently. Registering a corpus by
adding its table row is what un-blocks it.

**Fails CLOSED, and says so.** If the README is missing, its sections renamed, or its tables
reformatted such that the parse yields nothing, this gate blocks the commit and labels the
result a GUARD MALFUNCTION -- explicitly distinguished from a policy violation. A registry
guard that fails open silently passes everything, which is the same false confidence that
made a narrow #67 pattern unacceptable. The two failure kinds must be distinguishable at a
glance, or the first false block trains everyone to bypass the gate.

The parse runs on EVERY invocation, not only when a new directory is staged, so a broken
registry surfaces immediately rather than lying dormant until the next corpus is added.

**Scope -- empty directories are explicitly OUT.** A pre-commit hook guards what ENTERS the
repo; git cannot see an empty directory, so an empty directory can never enter the repo. That
is not a limitation to work around, it is evidence that a commit gate is the wrong mechanism
for it. Working-tree hygiene belongs to the `check.ps1` / `verify_*` family, which runs
against a working tree deliberately. (Operator ruling 2026-07-19; recorded in #68's done-when.)

Read-only: reads the staged raw diff, HEAD's tree, and the INDEX copy of
`docs/audits/README.md`; writes NOTHING.

Authority: BACKLOG #68; `docs/audits/README.md` (directory invariant + Live corpora);
ADR-60 docs/ taxonomy; CLAUDE.md §5 item 9 (no leftovers).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REGISTRY = Path("docs/audits/README.md")
REGISTERED_PARENT = "docs/audits"

_INVARIANT_HEADING = re.compile(r"^#+\s+Directory invariant\b", re.IGNORECASE)
_CORPORA_HEADING = re.compile(r"^#+\s+Live corpora\b", re.IGNORECASE)
_HEADING = re.compile(r"^#+\s+")
# A backticked token that names a SINGLE-SEGMENT directory: ends with `/`, no interior slash.
# Catches `archive/` in the invariant table and `2026-07-18-cli4-parity/` in the Path column.
# Single-segment is a hardening requirement, not cosmetics: a multi-segment token such as
# `docs/` admitted as a taxonomy name would exempt every path below it (sol finding).
_BACKTICKED_DIR = re.compile(r"`([^`/]+/)`")

# Never admissible as a taxonomy name however the README is edited -- admitting these would
# disable the guard wholesale.
_RESERVED_TAXONOMY = frozenset({"docs", ".", "..", ""})

# git raw-format dst modes that are directory-shaped despite being a single staged path.
_GITLINK_MODE = "160000"
_SYMLINK_MODE = "120000"


class RegistryError(RuntimeError):
    """The registry could not be located or parsed -- a guard malfunction, not a violation."""


def _section_lines(text: str, heading: re.Pattern[str]) -> list[str]:
    """Lines belonging to the section introduced by `heading`, up to the next heading."""
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for ln in lines:
        if heading.match(ln):
            inside = True
            continue
        if inside and _HEADING.match(ln):
            break
        if inside:
            out.append(ln)
    return out


def _table_rows(lines: list[str]) -> list[list[str]]:
    """Markdown table rows -> their cells. Header and `|---|` delimiter rows are dropped."""
    rows: list[list[str]] = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} and c for c in cells):
            continue  # delimiter row
        rows.append(cells)
    return rows


def parse_registry(text: str) -> tuple[set[str], set[str]]:
    """`docs/audits/README.md` -> (registered corpus dir names, admissible taxonomy dir names).

    Raises RegistryError if either section is absent or yields nothing -- fail CLOSED.
    """
    inv_lines = _section_lines(text, _INVARIANT_HEADING)
    if not inv_lines:
        raise RegistryError(
            "no 'Directory invariant' section found (heading renamed, moved, or removed)")
    taxonomy: set[str] = set()
    for row in _table_rows(inv_lines):
        for cell in row:
            for m in _BACKTICKED_DIR.finditer(cell):
                name = m.group(1).strip("/")
                if name not in _RESERVED_TAXONOMY:
                    taxonomy.add(name)
    if not taxonomy:
        raise RegistryError(
            "the 'Directory invariant' table yielded no admissible directory names "
            "(table reformatted, or its backticked `<dir>/` entries changed shape)")

    corpora_lines = _section_lines(text, _CORPORA_HEADING)
    if not corpora_lines:
        raise RegistryError(
            "no 'Live corpora' section found (heading renamed, moved, or removed)")
    rows = _table_rows(corpora_lines)
    if not rows:
        raise RegistryError(
            "the 'Live corpora' section contains no table rows (registry table missing or "
            "reformatted)")
    # Row 0 is the header. Its presence is what proves the table is structurally intact, which
    # is what lets ZERO live corpora be represented as a valid state rather than a malfunction
    # -- once the last corpus exits to archive/ the table is legitimately empty, and #27's
    # unseal is exactly the event that produces it. Bricking every commit at that moment would
    # be a lifecycle bug, not a guard.
    header, data = rows[0], rows[1:]
    if not header or "path" not in header[0].strip().lower():
        raise RegistryError(
            "the 'Live corpora' table has no recognisable 'Path' header column (column order "
            "or formatting changed)")
    registered: set[str] = set()
    for row in data:
        if not row:
            continue
        m = _BACKTICKED_DIR.search(row[0])  # Path is the first column
        if m:
            registered.add(m.group(1).strip("/"))
    # Data rows that exist but none of which parse means the Path column changed shape --
    # distinct from a structurally valid table with zero data rows (an empty registry).
    if data and not registered:
        raise RegistryError(
            "the 'Live corpora' table has data rows but no parseable path in the first column "
            "(expected a backticked `<dir>/`; column order or formatting changed)")
    return registered, taxonomy


def load_registry() -> tuple[set[str], set[str]]:
    """Parse the registry AS STAGED, not as it sits in the working tree.

    Reading the working-tree copy is bypassable: `git rm --cached docs/audits/README.md`
    stages the registry's deletion while leaving an untracked copy on disk, so the guard would
    happily parse a file that the commit is removing -- and an attacker could add an admitting
    row to that untracked copy alone (sol finding). `git show :<path>` reads the index, which
    is exactly what the commit will contain.
    """
    out = subprocess.run(
        ["git", "show", f":{REGISTRY.as_posix()}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        raise RegistryError(
            f"could not read {REGISTRY.as_posix()} from the git index -- it may be staged for "
            f"deletion or replaced by a non-file entry ({out.stderr.strip()})")
    return parse_registry(out.stdout)


def _posix(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def new_dirs(added: list[tuple[str, str]], tracked_dirs: set[str]) -> list[str]:
    """Directories under `docs/` the staged adds would introduce, not already in HEAD.

    `added` is (mode, path). A gitlink (submodule) or symlink is a single staged path that is
    nonetheless directory-shaped -- `git submodule add <url> docs/rogue` stages exactly
    `docs/rogue` with no child component, so treating it as a file would let a fully populated
    directory enter unregistered (sol finding). Those entries are therefore treated as the
    directory itself.
    """
    found: set[str] = set()
    for mode, raw in added:
        parts = _posix(raw).split("/")
        if len(parts) < 2 or parts[0] != "docs":
            continue
        # A file contributes its parent dirs; a gitlink/symlink contributes ITSELF as a dir.
        depth = len(parts) + 1 if mode in (_GITLINK_MODE, _SYMLINK_MODE) else len(parts)
        for i in range(2, depth):
            d = "/".join(parts[:i])
            if d not in tracked_dirs:
                found.add(d)
    return sorted(found)


def violation(directory: str, registered: set[str], taxonomy: set[str]) -> str | None:
    """A BLOCK reason for this new directory, or None if it is admissible."""
    parts = directory.split("/")
    name = parts[-1]
    parent = "/".join(parts[:-1])
    # A taxonomy folder is admissible only at a SANCTIONED DEPTH: `docs/<taxonomy>` or
    # `docs/<section>/<taxonomy>` -- i.e. ADR-60's "each one's own archive/ child". Matching a
    # taxonomy name at arbitrary depth turns it into an escape hatch: `docs/x/y/archive/`
    # would launder anything beneath it (sol finding).
    if name in taxonomy and len(parts) in (2, 3):
        return None
    # Anything INSIDE an admissible taxonomy folder is governed by that folder's own README,
    # not by the Live-corpora table: invariant class b reads "`archive/` -- governed by its own
    # `archive/README.md`". Without this the guard would block the sanctioned EXIT path -- a
    # corpus moving to `docs/audits/archive/<corpus>/` at unseal, exactly what #27 must do
    # (proven: that move was rejected before this branch existed). The ancestor must itself sit
    # at a sanctioned depth, so the exemption cannot be conjured at arbitrary depth.
    for i in (2, 3):
        if len(parts) > i and parts[i - 1] in taxonomy:
            return None
    if parent == REGISTERED_PARENT and name in registered:
        return None  # a registered live corpus
    # A registered corpus owns its internal structure -- the registry registers the CORPUS, not
    # each folder inside it. cli4-parity already carries a `blinded/` child; that one passes
    # today only because it is grandfathered, so a corpus gaining new internal structure after
    # registration would otherwise be blocked (terra finding).
    if len(parts) > 3 and parts[0] == "docs" and parts[1] == "audits" and parts[2] in registered:
        return None
    if parent == REGISTERED_PARENT:
        return (f"'{directory}/' is a new directory in {REGISTERED_PARENT}/ with no row in the "
                f"'Live corpora' table of {REGISTRY.as_posix()}. An unregistered folder is "
                f"indistinguishable from a leftover.")
    return (f"'{directory}/' is a new directory under docs/ that is neither a sanctioned "
            f"taxonomy folder nor a registered live corpus (registries live in "
            f"{REGISTRY.as_posix()}).")


def staged_added_paths() -> list[tuple[str, str]]:
    """Staged additions as (dst-mode, path), read NUL-delimited.

    `--name-only` without `-z` C-quotes any non-ASCII path (`core.quotePath` defaults on), so
    `docs/audits/évasion/f.md` arrives as `"docs/audits/\\303\\251vasion/f.md"` -- the leading
    quote makes `parts[0] != "docs"` and the whole path is skipped, silently bypassing the
    guard entirely (sol finding, reproduced). `-z` emits raw bytes with no quoting. `--raw`
    additionally yields the dst mode, needed to spot gitlinks and symlinks.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--diff-filter=A", "--no-renames", "--raw", "-z"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        raise RegistryError(out.stderr.strip() or f"git diff exited {out.returncode}")
    # -z raw records: ":<srcmode> <dstmode> <srcsha> <dstsha> <status>\0<path>\0"
    tokens = [t for t in out.stdout.split("\0") if t != ""]
    entries: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        meta = tokens[i]
        if meta.startswith(":") and i + 1 < len(tokens):
            fields = meta[1:].split()
            entries.append((fields[1] if len(fields) > 1 else "", tokens[i + 1]))
            i += 2
        else:
            i += 1
    return entries


def tracked_dirs() -> set[str]:
    """Every directory tracked in HEAD -- the grandfathered set. Empty on an unborn HEAD."""
    out = subprocess.run(
        ["git", "ls-tree", "-d", "-r", "--name-only", "HEAD"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        if "Not a valid object name" in out.stderr or "unknown revision" in out.stderr:
            return set()  # unborn HEAD: the initial commit has nothing grandfathered
        raise RegistryError(out.stderr.strip() or f"git ls-tree exited {out.returncode}")
    return {_posix(ln) for ln in out.stdout.splitlines() if ln.strip()}


def _malfunction(exc: Exception) -> int:
    print("validate_docs_registry: GUARD MALFUNCTION -- this is NOT a policy violation.",
          file=sys.stderr)
    print(f"  Registry: {REGISTRY.as_posix()}", file=sys.stderr)
    print(f"  Problem:  {exc}", file=sys.stderr)
    print("  Nothing is wrong with what you staged. The guard cannot read its own registry, "
          "so it cannot tell a registered folder from a leftover.", file=sys.stderr)
    print("  Blocking rather than passing: a registry guard that fails open silently permits "
          "everything, which is worse than no guard.", file=sys.stderr)
    print(f"  Fix: restore the 'Directory invariant' and 'Live corpora' sections of "
          f"{REGISTRY.as_posix()} (a markdown table whose Path column holds a backticked "
          f"`<dir>/`), then retry.", file=sys.stderr)
    return 1


def main() -> int:
    # The registry is parsed on EVERY run, not only when a docs/ directory is staged, so a
    # broken registry surfaces immediately instead of lying dormant.
    try:
        registered, taxonomy = load_registry()
        added = staged_added_paths()
        tracked = tracked_dirs()
    except (RegistryError, OSError) as exc:
        return _malfunction(exc)

    reasons = [
        r for d in new_dirs(added, tracked)
        if (r := violation(d, registered, taxonomy)) is not None
    ]
    if reasons:
        print("validate_docs_registry: refused -- unregistered new docs/ directory (#68):",
              file=sys.stderr)
        for r in reasons:
            print(f"  {r}", file=sys.stderr)
        print(f"  To register a live corpus: add an essence markdown at the parent root AND a "
              f"row to the 'Live corpora' table in {REGISTRY.as_posix()} naming the path, what "
              f"it is, the ruling that keeps it there, its essence markdown, and its exit "
              f"condition. Otherwise the artifact belongs in an existing folder.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
