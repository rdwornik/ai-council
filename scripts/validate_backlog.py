#!/usr/bin/env python
"""Read-only narrow schema validator for BACKLOG.md (ADR-66 story-map hierarchy).

Layer-2 invariant: reads BACKLOG.md only; never writes, never orchestrates.

Structure (ADR-66):
    # .dev-knowledge BACKLOG
    ## Big picture            <- paragraph + theme backbone (no stories/tasks)
    ## <Theme>                <- backbone header
    ### <User story>          <- human goal
    So that <why>.            <- the why (required, immediately under the story)
    - [#id] [P1][M] <action> · Done when: <criterion> · refs <…>   <- task bullet

Hard-fail (exit 1) — objective structure only:
  - a task with no enclosing user story, or whose story has no enclosing theme
  - a task missing its [P][S|M|L] band, or missing "Done when:"
  - a missing or duplicate [#id]
  - a done task present — done tasks leave (ADR-65): a `status:done` suffix, a leading
    `[x]` checkbox, a struck bullet, an in-place `~~strikethrough~~`, or a bold
    `**RESOLVED`/`**DONE` marker on a task line
  - a user story with no "So that" line, or a story/task directly under ## Big picture
  - (#156 task-graph) a `· depends-on: #id` referencing an id that is not a live task
    (strict reference-existence — closed ids have left the file), or a cycle in the
    depends-on graph (direct A↔B, indirect A→B→C→A, or self A→A; the path is reported)

Warn-only: a user story with zero tasks.

(No repo: rule — entries are implicitly .dev-knowledge; cross-repo work names repos in
task text under the Cross-repo theme. Monotonic/never-reused id is an assignment
discipline; only uniqueness is enforced statically.)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BIG_PICTURE = "Big picture"
BACKLOG = Path(__file__).resolve().parent.parent / "BACKLOG.md"

_THEME_RE = re.compile(r"^## (.+?)\s*$")
_STORY_RE = re.compile(r"^### (.+?)\s*$")
_TASK_RE = re.compile(r"^- \[#(\d+)\]\s*(.*)$")
_SOTHAT_RE = re.compile(r"^So that\b", re.IGNORECASE)
_PSIZE_RE = re.compile(r"\[P[1-3]\]\[(?:S|M|L)\]")
_DONEWHEN_RE = re.compile(r"Done when:", re.IGNORECASE)
# done-marker: a structured status suffix, a leading done-checkbox, or a struck bullet —
# NOT a bare [x]/~~/"status: done" anywhere in prose (which is legitimate task text).
_DONE_MARKER_RE = re.compile(r"·\s*status:\s*done\b|^- \[[xX]\]|^- ~~")
# in-place resolution marker on a task line — the ADR-65 done-items-LEAVE violation
# class (#83, pilot finding F2). A live task must never be struck through (~~...~~)
# or carry a bold **RESOLVED/**DONE marker; done tasks LEAVE the file. _DONE_MARKER_RE
# above only catches a fully-struck bullet ("- ~~") or "- [x]" — it missed the
# "[#id] ~~...~~ **RESOLVED**" shape the #79 stub exhibited (commit 052e311).
_INPLACE_RESOLVED_RE = re.compile(r"~~.+?~~|\*\*\s*(?:RESOLVED|DONE)\b")
# === #156 task-graph machinery — CARRIER-DOCTRINE TWIN (ADR-78) ============================
# The regexes + _parse_deps + _check_dep_references + _check_dep_cycles below are a deliberate
# VERBATIM twin of the hub's scripts/validate_backlog.py. The floor is operator-generated /
# child-committed (ADR-78), not a shared module/symlink — so the two copies MUST be kept in
# sync BY HAND. The twin-drift edge is now PINNED by tests/test_validate_backlog_twin_parity.py
# (#206, GAP-2); de-dup into a single shared module remains the real fix. Do NOT add the parity
# check INSIDE this validator (Layer-2: validators stay logic-only) — it lives in tests/.
_DEPENDS_CLAUSE_RE = re.compile(r"·\s*depends-on\s*:\s*([^·]*)")
_DEPID_RE = re.compile(r"#(\d+)")


def _parse_deps(rest):
    """Return the depends-on ids as BARE strings (e.g. ['23', '45']) — matches task['id']
    form so membership tests are not silently always-false. Only the depends-on clause is
    read; ids in refs/prose are ignored."""
    m = _DEPENDS_CLAUSE_RE.search(rest)
    return _DEPID_RE.findall(m.group(1)) if m else []


def _check_dep_references(tasks):
    """Strict reference-existence: every depends-on id must be a live task id (#156)."""
    ids = {t["id"] for t in tasks}
    hard = []
    for t in tasks:
        loc = f'[#{t["id"]}] line {t["line"]}'
        for d in _parse_deps(t["rest"]):
            if d not in ids:
                hard.append(f'depends-on references non-existent id #{d} — {loc}')
    return hard


def _check_dep_cycles(tasks):
    """No-cycle: the depends-on graph must be acyclic — catches direct (A↔B), indirect
    (A→B→C→A), and self (A→A) cycles via a white/gray/black DFS (#156). Dangling ids are
    skipped here (owned by _check_dep_references) so this never KeyErrors."""
    ids = {t["id"] for t in tasks}
    line_of = {}
    adj = {}
    for t in tasks:
        adj.setdefault(t["id"], [])
        line_of.setdefault(t["id"], t["line"])
        for d in _parse_deps(t["rest"]):
            if d in ids:  # skip dangling — reference check reports those
                adj[t["id"]].append(d)

    hard = []
    color = dict.fromkeys(adj, 0)  # 0=white, 1=gray (on stack), 2=black (done)
    stack = []
    seen = set()

    def dfs(node):
        color[node] = 1
        stack.append(node)
        for nb in adj[node]:
            if color[nb] == 1:  # back-edge -> the stack slice [nb..node] is a cycle
                cycle = tuple(stack[stack.index(nb):])
                pivot = cycle.index(min(cycle, key=int))  # rotate to min id -> dedup rotations
                canon = cycle[pivot:] + cycle[:pivot]
                if canon not in seen:
                    seen.add(canon)
                    if len(canon) == 1:
                        hard.append(f'task #{canon[0]} depends on itself — '
                                    f'[#{canon[0]}] line {line_of[canon[0]]}')
                    else:
                        # ASCII arrow — '→' (U+2192) is not in cp1252 and raises
                        # UnicodeEncodeError on a Windows console, crashing the gate
                        # exactly on the cycle path that must print a clear message.
                        path = " -> ".join(f"#{n}" for n in canon) + f" -> #{canon[0]}"
                        hard.append(f'dependency cycle: {path}')
            elif color[nb] == 0:
                dfs(nb)
        stack.pop()
        color[node] = 2

    for node in sorted(adj, key=int):
        if color[node] == 0:
            dfs(node)
    return hard


def parse(text):
    """Return (themes, stories, tasks)."""
    themes, stories, tasks = [], [], []
    cur_theme = None
    cur_story = None
    expect_sothat = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        t = _THEME_RE.match(raw)
        if t:
            cur_theme = t.group(1).strip()
            cur_story = None
            expect_sothat = False
            themes.append(cur_theme)
            continue
        s = _STORY_RE.match(raw)
        if s:
            cur_story = {"name": s.group(1).strip(), "theme": cur_theme, "line": lineno,
                         "sothat": False, "ntasks": 0}
            stories.append(cur_story)
            expect_sothat = True
            continue
        k = _TASK_RE.match(raw)
        if k:
            tasks.append({"id": k.group(1), "rest": k.group(2), "raw": raw, "line": lineno,
                          "story": cur_story, "theme": cur_theme})
            if cur_story:
                cur_story["ntasks"] += 1
            expect_sothat = False
            continue
        if expect_sothat and _SOTHAT_RE.match(raw.strip()):
            if cur_story:
                cur_story["sothat"] = True
            expect_sothat = False
            continue
        if raw.strip():
            expect_sothat = False
    return themes, stories, tasks


def validate(themes, stories, tasks):
    """Return (hard_fails, warnings)."""
    hard, warn = [], []
    big = themes.count(BIG_PICTURE)
    if big != 1:
        hard.append(f'expected exactly one "## {BIG_PICTURE}" section, found {big}')
    seen = {}
    for t in tasks:
        loc = f'[#{t["id"]}] line {t["line"]}'
        if t["story"] is None:
            hard.append(f'task not under a user story — {loc}')
        elif not t["story"]["theme"] or t["story"]["theme"] == BIG_PICTURE:
            hard.append(f'task\'s story has no enclosing theme — {loc}')
        if t["id"] in seen:
            hard.append(f'duplicate id {t["id"]}: lines {seen[t["id"]]} and {t["line"]}')
        else:
            seen[t["id"]] = t["line"]
        if not _PSIZE_RE.search(t["rest"]):
            hard.append(f'task missing [P][S|M|L] band — {loc}')
        if not _DONEWHEN_RE.search(t["rest"]):
            hard.append(f'task missing "Done when:" — {loc}')
        if _DONE_MARKER_RE.search(t["raw"]):
            hard.append(f'done task present (done tasks leave the file, ADR-65) — {loc}')
        if _INPLACE_RESOLVED_RE.search(t["raw"]):
            hard.append(f'in-place resolved/struck-through task (done tasks leave the file, ADR-65) — {loc}')
    for s in stories:
        sloc = f'story "{s["name"][:48]}" line {s["line"]}'
        if not s["theme"] or s["theme"] == BIG_PICTURE:
            hard.append(f'user story not under a theme — {sloc}')
        if not s["sothat"]:
            hard.append(f'user story missing a "So that" line — {sloc}')
        if s["ntasks"] == 0:
            warn.append(f'user story with no tasks — {sloc}')
        # NOTE: the hub's governance-backlog-story-id ([S<n>]) rule is deliberately NOT
        # carried here — child story-id adoption is a Wave-1 rollout (#281/#286), so the
        # floor twin must keep classifying not-yet-migrated child backlogs as conformant.
    # #156 task-graph checks (CARRIER-DOCTRINE TWIN of the hub) — run independently so a
    # reference failure doesn't mask a real cycle among the valid edges.
    hard += _check_dep_references(tasks)
    hard += _check_dep_cycles(tasks)
    return hard, warn


def main():
    if not BACKLOG.exists():
        print(f"validate_backlog: {BACKLOG} not found", file=sys.stderr)
        return 1
    themes, stories, tasks = parse(BACKLOG.read_text(encoding="utf-8"))
    hard, warn = validate(themes, stories, tasks)
    for w in warn:
        print(f"WARN  {w}")
    for h in hard:
        print(f"FAIL  {h}")
    if hard:
        print(f"validate_backlog: {len(hard)} hard-fail(s), {len(warn)} warning(s)")
        return 1
    n_themes = len([t for t in themes if t != BIG_PICTURE])
    print(f"validate_backlog: OK ({n_themes} themes, {len(stories)} stories, {len(tasks)} tasks, "
          f"{len(warn)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
