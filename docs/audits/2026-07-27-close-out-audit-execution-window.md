# Close-out audit — ai-council, 2026-07-27 execution window

**Auditor:** read-only pass over a session I did not run. Every claim below is re-derived from
disk and git; where the session's report and the repo disagree, the repo wins and it is called
out explicitly.

**Precondition (met).** `git fetch origin && git status -sb` → `## main...origin/main`;
`git rev-list --left-right --count origin/main...main` → `0	0`. Working tree clean.
HEAD = `17a239ba6f75a61c1f33ffbd0b0e9f8c0c3a89aa` (`17a239b`, 2026-07-27 17:55:08 +0200).

**Session range, derived not assumed.** `git rev-list --count 1dcee8b..HEAD` → **25**. That
matches the reported "25 commits". `1dcee8b` is the merge that closed the preceding
record-and-report turn and was `origin/main` when the execution window opened.

**No repo mutation by this audit.** Final `git status --porcelain=v1 -b` → `## main...origin/main`,
no entries. The only new filesystem artifacts are gitignored caches
(`.pytest_cache/`, `__pycache__/`) produced by the two required pytest runs.

---

## Section A — record coverage

### A1. JOURNAL.md — **UPDATED**

An entry exists: `JOURNAL.md:22` —
`### 2026-07-27 (2) — EXECUTION WINDOW: the ruling landed first, then three defect lanes (8caee48, fbf6711, 010f0b9, 422e4a3, d12b4d1)`.
(A second 2026-07-27 entry, `JOURNAL.md:42`, belongs to the earlier record turn.)

**Shas named in the entry (lines 22–41), all five verified:**

| sha | `git cat-file -t` | ancestor of main | subject |
|---|---|---|---|
| `8caee48` | commit | yes | Merge docs/0727-task0-117-ruling |
| `fbf6711` | commit | yes | Merge fix/126-validator-output-contract |
| `010f0b9` | commit | yes | Merge fix/125-rule4-four-surfaces |
| `422e4a3` | commit | yes | Merge fix/124-reproducible-toolchain |
| `d12b4d1` | commit | yes | Merge fix/123-repoint-gate-at-venv |

Command: `for s in ...; do git cat-file -t $s; git merge-base --is-ancestor $s main; done` — all
resolved, all ancestors. **ADR-85's anchor requirement is satisfied** (≥1 session sha named).

**Commits named in JOURNAL.md: 5 of 25. Not named: 20.** Of those 20:

- **18 are reachable from a named merge** — each work commit sits inside a merge the entry
  names, so it is anchored transitively (verified per-commit with `git merge-base --is-ancestor
  <commit> <named-merge>`; full mapping produced, e.g. `5af0a7a → 010f0b9`, `ac3fd62 → fbf6711`).
- **2 are anchored by NOTHING and cannot be:** `9b92c8c` (the session-close commit) and
  `17a239b` (its merge). Verified `NOT contained in ANY named merge`. This is the **structural
  one-commit tail** already recorded on `[#50]`'s W4 note: an entry cannot name the sha of the
  commit that contains it. It is the same debt the *previous* window handed to this one, and it
  is handed forward again.

**Finding A1-a.** The tail is now *two* commits, not one (`9b92c8c` + `17a239b`), because the
close is a commit plus a `--no-ff` merge. `[#50]`'s W4 text calls it a "ONE-COMMIT TAIL"; on the
evidence it is a one-commit-plus-its-merge tail. Minor, but the ticket's own wording understates
what the next session inherits.

### A2. LESSONS.md — **NOT UPDATED**

```
git log -1 --format='%h | %ci | %s' -- LESSONS.md
bffed5a | 2026-07-26 14:15:47 +0200 | docs: renumber arc, ADR-15, LESSONS class entry, #104 per-rule amendment

git log --oneline 1dcee8b..HEAD -- LESSONS.md   ->   0 commits
```

LESSONS.md was **not touched in this session**. Its newest entry (`LESSONS.md:11`) is dated
2026-07-26 and names precisely the class in question:

> `### 2026-07-26 | #97 checker arc + terra pre-merge review | a value published without the
> predicate that produces it cannot be checked | process | [scope: hybrid] | filed ADR-15,
> BACKLOG #118, amended #104 criterion (iii)`

**Finding A2-a (reported, not fixed).** This session produced **two fresh self-reported
instances of that same class**, and no LESSONS entry records them. Both are in the repo's own
commit messages:

- `6b8160e` — *"the try/finally form did NOT restore the caller's location on a failing gate —
  and the commit that introduced it asserted that it did."*
- `5757448` — *"The previous commit fixed the `exit` path and left the EXCEPTION path broken,
  then claimed 'every path'."*

Both were caught by **Codex review, not by any gate**. The 2026-07-26 entry's own "action taken"
field shows the class is normally discharged by filing something; here nothing was filed and
nothing was appended. The instances survive only in commit messages and the JOURNAL narrative.

### A3. ARCHITECTURE.md — **NOT UPDATED**

```
git log -1 --format='%h | %ci' -- ARCHITECTURE.md   ->   ed10b57 | 2026-07-26 16:59:11 +0200
git log --oneline 1dcee8b..HEAD -- ARCHITECTURE.md  ->   0 commits
ARCHITECTURE.md:2  last_reviewed: 2026-07-26
```

`last_reviewed` (2026-07-26) is **not stale against its own last commit touch** (2026-07-26), so
the `canonical_freshness` A2 gate correctly passes — confirmed live in section D.

**Does the roster describe validator behaviour? Yes.** `ARCHITECTURE.md:416` `## Validators and
enforcement`, quoted in full:

> - **`.\scripts\check.ps1`** — the pre-merge gate: `pytest` + `mypy` + `ruff`. Run before every
>   merge (CLAUDE §5); not wired to pre-commit. A non-blocking #97 claim-vs-reality report
>   (`scripts/validate_claims.py`) also runs as a section but does not gate.
> - **`tests/`** — pytest unit + integration suites. Unit suite (no API keys):
>   `pytest tests/ -m "not integration and not envcheck"`.
> - **Pre-commit:** `normalize-headers` … `canonical_freshness` (A2 `last_reviewed` gate; FAIL
>   blocks the commit) · `validate-sealed-keys` (#67 …) · `validate-docs-registry` (#68 … **fails
>   CLOSED** as `GUARD MALFUNCTION`) · `validate-audit-casing` (ADR-101 R4 audit-filename casing) ·
>   `validate-backlog` (ADR-66 story-map structure) · `ruff` … **Twelve hook ids total — this
>   roster mirrors `.pre-commit-config.yaml`.**
> - **External conformance (read-only):** `.dev-knowledge/scripts/audit.py` …

**Does it now misdescribe reality? No false statement found — but it is incomplete on two counts.**

- Nothing in it is falsified. "Twelve hook ids total" still holds
  (`grep -c "^\s*- id:" .pre-commit-config.yaml` → **12**). "pytest + mypy + ruff" still holds.
- **Finding A3-a (omission).** The block describes each validator's *gating* semantics and says
  nothing about *output*. The #126 reporting contract — every gating validator prints
  name/verdict/predicate/counts on success — was ruled and implemented this session and has **no
  home in ARCHITECTURE**. `grep -n -i -E "exit code|stdout|silent|prints|reporting contract|positive assertion"`
  over ARCHITECTURE.md returns no hit in the Validators section.
- **Finding A3-b (omission).** `check.ps1` is described only as "pytest + mypy + ruff". The
  session changed it to run every tool through `.venv\Scripts\python.exe` and to **fail loud when
  the venv is absent** — a behaviour a reader of ARCHITECTURE cannot predict. Not false; silent.

**Additional (outside the asked list, same class).** `CLAUDE.md` was also untouched this session
(`git log --oneline 1dcee8b..HEAD -- CLAUDE.md` → 0; `last_reviewed: 2026-07-26`). It carries
**five** separate "pre-merge gate = `.\scripts\check.ps1` (pytest + mypy + ruff …)" enumerations
— `CLAUDE.md:58`, `:85`, `:110`, `:159`, `:167`. None is false, but none mentions the venv
requirement either. CLAUDE.md v2.14's own history note records that these enumerations were
reconciled in-commit once before precisely so the gate description would not drift; that
discipline was not applied to this change.

### A4. docs/decisions/ — **NOT UPDATED**

- **15 ADRs.** Highest: **ADR-15**, slug `ADR-15-path-claim-resolution-model.md`.
- `git log --oneline 1dcee8b..HEAD -- docs/decisions/` → **0 commits**.

**Is the #126 reporting contract recorded only as a BACKLOG note?** **Yes — BACKLOG only.**
`grep -rln -i -E "reporting contract|positive assertion|exit 0 with zero bytes|silence, not a verdict" docs/decisions/`
returns **nothing**. The contract lives at `BACKLOG.md` inside the (now struck) #126 grooming
record and the ticket text carried into the log. **Finding A4-a:** a convention that binds every
gating validator in the repo, and that explicitly declares its own hub-supersession trigger, is
recorded only in an append-only work log — the surface ADR-65 says work items *leave*.

**Does ADR-11 carry a 2026-07-27 amendment?** **No — as expected.**
`grep -n "2026-07-27" docs/decisions/ADR-11-delegated-invocation-contract.md` → no hit.
ADR-11's latest markers are `:4` (2026-07-23 Deployment-Status refresh) and `:65`
(`## Amendment (2026-07-22)`); status line `:7` `**Status:** Accepted (ratified 2026-07-05 …)`.
This is architect-owed and correctly reported as such by #117 (see A6).

### A5. docs/audits/ — **NOT UPDATED**

- 46 audit `.md` files + `README.md` (`ls docs/audits/*.md | wc -l` → 47 incl. README).
- `git log --oneline 1dcee8b..HEAD -- docs/audits/` → **0 commits**.
- Last commit touching the directory: `d0dab43 | 2026-07-26 13:46:36`.
- `grep -rl "2026-07-27" docs/audits/` → **no file**. **No audit covers this session's arcs.**

**Finding A5-a — the load-bearing one in section A.** The session ran **9 Codex review passes
producing 13 findings** and **wrote none of them to disk**. The precedent in this very directory
is the opposite: `docs/audits/2026-07-26-codex-97-registry-repair.md`,
`2026-07-26-codex-crux-check-pass3.md`, `2026-07-26-codex-116-resolution-model.md` are all
committed Codex outputs from the day before. The reviews this session were run through direct
`codex exec` rather than the review script that writes a file, so **the entire review record —
including the two findings the session refuted and the four it introduced-then-fixed — exists
only in a chat transcript that the repo cannot see.** This directly weakens the closure claim
audited in A6 (criterion 3).

### A6. BACKLOG.md — live status of the six ids

Task-line presence (`grep "^- \[#<id>\]" BACKLOG.md`); total live tasks = **75**.

| id | live task line | status |
|---|---|---|
| #117 | yes, `BACKLOG.md:74` | open on one clause |
| #123 | yes, `BACKLOG.md:203` | open on the deferred clause |
| #124 | **no line** | closed (done tasks leave, ADR-65) |
| #125 | **no line** | closed |
| #126 | **no line** | closed |
| #127 | yes, `BACKLOG.md:161` | newly filed, open |

**#117 (`BACKLOG.md:74`) — ruling landed; open on the amendment clause. Quoted:**

> **RULED 2026-07-27 (operator, this window): NO — Lane A remains fire-and-forget by contract.
> Ambiguity resolves at the boost stage via the clarify loop. Interactivity remains a separable
> rider requiring its own ADR, to be opened only on evidence that the clarify loop fails in
> practice.** Routing executed in the same pass: the ruling forecloses interactivity, so #113 is
> driven to its named exit (b) and CLOSED as deliberately-not-doing with this ruling cited …
> **Remaining open clause, stated so this line cannot read as closed:** the done-when's ADR-11
> dated in-body amendment marker is deliberately NOT written this turn — ADR edits are out of the
> turn's scope and the amendment is flagged as a follow-up question to the architect; this line
> stays open on that single clause

**CONFIRMED: #117 is open on the ADR-11 amendment clause**, and A4 confirms ADR-11 carries no
2026-07-27 amendment — ticket and ADR agree.

**#123 (`BACKLOG.md:203`) — partial; open on the uninstall clause. Quoted:**

> **REPOINT LANDED 2026-07-27 (clauses 1 and 2 of the done-when DISCHARGED; the line stays OPEN
> on clause 3, which is not mine to execute).** …
> **CLAUSE 3 DEFERRED, and named rather than quietly dropped: `pip uninstall ai-council` on the
> SYSTEM interpreter is machine-level and outside this repo** … **The gap that leaves:** until it
> is uninstalled, a bare `python -m pytest` in this repo still resolves `ai_council` through the
> system editable install … the `conftest.py` guard only fires in a worktree, and its "fails
> loudly under bare python" half stays unproven until the uninstall happens ·
> **Done when (remaining): the operator runs `pip uninstall ai-council` on the system interpreter
> and the `conftest.py` guard is re-verified failing under bare `python` and passing under `.venv`**

**CONFIRMED: #123 is open on its deferred uninstall clause**, with the residual gap named.

**#127 (`BACKLOG.md:161`) — filed. Head + done-when quoted:**

> `- [#127] [P2][S] **The genai 2.x floor is proven for TYPING, not for live behaviour** (#124
> residual, filed 2026-07-27 on terra's pre-merge finding; id consumed per the Id-reservations
> note … and holds **#107** as the only live reservation — #127 is clear in both spaces)`
> … `Done when: a live `research --deep` gemini run is witnessed end-to-end against google-genai
> 2.x …, **and** the run's cost is explicitly authorised — this is a BILLED call and must not be
> spent unattended (the #66 discipline: `AICOUNCIL_LIVE_WITNESS=1` plus
> `AICOUNCIL_LIVE_WITNESS_BILLED=1`)`

**Closure records for the three struck ids** are present in the queue block and grooming log:
`~~#124~~ — CLOSED 2026-07-27 (L1a: …)`, `~~#125~~ — CLOSED 2026-07-27 (L2: …)`,
`~~#126~~ — CLOSED 2026-07-27 (L3: …)`.

#### Working-queue block — verbatim (`BACKLOG.md:16–48`)

```
## Working queue — 2026-07-27 (reserved axis slots; [#102]'s record half)

> **Status: NOT mechanically validated.** [#102]'s mechanism legs — the `validate_backlog.py`
> queue-membership check (leg i) and the `session_end_backpressure.py` regroom nudge (leg ii) —
> stay capped and unbuilt; nothing checks this block against the open set. An unvalidated queue
> that PRESENTS as current is the same defect class this window exists to close (LESSONS
> 2026-07-26 — a value published without its predicate); one that discloses its own status is
> not. Authored by the 2026-07-27 record-and-report window, from the session plan approved by
> the outgoing architect; it does not self-update.
>
> **The top slots are RESERVED, not ranked — the reservation is structural, not a ranking
> outcome.** Every axis item below is P3 and every defect is P2, so a purely priority-ranked
> queue re-buries the axes and reproduces the exact failure this queue exists to prevent
> (seventh window for #27; the non-cognitive axis at zero for six). Do not read this ordering as
> a priority sort.

Reserved top slots, in order:
1. #27 — CLI-4 parity run (operator-gated)
2. #117 — the ADR-11 decision-1 interactivity ruling (ruling owed; closure criterion 4 below governs it)
3. #103 — the #19 framing-defense design commission (architect-owed)
4. The Contract-Version 1.1 unit — #34 + #76 + #100, versioned together, never split

Defect queue, below the reserved slots (the plan's order):
5. ~~#124~~ — CLOSED 2026-07-27 (L1a: …) -> 6. #123 — **PARTIAL 2026-07-27**: the repoint landed
   and the gate runs through `.venv` (green, and it fails loud without one); the line stays open
   on its machine-level `pip uninstall` clause, which is the operator's to run
7. ~~#125~~ — CLOSED 2026-07-27 (L2: …)
8. ~~#126~~ — CLOSED 2026-07-27 (L3: …)
9. #105
10. #119 / #120
11. unit (b) — #97 rule 1 + rule 14 leg (b), one unit on the pinned predicate
12. #118

**Closure status, 2026-07-27 (execution window):** criterion **1 MET** … criterion **2 MET** …
criterion **3 MET** — every code lane carried a named terra review (13 findings across 9 passes:
11 accepted, 2 refuted with reproductions) … criterion **4 MET** — the ruling was made, not
deferred. Defect entries below struck accordingly; **the reserved axis slots were NOT reordered
and did NOT move.**
```

**Reserved axis slots: UNCHANGED and UNREORDERED.** Slots 1–4 are `#27`, `#117`, `#103`, the
`#34 + #76 + #100` unit, in the original order. Only defect-queue entries 5, 7, 8 were struck and
entry 6 annotated. Confirmed against the block as authored.

**Finding A6-a — stale text inside the queue block.** Slot 2 still reads
`#117 — the ADR-11 decision-1 interactivity ruling (ruling owed; …)`. The ruling **was made this
window** (see #117 above); what remains owed is the ADR-11 *amendment*, not the ruling. The
parenthetical is now false as written. The same block's closure status three lines below says
"the ruling was made, not deferred" — **the block contradicts itself.**

**Finding A6-b — a stale count inside #127.** `#127` states #124 proved a green gate at
"pytest 931, mypy, ruff, in BOTH environments". The suite is now **932** (section B1). 931 was the
count *before* #127's own guard test was added in the same lane. Not wrong at the moment it was
written; wrong as a present-tense claim on a live ticket.

**Finding A6-c — criterion 3 has no repo-side evidence.** The closure block asserts "13 findings
across 9 passes: 11 accepted, 2 refuted with reproductions." Per **A5**, nothing in `docs/audits/`
records any of it. This claim cannot be verified from the repository — it is a value published
without a retrievable predicate, asserted inside the very block that warns against that pattern.

---

## Section B — test-quality audit

### B1. Full pytest run — **932 passed, 6 deselected, 1 xfailed, 0 failed. Exit code 0.**

```
.venv/Scripts/python.exe -m pytest tests/ -m "not integration and not envcheck" -q
932 passed, 6 deselected, 1 xfailed in 195.84s (0:03:15)
PYTEST-EXIT-CODE:0
```

Matches the session's reported 932.

### B2. The test-isolation defect — **FOUND, FIXED, NOT FILED**

**The defect as it stood before the session** (`git show 1dcee8b:tests/test_validate_claims.py`,
lines 502–506):

```python
    def verdict():
        import importlib
        m = importlib.import_module("validate_claims")
        m._canonical_docs = lambda ctx: ["DOC.md"]     # <-- direct module attribute assignment
        return m.rule_2(m.RepoContext(repo))
```

A direct attribute assignment on the imported module inside a helper called twice — never
unwound, so `_canonical_docs` stayed pinned to `["DOC.md"]` for every later test in the session.

**Current state — repaired** at `tests/test_validate_claims.py:507` / `:522`:

```python
507: def test_r2_verdict_is_identical_with_and_without_untracked_debris(tmp_path, monkeypatch):
...
522:     monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
...
     def verdict():
         return vc.rule_2(vc.RepoContext(repo))
```

Fixing commit: `5af0a7a` (`git log -S "m._canonical_docs = lambda" -- tests/test_validate_claims.py`).

**Is it FILED as a ticket? NO.** `grep "^- \[#" BACKLOG.md` carries no task line for it. The only
BACKLOG mention is narrative prose inside the grooming log (`BACKLOG.md:224`):
*"Also repaired a pre-existing test-isolation defect this lane exposed: a rule-2 test assigned
`vc._canonical_docs` directly on the module instead of via `monkeypatch` …"*. That is a **record
of the repair, not a ticket**.

**Finding B2-a.** No mechanical guard prevents recurrence. A future test can reintroduce a direct
module-attribute assignment and nothing will catch it — the instance was closed, the class was
not. (A hub-owned fleet-intake commission "environment/test isolation" exists from 2026-07-26,
but it predates this instance and is not this repo's to discharge.) Sweep for survivors:
`grep -rn -E "^\s*(vc|m|module)\.[_a-zA-Z]+ = " tests/*.py | grep -v monkeypatch` → **no hits**;
the suite is currently clean of the pattern.

### B3. Different collection order — **no order-dependent failure**

Default order is alphabetical by file. I re-ran with **module order reversed**, which is the axis
on which module-level state pinning actually bites:

```
FILES=$(ls tests/test_*.py | sort -r)      # 30 files, reversed
.venv/Scripts/python.exe -m pytest $FILES -m "not integration and not envcheck" -q
932 passed, 6 deselected, 1 xfailed in 197.12s (0:03:17)
REVERSE-ORDER-EXIT:0
```

Reversed order puts `test_validate_sealed_keys.py`, `test_validate_docs_registry.py`,
`test_validate_claims.py` first instead of near-last. **Identical counts, exit 0.** No test passes
in default order and fails in reversed order.

**Caveat, stated so this is not over-read:** this is *file*-level reversal, not per-test shuffling.
It exercises cross-module pinning (the class B2 found) but would not catch intra-module ordering
dependence within a single file. No shuffling plugin (`pytest-randomly`, `pytest-reverse`) is
declared or installed, so per-test randomisation was not available without adding a dependency —
out of scope for a read-only audit.

### B4. Success-path output assertions — **all four validators have one**

| validator | test | success assertion (quoted) |
|---|---|---|
| `canonical_freshness_gate` | `tests/test_canonical_freshness_gate.py:63` `test_success_prints_positive_assertion` | `:68 assert "canonical_freshness: OK" in res.stdout` · `:69 assert "1 canonical doc(s) checked" in res.stdout` · `:70 assert "last_reviewed" in res.stdout` |
| `validate_audit_casing` | `tests/test_validate_audit_casing.py` `test_success_prints_positive_assertion` | `:220 assert "validate_audit_casing: OK" in res.stdout` · `:221 assert "1 audit filename(s) in R4 scope" in res.stdout` · `:222 assert "kebab-case" in res.stdout` |
| `validate_sealed_keys` | `tests/test_validate_sealed_keys.py:204` `test_e2e_success_prints_positive_assertion` | `:210 assert "validate_sealed_keys: OK" in res.stdout` · `:211 assert "1 staged add(s) checked" in res.stdout` · `:212 assert "sealed-key" in res.stdout` |
| `validate_docs_registry` | `tests/test_validate_docs_registry.py:354` `test_e2e_success_prints_positive_assertion` | `:361 assert "validate_docs_registry: OK" in res.stdout` · `:362 assert "1 staged add(s)" in res.stdout` · `:363 assert "0 new docs/ dir(s)" in res.stdout` · `:364 assert "2 registered corpus row(s)" in res.stdout` |

Each asserts on a **non-empty, count-bearing** success line, not merely a zero exit — so a
validator going silent on success would fail its test. Each also has a **zero-item** companion
(`"0 canonical doc(s) checked"`, `"0 staged add(s) checked"`, …), which is what makes a
checked-nothing run distinguishable from a clean one.

Two **negative** guards additionally pin that no false OK is printed on a non-success path:
`tests/test_canonical_freshness_gate.py:105 assert "canonical_freshness: OK" not in res.stdout`
(A2 FAIL) and `tests/test_validate_audit_casing.py:243 assert "validate_audit_casing: OK" not in captured.out`
(fail-open path, which checked nothing).

**Answer: none of the four is tested only on its failure path.**

### B5. Rule 4 — the surface-set regression guard **EXISTS**

`tests/test_validate_claims.py:1222`:

```python
def test_r4_reads_all_four_surfaces_on_the_live_repo():
    """Measurement against the real tree: the widened rule must actually consult four surfaces.
    Kept separate from the verdict so a live finding is reported as a measurement, never
    silenced by a doc edit inside this lane."""
    ctx = vc.RepoContext(_REPO)
    assert vc._adr_roster_docs(ctx) == [
        "CLAUDE.md", "ARCHITECTURE.md", "docs/decisions/README.md"], \
        "the spec names four surfaces: ADR files on disk + these three rosters"
```

**This is the test that fails if the surface set shrinks back to one** — it asserts list equality
against the exact three roster surfaces, so reverting `_adr_roster_docs` to `("CLAUDE.md",)` fails
it immediately.

Four behavioural guards back it up, each seeding a stale roster in a *different* surface:
`:975 test_r4_four_surfaces_all_agree_passes`, `:980 test_r4_catches_stale_architecture_roster`,
`:990 test_r4_catches_stale_readme_index`, `:997 test_r4_catches_stale_claude_roster`, plus
`:1004 test_r4_is_bidirectional_roster_names_a_local_adr_not_on_disk` for the reverse direction.
Shrinking the set would fail the ARCHITECTURE and README guards even if the equality assert were
deleted.

---

## Section C — claim-vs-reality on the session's own claims

### C1. "check.ps1 runs every tool through the repo venv" — **VERIFIED, no bypass**

Every tool invocation in `scripts/check.ps1`, exhaustively:

| line | invocation |
|---|---|
| `:20` | `$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"` |
| `:56` | `& $Py -m pytest tests/ -m "not integration and not envcheck" -v` |
| `:61` | `& $Py -m mypy src/` |
| `:67` | `& $Py -m ruff check src/ tests/` |
| `:76` | `& $Py scripts/validate_claims.py` |

`grep -n -E '^\s*(pytest|mypy|ruff|python|py) ' scripts/check.ps1` → **no hits**. No bare runner
survives; all four tools go through `$Py`. `:24 if (-not (Test-Path $Py))` guards absence.

One precision: `:76` invokes the checker as a script path rather than `-m`, but still through
`$Py`, so the claim holds. The claim is **true as stated**.

### C2. "declared ranges reproduce a green gate from a bare clone" — ranges reported verbatim

Not re-run (explicitly out of scope). `pyproject.toml`, verbatim:

```toml
dependencies = [
    "click>=8.1",
    "rich>=13.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "anthropic>=0.40",
    "openai>=1.50",
    "google-genai>=2.14,<3.0",      # (preceded by a 9-line comment recording the #124 rationale)
    "python-frontmatter>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.8",
    "ruff>=0.15.5",
    "mypy>=1.10",
    "types-PyYAML>=6.0",
    "pre-commit>=4.5",
]
```

**Observation, not a contradiction.** `google-genai` is the only bounded range. `openai>=1.50`,
`anthropic>=0.40`, `mypy>=1.10` and the rest remain **unbounded upward** — the identical shape
that let `google-genai>=1.55.0` span a breaking change and produce the red clean-room gate #124
existed to fix. #124 closed the instance; the range *class* is unchanged for seven other
dependencies. `#20` already tracks the `openai` 2.x type-stub half of this. The "reproduces a
green gate" claim is therefore **true only for the resolution set that existed when it was
measured**, and nothing in the repo pins or re-checks it on a schedule — #127 records exactly
this residual for google-genai but not for the others.

### C3. "rule 4 reads four surfaces bidirectionally" — **TRUE with one material precision**

`scripts/validate_claims.py:620`:

```python
_ADR_SURFACES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("CLAUDE.md", _ADR_SECTION, "adrs"),
    ("ARCHITECTURE.md", re.compile(r"governing adrs", re.IGNORECASE), "governing adrs"),
    ("docs/decisions/README.md", re.compile(r"^\s*index\s*$", re.IGNORECASE), "index"),
)
```

`:652`:

```python
def _adr_roster_docs(ctx: RepoContext) -> list[str]:
    """The roster surfaces to compare -- every one the spec names that exists in HEAD's tree."""
    return [rel for rel, _sec, _w in _ADR_SURFACES if ctx.exists(rel)]
```

**Returned set: three roster docs.** The "four surfaces" of the spec are the **ADR files on disk +
these three rosters** — the disk side enters separately via `ctx.glob("docs/decisions/ADR-*.md")`.
The live report line corroborates the four:
`reads: docs/decisions/ADR-*.md, CLAUDE.md, ARCHITECTURE.md, docs/decisions/README.md`.
So "four surfaces" is **correct**, and a reader who expects `_adr_roster_docs` to return four
items will be briefly misled — it returns three by design.

**Is the comparison genuinely both directions? Yes** — `scripts/validate_claims.py:816-817`:

```python
816:        missing = [a for a in disk_ids if a not in _declared_local_adrs(body)]
817:        extra = sorted(_claimed_local_adrs(body) - set(disk_ids), key=lambda a: (len(a), a))
```

`missing` = disk → roster; `extra` = roster → disk. Both directions are computed and both raise
findings.

**Material precision (Finding C3-a).** The two directions use **different predicates**, so this is
bidirectional but **not symmetric set-equality**, which is what the #97 spec's `==` literally
says:

- `:697 _declared_local_adrs` — *"PERMISSIVE set, for the disk -> roster direction: any entry in
  the local block naming an ADR documents it."*
- `:705 _claimed_local_adrs` — *"STRICT set, for the roster -> disk direction: only ENTRY-SHAPED
  lines (list items and table rows) make a local claim."*

The asymmetry is deliberate, documented in both docstrings, and was introduced to kill a measured
false-positive class (all three live rosters name hub-owned ADR-43 inside ADR-07's entry). It is
sound engineering. But **`A == B` is not what runs**; what runs is `A ⊆ permissive(B)` and
`strict(B) ⊆ A`. #97's rule-4 spec text still says set-equality across four surfaces, and #125's
own instruction was "do not narrow the SPEC to match the code". Nothing in #97 or the report
records that the implemented relation is two asymmetric containments. That is a spec-vs-code gap
of exactly the kind rule 4 exists to catch, one level up.

---

## Section D — validator runs (exit code and stdout reported separately)

Six organs; `validate_claims` is one of the six (`validate_*` × 5 + `canonical_freshness_gate`).
All run through `.venv/Scripts/python.exe`.

**D1 `scripts/validate_backlog.py`** — exit code **0**
```
validate_backlog: OK (8 themes, 15 stories, 75 tasks, 0 warning(s))
```

**D2 `scripts/canonical_freshness_gate.py`** — exit code **0**
```
canonical_freshness: OK (4 canonical doc(s) checked -- last_reviewed not predating each doc's last commit (A2), 30d cadence (A1, warn-only); 0 warning(s))
```

**D3 `scripts/validate_docs_registry.py`** — exit code **0**
```
validate_docs_registry: OK (0 staged add(s), 0 new docs/ dir(s) vs HEAD, checked against 2 registered corpus row(s) + 1 taxonomy folder(s) read from docs/audits/README.md -- no unregistered new docs/ directory)
```

**D4 `scripts/validate_sealed_keys.py`** — exit code **0**
```
validate_sealed_keys: OK (0 staged add(s) checked, 0 sealed key(s) explicitly authorized -- no unauthorized sealed-key-shaped .json (SEALED+KEY in either order) is staged)
```

**D5 `scripts/validate_audit_casing.py`** — exit code **0**
```
validate_audit_casing: OK (0 staged add(s) checked, 0 audit filename(s) in R4 scope -- docs/audits/*.md names all-lowercase kebab-case, ADR-101 R4)
```

**D6 `scripts/validate_claims.py`** — exit code **0**
```
  [SKIP] rule  1 module-table-completeness  (Unit 2)  reads: none
  [PASS] rule  2 path-existence  reads: HEAD:<tracked tree>, protocols/*.md, CLAUDE.md, ARCHITECTURE.md, VISION.md, CONTRIBUTING.md, protocols/COUNCIL_INVOCATION_CONTRACT.md, protocols/COUNCIL_QUESTION_GUIDE.md, protocols/README.md, protocols/SYNTHESIS_QUALITY_RUBRIC.md
  [PASS] rule  3 hook-roster-parity  reads: .pre-commit-config.yaml, CLAUDE.md, ARCHITECTURE.md
  [PASS] rule  4 adr-roster-parity  reads: docs/decisions/ADR-*.md, CLAUDE.md, ARCHITECTURE.md, docs/decisions/README.md
  [SKIP] rule  5 config-parity  (Unit 2)  reads: none
  [SKIP] rule  6 cli-surface-parity  (Unit 2)  reads: none
  [SKIP] rule  7 stamp-honesty  (Unit 2)  reads: none
  [PASS] rule  8 sha-reachability  reads: JOURNAL.md, BACKLOG.md
  [SKIP] rule  9 durations-regression  (Unit 2)  reads: none
  [SKIP] rule 10 dep-parity  (Unit 2)  reads: none
  [SKIP] rule 11 invariant-spot-checks  (Unit 2)  reads: none
  [SKIP] rule 13 ticket-reference  (Unit 2)  reads: none
  [SKIP] rule 14 layer-edge-conformance  (Unit 2)  reads: none
SUMMARY: pass 4 | FINDINGS 0 across 0 rules | anchor-missing 0 | skipped(Unit2) 9 | errors 0
```

**Read of D as a whole.** All five formerly-silent organs now print a name, a verdict, the
predicate evaluated, and an item count — the #126 contract is live and observable, and D2–D5 are
the direct evidence. Four of the five report **zero items examined** in this working-tree state
(nothing staged), which is exactly the condition that used to be indistinguishable from a clean
run and is now stated. `validate_claims` reports 4 rules PASS and **9 of 14 rules SKIP**: the
`FINDINGS 0` headline continues to rest on a third of the specified checker, and its own
`reads:` disclosure now makes that visible per rule.

---

## Summary of findings

| # | Finding | Severity |
|---|---|---|
| A5-a | 9 Codex passes / 13 findings written to no audit file; the entire review record is off-repo, breaking precedent set the day before | **high** |
| A6-c | Closure criterion 3 asserts the review record that A5-a shows does not exist in the repo | **high** |
| A2-a | Two fresh, self-reported instances of the repo's own "guarantee without its mechanism" class; no LESSONS entry, caught by Codex not by a gate | **high** |
| C3-a | Rule 4 implements two asymmetric containments, not the `==` set-equality its spec states; unrecorded | **medium** |
| A4-a | The #126 reporting contract binds every gating validator but lives only in BACKLOG, a log whose items leave | **medium** |
| A6-a | Queue slot 2 says "ruling owed" and the same block says "the ruling was made" — self-contradictory | **medium** |
| A3-a/b | ARCHITECTURE describes validator gating but not the new output contract or the venv requirement — omission, nothing false | **medium** |
| B2-a | The test-isolation defect was fixed but never filed, and no mechanical guard prevents recurrence | **medium** |
| C2 | Seven dependencies keep the unbounded-range shape that caused #124; only `google-genai` was bounded | **medium** |
| A6-b | #127 cites "pytest 931" as a present-tense claim; the suite is 932 | **low** |
| A1-a | `[#50]` calls the anchor tail "ONE-COMMIT"; it is a commit plus its merge (2) | **low** |

**Verified sound, no finding:** A1 anchor requirement (5 shas, all resolve, 18 of the 20 unnamed
covered transitively); A6 reserved axis slots unchanged and unreordered; B1/B3 (932 passed in both
default and reversed module order); B4 (all four validators assert success-path output, with
zero-item and no-false-OK companions); B5 (the surface-set regression guard exists and is
exact-equality); C1 (no bare runner in `check.ps1`); D (all six organs print a predicate-bearing
line; no organ is silent).

**Not done, per instruction:** nothing repaired, nothing filed, no repo file written.
