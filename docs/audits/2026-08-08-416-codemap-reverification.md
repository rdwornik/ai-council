# Codemap re-verification — hub [#416], `ARCHITECTURE.md` L23/L109

**Date:** 2026-08-08 · **Repo:** `ai-council` · **Branch:** `worktree-lane-b-416-codemap`
**Discharges:** hub `.dev-knowledge` BACKLOG **[#416]** — *"ai-council `ARCHITECTURE.md` codemap
drift at L23/L109"*, carved out of [#262]'s Done-when so the concrete defect survived that closure.
**Done-when (verbatim):** *"ai-council `ARCHITECTURE.md` L23/L109 are re-verified and corrected, or
recorded accurate-as-is with a reason."*

**Method.** Module rows were swept against disk (top-level `.py` files + subpackages + repo-root
`config/`, with each module's docstring one-liner). Dependency edges were re-derived from
`src/ai_council/` + `config/` by AST walk — runtime imports including function-level ones,
`TYPE_CHECKING`-guarded imports excluded, and the `providers` / `research` / `config` package nodes
collapsed exactly as the codemap's own module table collapses them. No generator was run: `codemap
generate` is policy-barred for this flat single-package layout (the L20 note, #262 gap-note).

**Derivation cross-check.** The AST run independently reproduces **14 `cli` edges** — the same
number the file's own 2026-07-24 AST note records, edge-for-edge. That agreement is what licenses
trusting the run's verdict on the *other* 19 modules, which no prior pass had derived.

---

## 1. Per-line verdicts (the two lines the row names)

| Line | What sits there | Verdict | Reason |
|---|---|---|---|
| **L23** | The `<!-- HAND-AUTHORED compact-text codemap … NOT generator-managed -->` disclaimer | **ACCURATE-AS-IS — no edit** | L23 is not codemap *content*; it is the hand-authored-by-policy ruling that [#262]/[#295] settled, correctly present and correctly worded. Its three claims each hold live: the block *is* hand-maintained; `codemap generate` *does* degenerate on this flat no-`tach.toml` layout; and `codemap check` *is* wired to no gate. Nothing to correct. |
| **L109** | `\| healthcheck.py \| foundation \| Provider health checks; startup gate \|` | **ACCURATE-AS-IS — no edit** | Matches the live module: `healthcheck.py:1` docstring reads *"Provider health checks — ping each API before starting a debate."* Both halves of the row (the checks, the startup gate) are true of the code. |

**Both named lines resolve accurate-as-is.** The census locators were filed 2026-07-08 against a
*Mermaid-era* file; the 2026-07-23 reconciliation had already converted both blocks to ADR-51
compact text, so the drift the census recorded was discharged before this pass began. What the sweep
found instead is real drift the census never named — see §3.

---

## 2. Fact table — every codemap module row ↔ live module

Codemap module rows sit at `ARCHITECTURE.md` L33–L52 (pre-commit numbering); responsibilities rows
at L92–L111.

| module | codemap row (layer, path) | on disk | docstring one-liner (locator) |
|---|---|---|---|
| cli | interface, `src/ai_council/cli.py` | ✓ | "Click CLI — parses args, builds RunRequest, delegates to CouncilRunner." (`cli.py:1`) |
| inbox | interface, `src/ai_council/inbox.py` | ✓ | "Inbox folder scanning, frontmatter parsing, and archive logic." (`inbox.py:1`) |
| orchestrator | orchestration, `src/ai_council/orchestrator.py` | ✓ | "CouncilRunner: coordinates the full debate lifecycle." (`orchestrator.py:1`) |
| runner | orchestration, `src/ai_council/runner.py` | ✓ | "Panel and provider utility functions." (`runner.py:1`) |
| doctor | orchestration, `src/ai_council/doctor.py` | ✓ | "council doctor -- liveness + config pre-flight (DRAFT-DOC-1 v1)." (`doctor.py:1`) |
| boost | core, `src/ai_council/boost.py` | ✓ | "Boost stage — the Council's input stage (ADR-11 boost→decide chain, Unit 2 P1)." (`boost.py:1`) |
| crux_check | core, `src/ai_council/crux_check.py` | ✓ | "Bounded crux check between Round 1 and Round 2 (#18)." (`crux_check.py:1`) |
| debate | core, `src/ai_council/debate.py` | ✓ | "Debate orchestration: parallel model calls, critique rounds." (`debate.py:1`) |
| synthesis | core, `src/ai_council/synthesis.py` | ✓ | "Final synthesis: build transcript, call synthesizer, return DebateResult." (`synthesis.py:1`) |
| mode_detector | core, `src/ai_council/mode_detector.py` | ✓ | "Cheap LLM call to classify a question into a debate mode." (`mode_detector.py:2`) |
| seat_router | core, `src/ai_council/seat_router.py` | ✓ | "Seat router — the CLI-seat admission gate + same-seat API fallback (L-CLI IF#2)." (`seat_router.py:1`) |
| providers | core, `src/ai_council/providers/` | ✓ | *no module docstring* (`providers/__init__.py` empty) |
| research | core, `src/ai_council/research/` | ✓ | "Research mode package: parallel multi-provider web research." (`research/__init__.py:1`) |
| output | output, `src/ai_council/output.py` | ✓ | "Rich console output and markdown file save for debate results." (`output.py:1`) |
| routing | output, `src/ai_council/routing.py` | ✓ | "Target project resolver for per-invocation transcript routing." (`routing.py:1`) |
| models | foundation, `src/ai_council/models.py` | ✓ | "Pure dataclasses for the AI Council debate pipeline. No logic, no deps." (`models.py:1`) |
| metrics | foundation, `src/ai_council/metrics.py` | ✓ | "Cost and performance metric computation for debate runs." (`metrics.py:1`) |
| healthcheck | foundation, `src/ai_council/healthcheck.py` | ✓ | "Provider health checks — ping each API before starting a debate." (`healthcheck.py:1`) |
| policy | foundation, `src/ai_council/policy.py` | ✓ | "RunPolicy: debate behavior thresholds and rules. No execution logic." (`policy.py:1`) |
| config | foundation, `config/` (repo root) | ✓ | "Load settings.yaml into typed dataclasses. Validates API keys at startup." (`config/config_loader.py:1`) |

- **Declared but absent from disk:** none.
- **On disk but absent from the codemap:** none.
- **In the codemap but missing from the responsibilities table (or vice versa):** none — 20/20, both directions.

**The module half of the block is clean.** No module row was corrected.

---

## 3. What the sweep did find — the dependency list (corrected)

The codemap declared **32** edges. Against source: **8 do not exist** and **34 real edges were
unmapped**. Corrected to **58**.

### 3a. Phantom edges removed (8)

| removed edge | why it is not real |
|---|---|
| `inbox -> orchestrator` | `inbox.py`'s only internal import is a `TYPE_CHECKING` one of `routing` |
| `runner -> mode_detector` | `runner.py` imports `providers.base` and `config` — nothing else |
| `runner -> healthcheck` | same |
| `runner -> debate` | same |
| `debate -> synthesis` | `debate.py` imports `models`, `policy`, `providers`, `seat_router`, `config` |
| `synthesis -> output` | `synthesis.py` imports `metrics`, `models`, `providers`, `config` |
| `output -> routing` | `output.py` imports `models` and nothing else internal; `routing` is reached by `cli` |
| `research -> providers` | collapse artefact: `research/` has its **own** `research/providers/` subpackage and never imports `ai_council/providers/` |

Four of these are the **reverse** of a real edge — the real ones are `orchestrator -> debate`,
`orchestrator -> synthesis`, `research -> output`, `cli -> routing`. A hand-maintained map drifts
directionally, not just by omission.

### 3b. `TYPE_CHECKING`-only edges

Real in the type graph, absent from the runtime graph, now named separately rather than silently
dropped: `inbox -> routing`, `models -> policy`.

### 3c. Consequence for the layer section (coupled prose)

Completing the map falsified three standing claims in *Layer Boundaries & Invariants*, all corrected
in the same commit — this is the "immediately adjacent prose" the contract permits, nothing wider:

1. **"11 real `cli` edges are absent from the hand-maintained codemap"** — no longer true; all 14
   are mapped. Recorded as *closed by hand*, with BACKLOG **#97 rule 14 leg (b)** (the
   *mechanised* check) explicitly still open. A hand-run is true on the day it runs and decays from
   the next commit; a one-off pass is not a gate.
2. **The allowed set's cited instances** — `output -> output` cited `output -> routing`, a phantom,
   so that class is now marked **instance-free** (retained as a target, not a live edge);
   `orchestration -> foundation` cited `runner -> healthcheck`, also a phantom, replaced with the
   real instances.
3. **"the only edge in the codemap that skips a layer"** (of `cli -> boost`) — true of the old map,
   false of source. `cli -> boost` stays the *named* open case (it has a live ruling question);
   the rest are unaccounted, not open.

**Newly surfaced: 5 non-`cli` unaccounted edges**, in 4 classes. The prior note was a hand-run of
leg (b) against `cli.py` alone, so it could not see them:

| class | instances | reading |
|---|---|---|
| `foundation -> core` | `healthcheck -> providers` | **A layer inversion — the sharpest finding.** `healthcheck.py` is classed `foundation` yet imports `providers.base`. Either it is misclassified (a startup gate over providers is orchestration-shaped work), or `providers.base` is a foundation-level ABC the `providers` node's `core` class hides. |
| `foundation -> foundation` | `metrics -> models`, `metrics -> config` | Same-layer; the set simply never declared a same-layer foundation class |
| `orchestration -> output` | `orchestrator -> output` | The orchestrator writes results directly rather than through `core` |
| `output -> foundation` | `output -> models` | Reads the shared dataclasses |

Unaccounted-edge inventory therefore grows **10 → 15**. Of the 58 real edges: **42 inside the
allowed set, 1 named open case, 15 unaccounted.**

**The allowed set was NOT widened** (R1, 2026-07-24). Completing a map exposes defects; it does not
legalise them — which is precisely why the inventory grew rather than shrank. Three of the four new
classes read as *declaration gaps* (the set was enumerated against an incomplete map, so classes
with no mapped instance were never written down); `foundation -> core` reads as a genuine defect.
**That triage is a reading, not a ruling** — none of them is resolved here.

---

## 4. Diffstat

```
d30db1a docs(architecture): re-verify the codemap against source; 8 phantom edges out,
        34 real edges in (hub [#416])
 ARCHITECTURE.md | 198 ++++++++++++++++++++++++++++++++++--------
 1 file changed, 151 insertions(+), 47 deletions(-)
```

Also in that commit: the `research/` responsibilities row gains `headless.py` (the no-console entry
point `crux_check.py` calls, #18) — it was the one file missing from that row's enumeration — and
names the 5 research providers explicitly. The `last_reviewed` stamp moves 2026-07-27 → 2026-08-08
with its **scope stated in-file**: this pass verified the Codemap / responsibilities / layer
sections deterministically against source; the remaining sections were re-read but not
independently re-derived, and their standing verification remains the 2026-07-23 reconciliation.

---

## 5. Suite

`pytest tests/ -m "not integration and not envcheck"` — **928 passed, 4 failed, 6 deselected,
1 xfailed** (501s).

**The 4 REDs are pre-existing baseline, not this lane's.** The run was executed in the primary
checkout on clean `main`, and this lane's diff touches exactly one markdown file (`ARCHITECTURE.md`)
— zero Python. **Lane test delta: 0.**

Characterised, not fixed (out of the frozen scope):

```
tests/test_cli.py::test_interactive_debate_required_write_failure_exits_nonzero
tests/test_cli.py::test_interactive_research_required_write_beats_runtimeerror_branch
tests/test_cli.py::test_inbox_batch_does_not_abort_and_exits_nonzero
tests/test_doctor.py::test_run_doctor_record_write_failure_contained_non_oserror
```

All four are **Rich-markup assertion failures, not product defects**: the expected text *is* in the
output, but Rich interleaves ANSI bold codes inside it, so the substring check misses. Witnessed
directly —

```
assert '(not written)' in '… record: \x1b[1m(\x1b[0mnot written\x1b[1m)\x1b[0m …'
```

`NO_COLOR=1` does **not** suppress it (re-run confirms both still fail), so this is Rich's
highlighter, not colour output. Environment-dependent RED. **Filed here as an observation for the
operator; no BACKLOG row added — the integrator owns row creation.**

---

## 6. Self-test against the frozen acceptance contract

| # | Item | Verdict |
|---|---|---|
| 1 | Fact table in the packet: every codemap row ↔ live module, with locators | **PASS** — §2, 20/20 both directions, docstring locators per row |
| 2 | L23 and L109 each carry an explicit verdict | **PASS** — §1; both **accurate-as-is**, reasons recorded, no edit to either line |
| 3 | Other drifted rows corrected in the same hand-authored shape, or listed as leftovers | **PASS** — §3; the dependency list corrected by hand in ADR-51 compact-text shape, no generator |
| 4 | No generator run; hooks untouched; ToC hook allowed to fire; suite green; branch clean at STOP | **PASS with one disclosure** — no generator, no hook edits, all gates ran and passed on commit (ToC hook: `no files to check` — this doc has no ToC), branch clean. **Suite is NOT green: 4 pre-existing baseline REDs**, lane delta 0 (§5). Disclosed rather than claimed green. |
| 5 | Disposition report committed; packet carries its path | **PASS** — this file |

**Declared decisions taken under the lane budget** (reported, not escalated):

1. **Scope of the coupled-prose edit.** The contract permits *"the codemap block and, only if a
   correction requires, its immediately adjacent prose."* Completing the map falsified three
   standing claims in the very next section; leaving them would have made the file
   self-contradictory, so they were corrected. The 5-edge non-`cli` inventory (§3c) goes one step
   further than strict adjacency — it is the leg-(a) reading of the completed map. Kept because the
   corrected intro sentence asserts "15 unaccounted" and an unsupported count is worse than none.
   **Flagged for the integrator to accept or trim.**
2. **Freshness stamp.** `canonical_freshness` blocks a commit to a canonical doc whose
   `last_reviewed` predates the edit, and the contract bars `--no-verify`, so the stamp had to move.
   It was bumped with its verification depth stated in-file rather than implying a whole-file
   re-derivation that did not happen.
3. **The 4 REDs were characterised but not fixed**, and no BACKLOG row was opened for them
   (contract: no BACKLOG edits — the integrator owns rows).

**Contract deviation, disclosed:** the lane was dispatched from the hub checkout rather than the
ai-council one, so it first landed on a branch in the **live** ai-council checkout. That was
reverted (`git checkout -- ARCHITECTURE.md`, branch deleted, tree confirmed clean on `main`) before
any commit, and the work was redone in the worktree `.claude/worktrees/lane-b-416-codemap` per
RULING-W. **No commit ever landed in the live consumer checkout**, and nothing was pushed.

---

**COMMIT-AND-STOP.** Branch `worktree-lane-b-416-codemap` is handed back unmerged; integration is
operator-gated through the consumer's primary checkout. The hub files the disposition.
