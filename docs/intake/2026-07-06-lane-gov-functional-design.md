# ai-council — L-GOV Functional Design: Record & Governance Hygiene (pillar 5)

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — DRAFT-GOV-1 ratified as ADR-14 (`85a692f`), currency pass #31; open: #50 (DRAFT-GOV-2 Watches). _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-06 · **Mode:** DESIGN — functional architecture only; zero source/config changes; this document is the lane's sole repo artifact
**Author:** Fable 5 (L-GOV lane session, one of five parallel worktree lanes; parallel-mode wrap — committed on the worktree branch, integration serialized by the primary)
**Sources verified live:** fleet-recon audit (`2593075`), Fable architecture audit (`cd4a5e1` copy), ratified state at merge `5c81e71` (ADR-11/12 Accepted, CONTRACT, Wave-0), `docs/decisions/ADR-01..12`, `BACKLOG.md`, `.pre-commit-config.yaml`, `.claude/settings.json`, `~/.claude/settings.json`, `scripts/session_end_backpressure.py`, `scripts/canonical_freshness_gate.py`, JOURNAL 2026-07-05 entries
**Checkpoint:** skeleton approved by operator 2026-07-06 (GOV-1 interpretation confirmed; fast-forward endorsed; pause fork = recommendation only, ruling stays with operator)

---

## 1. Lane charter & current state (verified live)

### 1.1 Entry-state verification — all charter claims checked against the live repo

| Charter claim | Verdict | Live evidence |
|---|---|---|
| ADR-09/10 status divergence | **CONFIRMED** | ADR-09/10 headers + `docs/decisions/README.md` index say `Proposed`; both are implemented and load-bearing (`protocols/` exists; `--return-dir` shipped 2026-07-02). ADR-11/12 say `Accepted (2026-07-05)`. JOURNAL 2026-07-05 already records the divergence "for a future hygiene pass" |
| CLAUDE.md §11 ADR range | **CONFIRMED** | §11 local list stops at ADR-08; ADR-09..12 absent. JOURNAL names it as deferred staleness (edits there trip the canonical-freshness gate — correctly so) |
| VISION:25 dual-output framing | **CONFIRMED** | The "Output paths (dual)" scope line still frames always-on dual-write; ADR-07 is `Superseded by ADR-43` (opt-in `target-project` routing). Flagged at C5b (2026-07-05), deliberately unchanged |
| CONTRIBUTING stamp age | **CONFIRMED** | `last_reviewed: 2026-06-02` — 34 days at session date, past the freshness gate's 30-day A1 cadence |
| Enforcement question empirically open | **CONFIRMED → now answered** | Full organ inventory in §3 Q1, including two classified hard-gate arcs and one advisory firing witnessed live in this very session |
| pytest flake watch (1 occurrence) | **CONFIRMED** | `test_inbox_exits_3_when_any_batch_run_degraded` flaked once on the 2026-07-05 full run (passes in isolation and on re-run; ordering-dependent). Recorded only in JOURNAL prose — no standing watch log exists anywhere |
| Commits unpushed on main | **CONFIRMED** | 10 commits (`2593075..5c81e71`) on local `main`, unpushed; `origin/main` at `75006db` |

### 1.2 Deltas found at Step 0 (reported, not silently absorbed)

1. **Stale worktree from unpushed main (operator-endorsed fix applied).** This lane's worktree was cut from `origin/main` (`75006db`) — 10 commits behind local `main` — and therefore lacked ADR-11/12, `COUNCIL_INVOCATION_CONTRACT.md`, and the fleet-recon audit: three of the lane's four sources of truth. Fixed by a pure fast-forward of the worktree branch to `5c81e71` (no local commits existed; nothing lost). **Governance significance:** the unpushed-main condition stopped being hypothetical debt and produced a witnessed divergence cost inside a governance-design session. This is direct Q1/pause-fork evidence and seeds watch W2 (§3 Q5).
2. **"GOV-1" is not a repo artifact.** Confirmed by the operator at checkpoint: GOV-1 = the currency-pass list = the four ratification follow-ups in the table above. §3 Q4 confirms/adjusts that list.

---

## 2. Functional target state

**One line:** every record-discipline rule in this repo is either held by a witnessed-firing mechanism or is an explicitly named, watched manual rule with a promotion path — so the feature pause becomes a decision on evidence, not caution.

What the operator sees and lives through when the lane is done, by moment:

- **At session start** (unchanged, now named as a contract): the floor file is presence-and-hash-verified and the commit gates are re-armed automatically. An operator can trust that a session which started is a session whose gates are live — armed-state is part of the mechanism, not an installation footnote.
- **At commit time:** editing a canonical doc without a genuine re-review blocks the commit (canonical_freshness A2 — the C5b pattern is the designed norm, not an incident). Dated-log headers normalize themselves. The floor cannot be edited out of sync with its hash.
- **At session end:** the ADR-85 hard gate refuses to let a session that landed commits wrap without a JOURNAL entry naming its SHAs; advisories (BACKLOG marker, dirty tree, cadence) surface exactly once and never loop. The only exit besides compliance is `/override` — logged, HEAD-bound, and zero uses to date.
- **At ratification** (new — DRAFT-GOV-1): an ADR's status means one thing, lives in exactly two places (file header + index) that change in the same commit, and `Proposed` is a state that *expires* — an implemented, load-bearing ADR gets flipped to `Accepted` at the next currency pass instead of drifting indefinitely. Revision-in-place vs supersession has a bright line.
- **On a recurring cadence** (new — DRAFT-GOV-2): a currency pass with a fixed trigger, a fixed checklist, and a fixed record shape sweeps canonical docs and ADR statuses, and closes by pushing `main`. Currency work stops being ad-hoc heroics performed when someone notices.
- **When something flakes or smells** (new — DRAFT-GOV-2): the observation lands in a standing Watches block with an n=2 promotion criterion, instead of evaporating into JOURNAL prose. Second occurrence → tracked task; sustained silence → retirement.
- **Failure semantics throughout:** hard gates are deterministic, this-session-repairable, fail-open on their own errors and fail-closed on real non-compliance; every hard gate ships with a logged escape hatch. Advisories are structurally loop-safe. Nothing in this lane adds LLM judgment to enforcement (ADR-74 upheld).

A non-implementer validates this state by asking, for any record rule: *"what fires if this is violated — and if nothing fires, which watch or manual-rule row names it?"* Every rule must have an answer in the §3 Q1 table.

---

## 3. Design answers

### Q1 — Enforcement inventory (done first, per charter)

**Organ → where it lives → witnessed firing → classification.**

| # | Organ | Where it lives | Witnessed firing | Classification |
|---|---|---|---|---|
| 1 | `canonical_freshness` A2 gate | Repo `scripts/canonical_freshness_gate.py`, deployed **verbatim by the hub enforcement-mesh carrier** (#236; deploys `bda4fff` v1.1.0, `31e785d` v1.2.0); wired `always_run` in repo pre-commit | **HARD BLOCK, true positive — C5b (2026-07-05):** blocked the commit after C5 edited VISION without re-stamping; the forced genuine re-review caught real staleness (deprecated o4-mini/o3 research-model names). SHA `10dd355` is the compliance commit | Portable Group-A organ; commit-time hard gate; **value-adding block witnessed** |
| 2 | `session_end_backpressure.py` (seb) — ADR-85 hard leg (JOURNAL SHA anchor) | Repo `.claude/settings.json` Stop hook (hub-local concept per Q2 ruling 2026-06-07 — deliberately NOT in the fleet plugin) | **Fired at every 2026-07-05 session end; satisfied by compliance:** both JOURNAL entries name their session SHAs (`2593075`; the five ratification SHAs). Zero `/override` tokens ever written (`logs/.session-override-token` absent) | Session-end hard gate working as **deterrence** — no block needed because the record was written to satisfy it. Block-until-compliant; fail-open on own errors |
| 3 | seb — advisory legs (BACKLOG marker, dirty tree, canonical cadence) | Same hook; fire-once + structural floor (block-cap loop-safety designed in) | **Witnessed live in THIS session (2026-07-06):** BACKLOG-marker advisory fired once at the checkpoint stop, correctly self-exempted ("pure advance that finishes nothing"), did not loop | Advisory backpressure; loop-safe design confirmed in the field |
| 4 | `floor-hash-verify` | Repo pre-commit (local hook → `.claude/check_floor_hash.py`) | Green at all merges; no block recorded (no tamper has occurred) | Integrity hard gate for the methodology floor |
| 5 | Floor presence guard + gate arming | Repo `.claude/settings.json` SessionStart: `check_floor_hash.py --require-present` + `python -m pre_commit install` | Fires every session start | **The arming step is itself an organ** — gates are dead weight uninstalled; this makes armed-state automatic |
| 6 | `normalize-headers` | Repo pre-commit (local) | Green at every JOURNAL/LESSONS commit | Mechanical normalizer (hygiene, not a gate) |
| 7 | `toc-freshness` / `toc-generate` | **Hub rev-pinned** pre-commit (`repo: ../.dev-knowledge`, `rev: v1.2.0`) | Verified fail-stale/pass-fresh at install (2026-06-03); green on the 2026-07-05 GUIDE edit | Rev-pinned-pull pattern — the consumer knows exactly which hook version it runs |
| 8 | ruff pre-commit gate | **PRUNED 2026-07-04** ([#244] P2 n=1 remove-leg; deploy `31e785d`) — then **RE-ACTIVATED 2026-07-12** (fleet ruling overriding the prune; `astral-sh/ruff-pre-commit` mirror `v0.15.5`, gate mode, prune-safe bare `id: ruff`; declared in `.methodology.yaml` `ruff-gate`) | RE-ACTIVATED 2026-07-12 — probe F401 block witnessed | Precedent: prunes **and** re-activations are first-class, evidence-based deploys (this row amended in place 2026-07-12, prune fact preserved) |
| 9 | tier1-lifecycle plugin (Stop→`propose_closures`, SessionStart→surface) | Repo `.claude/settings.json` enabledPlugins → hub marketplace (fleet-distributed) | Fires each session (closure proposals surface at start) | Advisory closure loop (ADR-70 Tier-1) |
| 10 | `/override` command | Repo `.claude/commands/override.md` | **Never used** (no token file has ever existed) | The mandatory escape hatch of organ #2 — logged, HEAD-bound |
| 11 | `block-onedrive.ps1` | **User `~/.claude`** PreToolUse (all repos) | Standing; no violation attempted from this repo | Fleet safety hard gate (mesh member, out of repo scope) |
| 12 | `surface-closures.ps1` | User `~/.claude` SessionStart | Fires each session | Fleet advisory (closure-loop surface) |
| 13 | `claude-notify.ps1` | User `~/.claude` Stop/Notification | Fires each session | **Notification only — NOT enforcement** (listed to keep the inventory honest) |
| 14 | `audit.py` 10-check machine floor | **Hub-only**, run ad-hoc | Witnessed 2026-06-02 G1 audit: 9/10 → 10/10 pass | Hub-side audit leg; shares the `canonical_freshness` module with organ #1 |
| 15 | `check.ps1` (pytest + mypy + ruff) | Repo `scripts/`, **MANUAL** pre-merge rule (CLAUDE.md §5.3) | Run at merges per JOURNAL (426 green 2026-07-05) | Manual rule — held by compliance, no mechanism |
| 16 | branch → merge `--no-ff` discipline | **Rule only** (floor + user core-invariants); no mechanism | Held by compliance; one authorized exception witnessed (2026-07-05 recon close-out direct-to-main per explicit operator instruction, recorded in JOURNAL) | Manual rule with a clean paper trail — but unmechanized |
| 17 | Push cadence for `main` | **NOTHING owns it** | **Witnessed cost 2026-07-06:** this lane's worktree cut from `origin/main`, 10 commits stale, missing 3 of 4 sources of truth (§1.2) | The inventory's one genuinely ownerless rule → watch W2 + consumer requirement R8 |
| 18 | ADR status/header ↔ index sync; `Proposed` expiry | Manual, no mechanism | The ADR-09/10 divergence is the witnessed failure mode (drifted since 2026-07-02) | Gap → DRAFT-GOV-1 |
| 19 | LESSONS append-only (ADR-29); JOURNAL 3-line schema | Manual + organ #6 partial (headers only) | Held by compliance | Manual rule, low drift risk, watched via currency pass |

**Classification of the two charter-named firings:** the canonical-freshness firing at C5b (`10dd355`) is a **commit-time hard gate, true positive, value-adding** — the forced re-review found staleness a human sweep had missed. The 2026-07-05 session-end firings are **two distinct organs** — the seb ADR-85 hard leg (deterministic, hub-local, satisfied by compliance: the JOURNAL entries naming SHAs *are* its output) and the plugin's advisory closure proposer. Neither blocked, because both shaped behavior upstream — which is enforcement succeeding, not enforcement absent.

**Done-contract: the feature-pause rationale is now re-evaluable on evidence.** Coverage map:

- **Held by mechanism (witnessed):** commit-time canonical-doc freshness · session-end JOURNAL SHA anchoring · floor integrity + presence · TOC freshness · header normalization · gate arming · OneDrive exclusion.
- **Held by manual rule (named, unmechanized):** branch/`--no-ff` discipline · pre-merge `check.ps1` · lint (post-prune) · ADR status hygiene (§Q2 closes the convention gap) · LESSONS/JOURNAL schema · BACKLOG reflection (advisory only, by design).
- **Ownerless until this design:** push cadence (→ W2 + R8).

The enforcement question is no longer empirically open: the mesh demonstrably fires, blocks correctly, and has never needed its escape hatch. The remaining gaps are enumerable manual rules, each named above. The pause *ruling* on this evidence stays with the operator (§5, §6).

### Q2 — ADR lifecycle states

Designed as DRAFT-GOV-1 (§4). Functional summary: four states — `Proposed` (ratified shape, not yet load-bearing; **expires**: once implemented and survived one review cycle it must flip at the next currency pass), `Accepted` (ratified + load-bearing; the flip is an operator ruling recorded in a JOURNAL entry naming the SHA — the witnessed ADR-11/12 pattern is codified as *the* pattern), `Revised (dated)` (a dated amendment **appended** to the same file, original decision text intact — legal only for parameter-level changes that do not invert the decision; reconciles the existing `Revised` statuses on ADR-01/02/06 with the "ADRs are immutable" rule in CLAUDE.md §5.5), and `Superseded (by X)` (decision inverted or replaced; new ADR required; ADR-07 → ADR-43 is the conforming precedent). Status lives in exactly two places — file header and `docs/decisions/README.md` index row — and both change **in the same commit** (the sync rule the ADR-09/10 divergence proved necessary). Pre-number discipline: designs carry `DRAFT-<lane>-<n>`; the operator assigns real numbers at ratification (ADR-13 stays reserved by the audit's crux-resolver draft). **Disposition of ADR-09/10:** flip both to `Accepted` at GOV-1 execution — both are implemented, load-bearing, and have survived multiple review cycles; the flip needs the operator's ratification ruling, not new evidence.

### Q3 — Consumer requirements for the hub enforcement-mesh carrier

Requirements INPUT to the hub-level carrier arc, from ai-council's lived experience (the JOURNAL-gap arc that produced ADR-85, the C5b block, the [#244] prune, this session's stale worktree). Stated as WHAT a consumer repo needs — never HOW the hub implements it (seam §C.5; no lane proposes installing hub hooks). Full text in §7; headline requirements:

- **R1 Commit-time freshness organ, deployed verbatim and rev-traceable** — the proven pattern (deploy commits are first-class, named in the consumer's history).
- **R2 Session-end record gate** — deterministic, block-until-compliant, fail-open on own errors, and it **must ship with its logged, HEAD-bound override**.
- **R3 Automatic arming** — the carrier must guarantee armed-state at session start (and ideally verify it), not just file presence; an unarmed gate is worse than none because it reads as coverage.
- **R4 Version-pinned hooks + a documented deploy/prune protocol** — consumers must know which mesh version they run; prunes follow the [#244] evidence pattern (n-based, recorded, reversible).
- **R5 Hard-gate qualities** — deterministic (no LLM judgment, ADR-74), this-session-repairable, clean-tree-scoped (no mid-work nagging).
- **R6 Advisory loop-safety** — no advisory may persist into the Stop-hook block-cap auto-override (the "persistence beats policy" bypass ADR-85 forbids); fire-once + structural floor is the proven shape.
- **R7 Organ integrity self-guard** — hash sidecar + presence check; an organ that can be silently edited or deleted is not enforcement.
- **R8 (new, from this session's evidence) Push-cadence surface** — an advisory naming unpushed-`main` age/commit-count at session end, because the witnessed failure mode (stale worktrees, divergent parallel sessions) is a fleet problem, not an ai-council quirk. A branch-discipline verify (`git log --first-parent main` shows only merges) is the companion candidate.
- **Stays hub-side:** the audit.py machine floor, carrier deploy machinery, plugin distribution, ADR-74/78/85 policy evolution.

### Q4 — Currency-pass scope & recurring process

**GOV-1 list confirmed** (operator, checkpoint) and **adjusted** against live state:

1. ADR-09/10 → `Accepted` — **adjusted:** header AND index rows flip in the same commit (the DRAFT-GOV-1 sync rule applied at first execution).
2. CLAUDE.md §11 — extend the local ADR list through ADR-12 (edit will trip canonical_freshness; genuine re-review + re-stamp is the designed cost).
3. VISION:25 — reconcile the dual-output scope line to the ADR-43/ADR-10 reality (opt-in `target-project` + return-dir; local default, hub never).
4. CONTRIBUTING — genuine re-read + re-stamp (34 days; A1 territory).
5. **Added — pass close-out: push `main`.** Deciding evidence is §1.2: unpushed main already cost a session its sources of truth. A currency pass that leaves the record unpublished has not finished making the record current.

**Recurring process** (DRAFT-GOV-2 part A): **Triggers** — T1: any ratification/supersession merge queues a pass (the 2026-07-05 follow-ups are exactly what T1 would have caught); T2: 30-day backstop aligned to the gate's A1 cadence; T3: any A1 WARN surfacing in a session. **Scope** — fixed checklist: canonical docs (VISION, ARCHITECTURE, CLAUDE.md, CONTRIBUTING) + ADR headers ↔ index sync + `Proposed`-expiry review + push state. **Gate interaction** — the pass *deliberately* trips A2 and satisfies it with genuine re-reviews; re-stamps without re-reading are forbidden (both 2026-07-05 genuine re-reviews caught real staleness — the practice pays). **Record** — one `docs/` branch, `--no-ff` merge, JOURNAL entry naming SHAs; status flips require the operator's ratification ruling in-session. **Executor** — any CC session; the process is designed to be commissioned as a routine, not remembered as a habit.

### Q5 — Watch protocol

(DRAFT-GOV-2 part B.) **Where:** a standing `## Watches` block in `BACKLOG.md` — the living work file, already carrying non-story sections ("About this file", grooming log), so the ADR-66 story-map is not violated; JOURNAL is append-only prose (wrong shape for state), and a new top-level file fights the ADR-53/60 file discipline. **Entry schema:** `W<n> · what · first occurrence (date + SHA/evidence) · promotion criterion · retire criterion`. **Promotion:** second occurrence → a real BACKLOG task (or a gotchas entry if the pattern is cross-repo) — mirroring the [#244] n-discipline and the global gotchas auto-promote rule. **Retirement:** zero recurrence by the next quarterly grooming (or 90 days) → strike the watch with a grooming-log line. **Seed watches:**

- **W1 — pytest flake:** `test_inbox_exits_3_when_any_batch_run_degraded`, n=1 (2026-07-05 full run; passes isolated + on re-run; ordering-dependent). Promotion: second flake → Epic C task to pin the ordering dependency.
- **W2 — unpushed-main age:** n=1 witnessed cost (2026-07-06 stale worktree, §1.2). Promotion: second divergence incident → mechanize (consumer requirement R8 escalates from advisory-wanted to task).
- **W3 — seb block-cap auto-override:** n=0; structural watch on the one designed bypass of ADR-85. Promotion: any single occurrence → immediate task (this is a safety property, so n=1 promotes).

---

## 4. Draft ADRs

### DRAFT-GOV-1 — ADR lifecycle states for ai-council

**Status: DRAFT** (mini-ADR; operator assigns the real number at ratification — ADR-13 remains reserved by the audit's crux-resolver draft)

- **States:** `Proposed` → `Accepted` → (`Revised (dated)`)* → `Superseded (by ADR-X)`.
- **Proposed** = shape ratified for authoring, not yet load-bearing. **Expiry rule:** once the decision is implemented and has survived one review cycle, the status MUST flip at the next currency pass — `Proposed` is a waiting room, not a resting state. (Witnessed failure mode: ADR-09/10, `Proposed` since 2026-07-02 while long since load-bearing.)
- **Accepted** = operator ratification ruling, recorded in a JOURNAL entry naming the ratification SHA (codifies the witnessed ADR-11/12 pattern).
- **Revised (dated)** = a dated amendment **appended** to the same file; the original decision text stays intact. Legal only for parameter-level changes that do not invert the decision (panel size, defaults, cost tables). This reconciles the live `Revised` statuses (ADR-01/02/06) with CLAUDE.md §5.5 "ADRs are immutable — supersede, never edit": *immutable* means the decision record is never rewritten; it does not forbid dated, append-only amendments. Grandfathered: ADR-01/02/06 conform as-is.
- **Superseded (by X)** = the decision is inverted or replaced; requires a new ADR; the old file gains only the status line + pointer (ADR-07 → ADR-43 is the conforming precedent).
- **Sync invariant:** status lives in exactly two places — the ADR file header and its `docs/decisions/README.md` index row — and any status change touches **both in the same commit**. (Testable; a future mechanization candidate, not mechanized by this ADR.)
- **Pre-number discipline:** unratified designs carry `DRAFT-<lane>-<n>` inside audit documents; numbers are assigned by the operator at ratification only.
- **Immediate disposition on ratification of this draft:** flip ADR-09 and ADR-10 to `Accepted` at GOV-1 execution (operator ruling; no new evidence needed).

### DRAFT-GOV-2 — Recurring currency pass + watch protocol

**Status: DRAFT**

- **Currency pass — triggers:** T1 any ratification/supersession merge queues a pass · T2 30-day backstop (aligned to canonical_freshness A1) · T3 any A1 WARN observed in a session.
- **Scope (fixed checklist):** canonical docs (VISION, ARCHITECTURE, CLAUDE.md, CONTRIBUTING) genuine re-read + re-stamp · ADR header ↔ index sync sweep · `Proposed`-expiry review (per DRAFT-GOV-1) · unpushed-`main` state; **pass close-out = push `main`**.
- **Gate interaction:** the pass deliberately trips the A2 gate and satisfies it with genuine re-reviews; a re-stamp without a re-read is a protocol violation (evidence the genuine practice pays: both 2026-07-05 re-reviews caught real staleness).
- **Record:** one `docs/` branch → `--no-ff` merge → JOURNAL entry naming SHAs; status flips carry the operator's in-session ratification ruling.
- **Watch protocol:** standing `## Watches` block in BACKLOG.md; schema `W<n> · what · first occurrence (date + evidence) · promotion criterion · retire criterion`; promotion at n=2 to a tracked task (n=1 for safety properties, e.g. W3); retirement at quarterly grooming after zero recurrence. Seed watches W1 (pytest flake), W2 (unpushed-main), W3 (seb block-cap) per §3 Q5.
- **Explicitly out of scope:** mechanizing any of this (hub carrier arc owns mechanisms — seam §C.5); editing hub files; ADR numbering.

---

## 5. Refined fork — the feature-work pause (this lane owns its deciding evidence)

**The fork:** the feature-work pause was declared while the enforcement question was empirically open. The §3 Q1 inventory is the deciding evidence the pause was waiting for.

**Bounded options:**
- **O1 — lift now.** The mesh demonstrably fires and blocks; manual-rule gaps have held for weeks.
- **O2 — lift after GOV-1 executes (including the push of main).** Same evidence, plus the record is actually current and published before feature diffs start landing on top of it; cost is one short docs session.
- **O3 — hold until the hub carrier lands.** Treats the enumerable manual rules (push cadence, branch discipline, `check.ps1`) as blocking; highest safety, indefinite timeline (hub-side arc, not this repo's to schedule).

**Deciding evidence, as witnessed:** two hard-gate arcs held under real pressure (C5b block → genuine review value; ADR-85 compliance with zero overrides); the advisory layer proved loop-safe in the field (this session); the one ownerless rule (push cadence) already exacted its cost (§1.2) and is now named, watched (W2), and forwarded as a carrier requirement (R8). Nothing in the inventory shows enforcement *absent* — it shows a working mesh plus an enumerated, watched manual perimeter.

**Recommendation (not a ruling):** O2. **The ruling stays with the operator.**

---

## 6. Questions for the operator (genuine forks only)

1. **Pause disposition** — O1 / O2 / O3 per §5. *Recommendation: O2 — lift after GOV-1 executes and `main` is pushed.*
2. **DRAFT-GOV-1 ratification + the ADR-09/10 flip** — ratify the lifecycle convention and authorize the flip at GOV-1 execution? *Recommendation: yes; the flip is overdue by the convention's own expiry rule and needs only your ruling.*
3. **Watch location** — BACKLOG `## Watches` block (recommended) vs a separate file vs JOURNAL prose (status quo)? *Recommendation: BACKLOG — living file, survives the story-map schema, zero new files.*
4. **Push cadence interim rule** — until the hub carrier ships R8, adopt "push `main` at every currency-pass close-out and after any ratification merge" as a manual rule? *Recommendation: yes — it is the minimum that would have prevented §1.2, at zero mechanism cost.*

---

## 7. Inputs forward for the technical architect (requirements & constraints — NOT backlog items)

**Consumer requirements for the hub enforcement-mesh carrier** (ai-council as consumer; hub designs mechanisms):
- R1 commit-time freshness organ, deployed verbatim, rev-traceable deploys · R2 session-end record gate with mandatory logged HEAD-bound override · R3 automatic arming + armed-state verification · R4 version-pinned hooks + documented deploy/prune protocol (n-evidence discipline) · R5 hard gates deterministic / this-session-repairable / clean-tree-scoped · R6 advisory loop-safety (no path to the block-cap bypass) · R7 organ integrity self-guard (hash + presence) · R8 push-cadence advisory + branch-discipline verify (new; evidence §1.2).

**Testable invariants this design creates:**
- ADR status header ↔ index row always agree (single-commit sync rule).
- No ADR sits `Proposed` while implemented past one review cycle.
- Every currency pass ends with `origin/main == main`.
- Every hard gate in the repo has a named, logged escape hatch; every advisory is fire-once-safe.
- Every record rule resolves in the Q1 inventory to a mechanism, a named manual rule, or a watch.

**Constraints binding any implementation:** no hub edits from this repo; no LLM judgment in gates (ADR-74); no new top-level files for watches; ADR numbering is operator-only; the seam §C.5 boundary — this repo states requirements, the hub carrier arc designs the mechanism; the two `DRAFT-GOV-*` texts settle shape only, and nothing here is an epic or backlog item.

---

**Done-contract check:** charter Q1–Q5 each answered explicitly ✔ · entry-state claims all verified live, two deltas reported at checkpoint, both operator-ruled ✔ · draft ADRs carry `DRAFT-GOV-n`, no real numbers claimed ✔ · pause fork refined with deciding evidence + recommendation, ruling left open ✔ · zero writes outside this document ✔ · functional level held (no code, no signatures, no file diffs) ✔
