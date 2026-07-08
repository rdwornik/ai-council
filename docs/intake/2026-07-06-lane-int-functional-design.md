# L-INT — Delegation Interface: Functional Design (pillar 4, "the window on the world")

**Date:** 2026-07-06 · **Lane:** L-INT (`int`) · **Mode:** DESIGN — functional architecture only; zero source/config changes; drafts stay drafts (no ADR numbers claimed)
**Author:** Fable 5 (CC), one of five concurrent lane sessions · **Frame:** `CC-MASTER-lane-functional-design-sessions.md`
**Sources verified live at `5c81e71`:** ADR-11/12 (Accepted), `protocols/COUNCIL_INVOCATION_CONTRACT.md`, fleet-recon audit 2026-07-05 (P3, F6–F8), Fable audit 2026-07-04 (D1/D2/D8), GUIDE, BACKLOG.

---

## 1. Lane charter & current state (verified live)

**Charter:** the caller's complete experience — what a foreign repo's CC session lives through commissioning the Council, end-to-end, including every failure path.

**Entry-state verification (Step 0):**

| Claim | Verdict | Evidence |
|---|---|---|
| ADR-11 Accepted | **CONFIRMED** | `docs/decisions/ADR-11-delegated-invocation-contract.md` — `Status: Accepted (ratified 2026-07-05 …)` |
| CONTRACT live, Known-deviations names the two D2 gaps | **CONFIRMED** | `protocols/COUNCIL_INVOCATION_CONTRACT.md` §7: (1) `--file` does not parse frontmatter; (2) research ignores `--return-dir` — both marked "code has not shipped yet" |
| No real foreign commission has run yet | **CONFIRMED (with caveat)** | No commission artifacts tracked in-repo; no JOURNAL entry records one. Caveat: an untracked `output/` in the primary checkout is not visible from this design worktree — conclusion rests on the tracked record |

**Delta flagged (checkpoint-reported):** this worktree branch was created stale at `75006db` (pre-ratification — no ADR-11/12, no CONTRACT, no recon audit). Fast-forwarded to main `5c81e71` before any design work; all verification above ran against the ratified state.

**One material current-state finding beyond the charter's list:** a *debate-seat* dropout that leaves ≥ the minimum panel exits **0**, not 3 (CONTRACT §6; exit 3 is research-threshold and inbox-batch only, per ADR-08). A caller that trusts the exit code alone can record a full-panel verdict that was actually produced by a shrunk panel. ADR-08 semantics are settled (§B — not reopened); the gap is closed at the package layer, not the exit-code layer (see Q2/Q3, DRAFT-INT-1).

## 2. Functional target state

> **One line:** a foreign repo's CC session commissions the Council with one command and — without reading transcripts — walks away with a recordable verdict package, knowing on every exit path exactly what it observed, what it must do next, and what it must record.

**The journey, as observable behavior.** A caller repo hits an ADR-67 convene threshold (hub-owned judgment). Its CC session authors a GUIDE-conformant brief, optionally runs the L-DOC doctor pre-flight (consuming only the §C.2 seam: one command, machine-readable GREEN/YELLOW/RED + per-seat detail), then invokes Lane A from its own cwd:

```
council --file <brief.md> --return-dir <caller>/docs/decisions/inbox [--format json] [flags]
```

The caller blocks. When the process exits, the caller reads **two signals, always both**: the exit code (coarse: 0/1/2/3) and the **verdict package** at the return dir (fine: what actually happened inside the run — panel seated vs requested, verdict author, dissent, degradation detail). The exit code decides *whether* there is a usable verdict; the package decides *what the caller must record about it*. The caller then drafts its ADR in its own repo from the package (never from raw transcripts, never copying artifacts to the hub), commits, done.

**Failure semantics from the caller's chair** (what it observes / must do / must record):

| Path | Caller observes | Caller must do | Caller must record |
|---|---|---|---|
| **Success (0)** | Exit 0; verdict package + artifacts at return dir | Consume package; draft ADR | Provenance block (§Q4); panel-degradation field even on 0 (may be non-empty) |
| **Hard fail (1)** | Exit 1; no artifacts guaranteed | Retry once (transient causes: network, health-gate flake); then fall back to Lane B (drop brief in `council_inbox/`, notify operator) | The failed attempt (timestamp, error line) in its session record — not in an ADR (no verdict exists) |
| **Usage error (2)** | Exit 2 + Click message | Fix the invocation; this is a caller bug, not a council event | Nothing durable |
| **Degraded-but-complete (3)** | Exit 3; verdict package present; alarm content persisted in package (requirement R3, §7) | Consume verdict; apply the exit-3 obligations spec (§Q3) | The full Degradation Record block (§Q3) in the derived ADR |
| **RoutingError** | Fail-loud abort; allow-list printed | Drop or fix `--target-project` (foreign callers rarely need it — return-dir suffices; use only for explicit transcript mirroring) | Nothing durable |
| **Timeout (caller-side)** | No exit within the caller's wall-clock budget | Kill; treat as hard fail (retry once → Lane B). Budgets set from the contract's duration envelope (requirement R5, §7): standard debate = minutes-class; `--deep` research = hour-class | The abandoned attempt in its session record |
| **Empty return dir on exit 0** | Exit 0 but no artifacts at `--return-dir` | Check canonical `./output/` in the ai-council repo (known path; covers deviation #2 for research until D2 lands); if artifacts exist there, consume + record the deviation; if nowhere, treat as hard fail | Where the artifacts were actually found, in the ADR's provenance block |

**Boundaries held throughout (ADR-10/11, restated as observed behavior):** the caller never writes into ai-council; ai-council writes into the caller only at the declared return dir (plus explicit `--target-project` mirrors); the hub is never a destination; verdict artifacts are *referenced* from caller ADRs by path, never copied hub-side.

A non-implementer validates this target state by running one commission per row of the table and checking that the observed behavior, the required caller action, and the recorded output match the row.

## 3. Design answers (charter Q1–Q7)

### Q1 — Caller journey

Answered in full by §2 (happy path + six failure paths, each with observe/do/record). Two design rulings embedded there, stated explicitly:

1. **Two-signal rule.** The exit code is never sufficient alone. Exit 0 can hide a shrunk panel (§1 finding); exit 3 tells the caller degradation happened but not what. The verdict package (Q2) is the authoritative fine-grained signal; the contract should oblige callers to read both. This closes the exit-0 under-reporting gap without touching ADR-08 (settled).
2. **Fallback ladder is fixed:** retry-once → Lane B → operator. Never retry in a loop (a systematically failing council should surface to the operator, not burn spend), and never "fall forward" by weakening the question or panel to make the run pass.

### Q2 — Minimal verdict package

**Ruling: NOT sufficient as-is.** council-out + minority + metrics fails the "draft an ADR without reading transcripts" test on four gaps, all verified against the live output writer:

| # | Gap | Evidence |
|---|---|---|
| G1 | Verdict and transcript share one artifact — the synthesis is a section *inside* `council-out-*.md` (`## Synthesis (by …)` heading); a caller must parse prose in a mixed file to find the ruling | `output.py` save path: single file carries rounds + synthesis |
| G2 | No structured decision layer: the decision, its rationale, and considered-and-rejected options exist only as synthesis prose; `--format json` (DebateResult) carries the synthesis as one text blob — and stdout is lost unless the caller captured it | CONTRACT §5 |
| G3 | Degradation detail is console-only: the exit-3 alarm banner and per-seat dropout causes are not persisted into any artifact; the caller's §Q3 obligation is currently unfulfillable from artifacts alone | CONTRACT §4/§6; no alarm text in the artifact set |
| G4 | Panel-degradation on exit 0 is recorded nowhere caller-visible (metrics show per-provider cost rows, but requested-vs-seated is not first-class; the `seats[]` design that will carry it is L-CLI's, and a caller should not have to diff a cost sidecar against `settings.yaml` to learn a seat dropped) | §1 finding |

**The missing element, specified functionally: a `council-verdict-*.json` summary artifact** (deterministic sibling name in every destination, additive to the committed §5 set) plus a human-readable mirror block at the top of `council-out-*.md`. Content spec is DRAFT-INT-1 (§4). Seam discipline: the package *consumes* identity facts from L-EPI's `synthesis` metrics namespace (intended vs actual verdict author) and L-CLI's `seats[]` namespace (requested vs actual backend/model) by reference — it defines caller-facing fields sourced from those namespaces and designs neither.

### Q3 — Exit-3 caller obligations, as a checkable spec

The CONTRACT's sentence ("the caller MUST record the degradation … in whatever ADR/decision it derives") becomes: **the derived ADR must contain a `## Degradation record` section with all five fields below; a reviewer (or a future gate) checks presence and non-emptiness of each.**

1. **Run reference** — verdict-package `run_id` + timestamp + exit code.
2. **What degraded** — the seats/providers that failed or fell short, with the classified cause per seat (from the package, which persists the alarm content — requirement R3).
3. **Threshold vs actual** — the configured minimum (e.g. `min_successful_providers`) against what was achieved.
4. **Bearing on the ruling** — one of three declared values: `verdict-robust` (the degraded seats' absence plausibly doesn't change the outcome — say why), `verdict-sensitive` (it might — say how the caller compensated or why it accepted the risk), or `unassessed` (explicitly allowed, but must be stated, never implied).
5. **Disposition** — accepted as-is / re-run commissioned / escalated to operator.

**Extension beyond exit 3 (from the §1 finding):** the same section is mandatory whenever the verdict package's panel-degradation field is non-empty — *even on exit 0*. Checkable form: "Degradation record present ⇔ (exit code 3 ∨ package.degradation ≠ ∅)"; absent otherwise (a "none" section on clean runs is noise, not discipline).

### Q4 — Verdict→ADR transform (F7): template content, designed now

Section-keyed template content (the artifact itself lives hub-side; adoption stays gated on F7's evidence — this settles only *what the sections are and where each sources from*):

| Section | Sourced from |
|---|---|
| Title / Status / Date | Caller |
| Context — the decision faced, constraints, why the Council was convened | The caller's brief (question + constraints), referenced by path |
| Council verdict — the ruling, restated | Package `decision` + `rationale` (G2 fields); NOT re-derived from transcript prose |
| Considered and rejected | Package per-option summary where present; else the synthesis section of council-out (annotated as prose-sourced) |
| Dissent | Package `dissent` field: `unanimous`, or pointer to `council-minority-*` + a one-paragraph caller summary of the minority position. Annotation: this field inherits the D13 heading-heuristic quality until Epic B hardens dissent detection — the template carries the caveat verbatim while that holds |
| Degradation record | §Q3 spec (conditional presence rule) |
| Provenance | Package: run_id, contract version consumed (Q7), mode, panel seated, verdict author (actual, via L-EPI's namespace), artifact paths (referenced, not copied) |

Degradation annotations thread through: any prose-sourced or heuristic-sourced section carries its annotation inline, so a reader of the caller ADR can see which claims rest on structured fields vs parsed prose.

### Q5 — Batch/multi-question commissions: **explicitly OUT of scope**

**Boundary:** the Lane A contract commits to single-question commissions — one brief, one decision, one verdict package. Multi-question briefs and council-side batch orchestration are out.

**Rationale:** (a) one-brief-one-decision is the GUIDE's and ADR-67 loop's existing discipline; a multi-question brief is breadth-sprawl (an L-EPI-catalogued bias) smuggled in through the interface. (b) Exit-code and degradation semantics become ambiguous over N questions (which one degraded?) — ADR-08 is settled and single-run-shaped. (c) Batching needs no council-side concept: a caller wanting N verdicts runs N Lane A invocations (a caller-side loop it fully controls), or drops N briefs into Lane B for an operator-mediated batch — which is exactly what Lane B already is. If real demand for council-side batching ever appears, it arrives as evidence from repeated caller-side loops and gets its own ADR; nothing here forecloses it.

### Q6 — Lane B positioning: **keep both, permanently differentiated — no deprecation path**

Lane B's long-term role once Lane A is fixed:

1. **The operator lane** — human-authored questions without a CC session in the loop (Downloads-drop detection is an operator convenience Lane A will never replicate).
2. **The structural fallback** — CONTRACT §4 already names it as the exit-1 fallback; §2's fallback ladder depends on it. Deprecating Lane B would leave Lane A's hard-fail path dangling.
3. **The batch lane** — per Q5, operator-mediated N-brief batches are Lane B's job by design.

What Lane B is *not*: the default for foreign-repo agents (that's F6 — recommendation: Lane A default, evidence still to be gathered) and never a context-pull/interactive surface (ADR-11 §1 cut, upheld). Anti-pattern watch inherited from repo memory: features added to the interactive/Lane-A path must be explicitly mirrored into the inbox loop — the two-lane commitment makes lane parity a standing contract obligation, not a courtesy (this is the D2 deviation class, generalized; forwarded as requirement R6).

### Q7 — Versioning stance (F8): pre-designed version line + breaking-change criterion

**Version line:** the CONTRACT header gains an explicit `Contract-Version: <MAJOR>.<MINOR>` (today it declares itself "a versioned commitment" but carries no version identifier — verified live). The verdict package echoes the version it was produced under (`contract_version`), so every caller ADR's provenance names the surface it consumed.

**Breaking-change criterion (escalates to an ADR, MAJOR bump):** removing or renaming a flag; changing a flag's meaning or default; changing exit-code semantics or their caller obligations; renaming or removing an artifact from the §5 set; removing or re-typing a verdict-package field; changing the precedence order (flag > frontmatter > default).

**Additive change (MINOR bump + CONTRACT changelog line, no ADR):** new flags, new artifacts, new package fields, new frontmatter keys, documentation clarifications that don't alter behavior.

**Stamping:** `1.0` when the D2 parity fixes land and §7 Known-deviations empties — a `1.0` that ships with known deviations would make the first version a lie. Until then the CONTRACT stays version-line-less exactly as it is (the deviations section *is* the version statement). This is DRAFT-INT-2.

## 4. Draft ADRs

### DRAFT-INT-1 — Verdict Package: a caller-consumable summary artifact
**Status: DRAFT** (baseline-INDEPENDENT — fields source from existing mechanisms; the `dissent` field carries the D13-heuristic annotation until Epic B, but its shape doesn't depend on the Epic B ruling)

- **Decision (shape):** every run additionally emits `council-verdict-<ts>-<mode>-<slug>.json` in every destination (canonical + return-dir + mirrors; additive to the ADR-11 §5 committed set), mirrored by a human-readable summary block at the top of `council-out-*.md`.
- **Fields (functional):** `run_id`, `timestamp`, `contract_version` (Q7), `question` (echo), `mode`, `exit_semantics` (the code the process will return), `decision` (one-line ruling), `rationale` (key bullets), `options_considered[]` (where the synthesis structure yields them), `dissent` (`unanimous` | pointer to minority artifact + gist), `panel` (requested vs seated — sourced from L-CLI's `seats[]` namespace once built; until then from the health-gate/dropout record), `verdict_author` (intended vs actual — sourced from L-EPI's `synthesis` namespace; consumed, not designed), `degradation` (per-seat classified causes + persisted alarm text — closes G3/G4), `artifacts[]` (manifest of everything written, with destinations).
- **Consequences:** the "draft an ADR without reading transcripts" test becomes mechanically passable; the exit-0 shrunk-panel gap becomes caller-visible; the Q3 obligations spec becomes fulfillable from artifacts alone.
- **Seams respected:** consumes L-EPI's and L-CLI's metrics namespaces by reference; designs neither. Not a metrics-sidecar extension — a separate caller-facing artifact (the sidecar is telemetry; this is the deliverable).

### DRAFT-INT-2 — Invocation-contract versioning
**Status: DRAFT**

- **Decision:** `Contract-Version: MAJOR.MINOR` line in the CONTRACT; MAJOR/breaking criterion and MINOR/additive lists per §Q7; breaking changes require an ADR (making ADR-11 §5's commitment mechanical); package echoes `contract_version`; `1.0` stamps when §7 Known-deviations empties (D2 fixes landed).
- **Consequences:** F8 resolves at the shape level now; the first breaking flag change has a rail to run on instead of becoming the precedent-setting improvisation F8 feared.

## 5. Refined forks

| Fork | Refinement | Deciding evidence (sharpened) |
|---|---|---|
| **F6** — foreign-repo default lane | Recommendation stands: **Lane A default**, Lane B = operator/fallback/batch (Q6). | First **3 real commissions**, measuring per commission: (a) operator touches required, (b) wall-clock from convene-decision to recorded caller ADR. Lane A wins if it averages ≤1 operator touch (spend/profile approval only); ≥2 structural touches per run → revisit |
| **F7** — verdict→ADR transform | Template **content settled now** (§Q4 table); only *adoption* stays open. | Audit the first 3 caller-side ADRs against the §Q4 section list: "quality drift" is operationalized as (a) missing mandatory section (Degradation record when required, Dissent, Provenance), or (b) a Council-verdict section that misstates the package `decision`. **≥1 such defect in ≥2 of the first 3 ADRs → adopt the template** (hub-side artifact); 0-defect run of 3 → free-form drafting stands |
| **F8** — contract versioning | **Resolved at shape level** by DRAFT-INT-2 (doc version for additive, ADR for breaking — the fork's two options become tiers, not alternatives). | Residual: only the `1.0` stamping moment — recommendation: at D2-fix landing (Q7). The originally named evidence ("first breaking flag change") now just *exercises* the rail |

## 6. Questions for the operator (genuine forks only)

1. **Verdict-package vehicle** — (a) JSON artifact + human mirror block (DRAFT-INT-1 as written), (b) JSON artifact only, (c) structured header block in council-out only (no new artifact). **Recommendation: (a).** The JSON is the machine truth for agent callers and future gates; the mirror block keeps Lane B/operator reads self-contained. (c) alone fails agent callers (prose-parsing again); (b) alone degrades the operator experience.
2. **Exit-0-with-degradation obligation** (§Q3 extension) — oblige the Degradation-record section on exit-0 runs with a non-empty package degradation field, or keep obligations strictly exit-3-scoped as the CONTRACT reads today? **Recommendation: oblige it.** The shrunk-panel-on-exit-0 case is the one degradation the caller currently cannot see at all; an obligation keyed only to exit codes re-inherits ADR-08's coarseness. (This extends a caller obligation — an additive contract change under DRAFT-INT-2's own criterion, but it changes what callers must do, so it deserves an explicit ruling.)

## 7. Inputs forward for the technical architect (requirements/contracts/constraints — NOT backlog)

- **R1 — D2 parity fixes remain the precondition** for any real Lane A commission (frontmatter parse on `--file`; research `return_dir`) — already ADR-11 decisions; sequencing input only.
- **R2 — Verdict package per DRAFT-INT-1**: emission at output-writing time, deterministic name, all destinations, fields sourced as specified; consumes (never defines) L-EPI's `synthesis` and L-CLI's `seats[]` namespaces.
- **R3 — Persist degradation content into artifacts**: the exit-3 alarm banner text and per-seat classified dropout causes must land in the verdict package — console-only alarms make the Q3 obligation unfulfillable.
- **R4 — Fail-loud return-dir writes**: a failed copy to `--return-dir` must never yield a bare exit 0; the empty-return-dir row of §2 must be reachable only via the known research deviation (until R1) — never via a silent copy failure.
- **R5 — Duration envelope in the CONTRACT**: a stated wall-clock class per mode (standard debate: minutes-class; `--deep` research: hour-class) so callers can set kill budgets rationally (§2 timeout row).
- **R6 — Lane-parity as a contract obligation**: any behavior added to Lane A must state its Lane B disposition (mirrored / N-A with reason) — generalizes the D2 deviation class and the repo's known inbox-mirroring blind spot into a checkable rule.
- **R7 — Contract-version line + package echo** per DRAFT-INT-2; `1.0` stamps when §7 Known-deviations empties.
- **R8 — Conformance-test obligation**: the §2 failure-semantics table is the acceptance surface — each row (exit path × observe/do/record) should be exercisable as a contract-conformance check when the technical architect slices implementation.
- **Constraint (standing):** doctor pre-flight is consumed strictly through the §C.2 seam (one command, machine-readable GREEN/YELLOW/RED + per-seat detail); no L-INT requirement may reach deeper into L-DOC's design.

---

**Done-contract check:** entry state verified with one delta flagged (stale worktree base, fixed) ✔ · Q1–Q7 each answered explicitly, none deferred ✔ · 2 draft ADRs, no numbers claimed, both baseline-independent ✔ · F6/F7/F8 refined with sharpened deciding evidence ✔ · 2 operator questions with recommendations ✔ · 8 requirements + 1 constraint forwarded, zero backlog items ✔ · seams (§C) consumed by reference only ✔ · settled list (§B) untouched ✔
