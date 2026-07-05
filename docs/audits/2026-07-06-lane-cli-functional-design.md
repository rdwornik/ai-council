# L-CLI — CLI Backend & Cost Lanes: Functional Design

**Date:** 2026-07-06 · **Lane:** L-CLI (pillar 3, economics) · **Mode:** DESIGN — functional architecture only; no code, no config, no backlog items
**Author:** Fable 5 (functional-architect session, one of five parallel lane sessions)
**Authority context:** ADR-12 Accepted (ratified 2026-07-05); this document designs *within* it — nothing here reopens §B of the lane frame.
**Sources:** `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` (recon; probe IDs cited as `witnessed(<id>)`), `docs/audits/2026-07-04-fable-architecture-audit.md` (D6/D7, ADR-13 draft), `docs/decisions/ADR-12-*.md`, `protocols/COUNCIL_INVOCATION_CONTRACT.md`, `protocols/SYNTHESIS_QUALITY_RUBRIC.md`.

---

## 1. Lane charter & current state (verified live 2026-07-06)

| Charter entry-state claim | Live verdict |
|---|---|
| ADR-12 Accepted (v1 = claude+codex, four invariants, gradient, gated flip) | **VERIFIED** — `docs/decisions/ADR-12-provider-backend-engine-and-cost-lanes.md`, Status: Accepted; §5 flip clause evidence-gated |
| Witnessed fleet matrix in recon §2 | **VERIFIED** — present in repo at merge `5c81e71` |
| grok API-billed as configured (F3 open) | **VERIFIED per recon** — witnessed cell (`grok models` header "You are using XAI_API_KEY"; `~/.grok/auth.json` absent); not a SKIPPED cell, no re-probe required |
| No CliProvider code (0 subprocess hits) | **VERIFIED** — live grep `subprocess|Popen` over `src/` = 0 hits |

**Delta flagged (session mechanics, not design):** this worktree was created from `75006db` (pre-ratification main); the ratification merge `5c81e71` existed on main but not on the branch. The branch was fast-forwarded to `5c81e71` (clean ancestor, no divergence) so all sources are in-worktree.

**Additional live grounding used below:** the current `_metrics.json` sidecar (written by `_save_metrics_json` in `output.py`) already carries a flat `calls[]` array (provider, round, tokens, cost, latency, `was_retry`) plus an optional `synthesis` block — the `seats[]` design in §3.Q5 is a purely **additive** extension alongside these; the rubric is a 5-item binary checklist with a 25–50-sample smoke-test convention, which §3.Q2 reuses.

---

## 2. Functional target state

*How the lane looks and behaves when done — written for a non-implementer to validate.*

**One line:** a CLI-backed seat is epistemically indistinguishable from an API seat — identity witnessed on every call, contamination contained by construction, every deviation recorded and degrading loudly — so the subscription lane saves money without the debate ever being able to silently pay for it.

### 2.1 What the operator sees

- **At run start (standard profile):** a one-line-per-seat routing note — seat name, requested backend, model pin. No new prompts, no new flags required for the default path; `important` and research runs look exactly as today (all-API, by ratified rule).
- **During the run:** nothing new when all is well. When a CLI seat fails or its identity cannot be read, a single visible YELLOW line: *"seat `<name>`: CLI backend degraded (`<cause>`) → retried via API"*. Never silent.
- **At run end:** the existing cost summary gains a lane split — API-billed cost vs subscription-lane calls (recorded at $0 marginal cost, flagged as such, never pretending the tokens were free in kind). Fallback events are summarized in one line if any occurred.
- **In the artifacts:** transcript and verdict are byte-format-identical to an all-API run — no CLI banners, envelopes, or harness artifacts anywhere in debate text. The `_metrics.json` sidecar carries the full `seats[]` truth (§3.Q5). The transcript itself does not mark backends (anonymization posture unchanged); the sidecar does.

### 2.2 Process flow (one debate turn through a CLI seat)

1. **Panel build:** routing (ADR-12 §4) assigns each seat a backend from profile + config. Doctor's latest verdict, when present, is advisory input (L-DOC owns doctor; this lane only consumes).
2. **Invocation:** the seat receives *exactly* the prompt string an API seat would receive (persona, mode directive, critique template — assembled upstream of transport). Transport executes under the ADR-12 safety floor: scratch cwd, stdin closed, read-only/tools-off flags, explicit model pin.
3. **Admission gate (new, the lane's hinge):** before a CLI response may enter the round, its **actual served model identity** must be read from that CLI's witnessed channel. Identity unreadable → the response is discarded *unentered*, the seat retries via API, and a degradation event is recorded. No content that fails the gate is ever seen by critics or synthesizer.
4. **Normalization:** the admitted response is reduced to the same response semantics as an API call (text + usage where available) and enters the normal pipeline — ADR-03 anonymization applies unchanged.
5. **Recording:** the `seats[]` sidecar entry is written for *every* seat (API seats too, uniformly), with requested-vs-actual backend and model, identity channel used, and any fallback events.

### 2.3 Failure semantics (operator-visible contract)

| Failure | Behavior | Exit-code effect |
|---|---|---|
| CLI process error / timeout / parse failure | Same-seat API retry (visible YELLOW; `fallback_events[]` entry) | None (cost degradation ≠ epistemic degradation) |
| CLI quota exhaustion | Same as above — quota exhaustion degrades cost, never the debate (ADR-12 §4) | None |
| Identity unreadable | Response discarded pre-admission; same-seat API retry; degradation recorded | None if API retry succeeds |
| CLI fails AND API fallback fails | Ordinary seat dropout — existing minimum-panel semantics | Existing semantics (ADR-08 unchanged) |
| Post-hoc discovery that unverified content entered a round | **Contract breach** — cannot happen by design (gate is pre-admission); if a bug lets it happen, the run is marked invalid in metrics and the operator alerted | Treated as hard error for the run's evidentiary standing |

The deliberate consequence: **rounds are never invalidated retroactively.** All verification happens before content is admitted, so a completed round is always built on verified inputs.

---

## 3. Design answers (charter questions Q1–Q7)

### Q1 — Seat-equivalence contract (the centerpiece)

The claim to make testable: *transport is epistemically irrelevant.* A debate reading only the transcript and verdict must have no way to distinguish a CLI-backed seat from an API seat — every real difference is confined to the metrics sidecar and the cost line. Six invariants, each with an observable check:

| # | Invariant | Functional statement | Observable check |
|---|---|---|---|
| **I1** | **Identity witnessed** | Every CLI response carries actual served-model identity, read from that CLI's witnessed channel (claude: in-band `.modelUsage`; codex: plain-mode stderr banner; grok: session `events.jsonl` via `sessionId`). Unreadable identity → response never admitted. | Every admitted CLI response has a non-null `actual_model` in `seats[]`; an induced identity-channel break produces a fallback event and an API-served answer, never an admitted unverified answer. |
| **I2** | **Context isolation** | Invocation cwd is a dedicated scratch dir, never a repo; stdin closed; read-only/tools-off flags set. Scratch-cwd is the PRIMARY isolation (tools-off does not block ingestion — witnessed CL-3, GR-3). | Canary test: with a canary `CLAUDE.md`/`AGENTS.md` placed in a repo-like cwd, the canary string never appears in any seat response across the suite; invocations observed launching from scratch. |
| **I3** | **Prompt parity** | The CLI transport receives the identical prompt string an API seat would — persona injection, mode directive, critique template (including `{previous_responses_anonymized}`) assembled upstream of the transport choice. | For a fixed seed debate state, the prompt handed to the CLI transport is string-equal to the prompt the API transport would receive. |
| **I4** | **Output normalization** | CLI output is reduced to the same response semantics as an API call; no harness artifacts (banners, JSON envelopes, stderr noise, `.thought` fields) leak into debate text. | Transcript diff discipline: a CLI-backed transcript is structurally indistinguishable from an API transcript; grep for known envelope markers = 0. |
| **I5** | **Anonymization unaffected** | Admitted responses enter the ADR-03 shuffle/relabel path unchanged; the transcript never marks backend. Backend truth lives only in the sidecar. | The anonymized critique inputs for a CLI-backed round are format-identical to an API round; no backend-correlated label leakage. |
| **I6** | **Failure equivalence** | CLI failures classify into the existing error taxonomy (extended with the CLI-specific causes in §3.Q3) and honor the same `timeout_sec` as a hard kill; failure handling downstream (retry, dropout, alarm) is the same machinery an API failure uses. | Induced CLI hang is killed at `timeout_sec` and surfaces as a classified failure identical in downstream handling to an API timeout. |

Two ratified corollaries restated as contract consequences (not new rules): the synthesizer seat is never CLI-backed (highest epistemic load → pinned API identity), and `important`/research profiles are all-API — so the contract's exposure is confined to standard-profile debate seats.

This contract is `DRAFT-CLI-1` (§4). Its acceptance test is the checklist in the right column — a non-implementer can validate each row by observation.

### Q2 — Parity-run methodology (CLI-4; the ADR-12 §5 evidence machine)

**Design: paired, blinded, rubric-scored non-inferiority trial.**

- **Unit:** a *pair* = the same brief run twice — once `standard` CLI-backed (codex + claude CLI seats, all pins explicit per the per-call pin rule), once all-API. Same panel composition, same synthesizer (API, both arms — the synthesizer is never part of the manipulation), same rounds.
- **Question classes:** decision-mode only — `pick` and `judge` (research is always API by ratified rule; `ideas` excluded because the rubric's faithfulness items bind weakly to divergent output). Stratify: half pick, half judge; briefs drawn from real past inbox briefs where available (ecological validity) plus fresh ones to avoid training-on-the-test.
- **Sample size:** **n = 12 pairs** (24 syntheses) — inside the rubric's own 25–50-sample smoke-test convention at the low end, proportionate to a config-default decision that stays reversible (the flip is a default, not a capability).
- **Blinding (the hard requirement):** the scorer must not know the backend. Mechanics: a preparation step (performed by a session that will not score) strips backend-revealing content — the metrics sidecar is withheld entirely, run banners/cost lines removed — and presents each pair as artifacts A/B with per-pair randomized assignment. The assignment key is written to a sealed mapping file, opened only after all 12 pairs are scored. Scorer: the operator, applying the rubric per its own operating principle (any-item failure is a signal, investigate before concluding).
- **Success threshold (ratifies the flip):** on each of the five rubric items, the CLI arm fails **at most one more pair** than the API arm (non-inferiority margin = 1/12 per item), **except** items 2 (no hallucinated consensus) and 4 (faithfulness), where the margin is **zero** — these are the signature failure modes of harness contamination (§3.Q6) and get no allowance.
- **What keeps API default forever:** a faithfulness-class regression (item 2 or 4) that *persists across two parity attempts separated by a containment fix* — i.e., the tax is measured, a fix is attempted, and the tax remains. One failed attempt pauses the flip and triggers diagnosis (per the rubric's operating principle); a second consecutive failure on the same items closes the fork: the flip clause is retired and `backend: cli` stays per-seat opt-in indefinitely.
- **Cost accounting note:** the parity run itself is half API-billed by construction; it is a one-time evidence purchase, sized accordingly (12 pairs ≈ 12 ordinary debates of API spend).

This methodology, with an **empty evidence slot**, is `DRAFT-CLI-3` (§4) — the flip amendment pre-draft the charter expects.

### Q3 — Fallback semantics

**Rule: visible degradation, same-seat API retry, pre-admission gating — never silent, never retroactive.**

- **Mid-debate CLI failure** (process error, timeout, parse failure, quota exhaustion, identity unreadable): the seat retries via API *within the same round*, under the round's existing time budget. The debate content is unaffected (equivalence contract: same prompt, same seat, same downstream handling); the *cost* is affected, and that is recorded.
- **Visibility:** one YELLOW console line per event (cause + resolution); an entry in `seats[].fallback_events[]`; a one-line roll-up in the end-of-run summary. "Silent same-seat retry" is rejected: silence here would make the cost lane's failure modes invisible exactly where F5 (channel fragility) predicts they will occur, and would starve F2/F5 of their deciding evidence.
- **Transcript:** records nothing — backend is not an epistemic fact of the debate (I5). Metrics record everything.
- **When a fallback invalidates a round: never.** The admission gate (§2.2 step 3) runs *before* content enters the round, so critics and synthesizer only ever see verified content. The only invalidating event is a contract breach (unverified content discovered post-admission), which is a bug-class event, not an operating mode — it marks the run's evidentiary standing as invalid in metrics and alerts the operator.
- **Exit codes:** unchanged (ADR-08 untouched). A successful API fallback is exit-code-invisible; a seat that fails both lanes follows existing dropout/minimum-panel semantics.
- **Taxonomy extension (functional):** the error classifier gains CLI-specific causes — `quota`, `timeout`, `parse`, `identity-unreadable`, `process-error` — which are exactly the `fallback_events[].cause` vocabulary (§3.Q5), so the classifier and the record share one vocabulary by construction.

### Q4 — Quota-competition policy (the unruled fork; recommendation produced)

**Problem:** the council's CLI lane draws on the operator's own subscriptions. The claude seat competes *directly* with the operator's primary work tool (Claude Code); codex idles (witnessed: ChatGPT subscription, logged in, not the operator's daily driver).

**Options considered:**
- *Profile-based time windows* (council may use CLI lane only in defined hours) — rejected: the operator's schedule is not machine-predictable; windows add configuration surface with no witnessed failure they'd prevent.
- *Per-run quota probes* (check remaining quota before routing) — rejected on witnessed evidence: quota state is not reliably readable anywhere in the fleet (agy invisible; claude/codex expose no headless quota surface in the probe record). A policy gated on unreadable state is a policy that lies.
- *Dynamic quota accounting* — already rejected in ADR-12 (Considered and rejected); not reopened.

**Recommendation — asymmetric static defaults + fallback-as-relief-valve (`DRAFT-CLI-2`):**

1. **codex seat: default-CLI in `standard` profile.** Its subscription idles; competition with operator work is ~zero. This is where the cost win actually lives (recon §5.5 revised gradient: codex first).
2. **claude seat: CLI is per-run opt-in** (flag or frontmatter), API default *even in `standard`* — because every council claude-CLI call competes with the operator's own CC session quota, and the operator should spend that budget knowingly, per run, not by standing default. (Hub doctrine "cost gate lives in operator judgment" — this makes the judgment per-run explicit.)
3. **grok seat: outside the policy until F3 resolves** (no subscription lane configured — nothing to compete over; an API-billed CLI call is strictly worse than the API seat).
4. **Relief valve:** quota exhaustion is already a fallback cause — a starved CLI seat degrades to API visibly and the debate proceeds. The council can therefore never *block* on quota, and the operator can never be blocked *by* the council: the collision cost is always money (API tokens), never a stalled debate or a stalled dev session.
5. **Learning loop:** `fallback_events[]` with cause `quota` is the policy's own telemetry — if codex quota-fallbacks are frequent, the "idle subscription" premise is wrong and the policy is revisited on that evidence.

### Q5 — Metrics `seats[]` namespace (seam owner: this lane)

Additive extension to the existing `_metrics.json` sidecar (current `calls[]` and `synthesis` blocks untouched; the `synthesis` namespace contents are **L-EPI's** — reserved, not designed here). One entry per seat per run, **uniform for API and CLI seats** so consumers never branch on backend:

| Field | Type / vocabulary | Semantics |
|---|---|---|
| `seat` | string | The settings.yaml seat name (e.g. `openai`, `claude`) |
| `requested_backend` | `api` \| `cli` | What routing assigned at panel build |
| `actual_backend` | `api` \| `cli` | What finally served the seat's admitted responses |
| `cli` | object \| null | When any CLI attempt occurred: `{ name, version }` (version captured at run start — F5's forensic anchor) |
| `requested_model` | string | The seat's settings.yaml pin, as passed on the call |
| `actual_model` | string \| null | Served identity from the witnessed channel; API seats: the model echoed in the API response. **Null only in a degradation record — never on an admitted response** |
| `identity_channel` | `modelUsage` \| `stderr-banner` \| `session-events` \| `api-echo` | Which channel supplied `actual_model` |
| `identity_readable` | bool | False ⇒ the CLI attempt was rejected at the admission gate (and a fallback event exists) |
| `fallback_events[]` | list | Each: `{ round, from_backend, to_backend, cause, detail }`; `cause` ∈ `quota` \| `timeout` \| `parse` \| `identity-unreadable` \| `process-error`; `detail` = classified error string |

**Degradation-event semantics for unreadable identity (charter's explicit ask):** `identity_readable: false` and an admitted response are mutually exclusive by contract (I1). The record of an unreadable-identity event is therefore always the *pair*: a `fallback_events[]` entry with cause `identity-unreadable`, plus `actual_backend`/`actual_model` reflecting whatever finally served (normally the API retry). A run whose sidecar shows `identity_readable: false` with `actual_backend: cli` is self-evidently contract-breaching — the sidecar is designed so that this breach is *detectable from the record alone*.

**Consumers served:** the parity run's blinding prep (withholds this file), L-EPI's archaeology segmentation (backend per seat per run, forever answerable), L-DOC's doctor (version + channel fields feed the re-probe requirement), F3/F5's deciding evidence.

### Q6 — Harness-contamination stance

**Name the tax:** a CLI does not serve a bare model — it serves model-plus-vendor-harness. Each wrapper injects an agentic system prompt (coding-assistant persona, tool-use framing, vendor safety scaffolding). The epistemic tax, precisely: (a) **voice shift** — the seat's style and register are harness-colored, a new *correlated* styling layer on top of the pre-existing per-model style leakage ADR-12 already accepts; (b) **instruction-hierarchy interference** — persona and mode directives arrive as *user-level* text under a vendor *system* prompt, so directive adherence may differ from the API seat where our text sits closer to the top; (c) **refusal/hedging profile shifts** — vendor safety framing can change what a seat will assert confidently; (d) **context-ingestion risk** — contained by scratch-cwd (I2), witnessed rather than assumed.

**Containment already ratified (restated, not redesigned):** `important`/research all-API; synthesizer never CLI; anonymization unchanged; residual risks accepted and logged per-run (ADR-12 §3).

**Is more needed? Yes — two additions, both cheap; and one refusal:**
1. **Measurement, not mitigation:** the parity run (Q2) is the tax's measuring instrument — its zero-margin items (2, 4) are chosen because hallucinated consensus and unfaithful representation are exactly what instruction-hierarchy interference would produce. The tax is not argued about; it is measured, with a named threshold.
2. **Permanent segmentability:** every response's backend is recorded per seat per run (Q5), so any future epistemic scoring can segment CLI-served from API-served output — the direct lesson of the synthesizer-contamination history (L-EPI's archaeology exists because this recording was missing there). We never again produce output whose provenance can't be reconstructed.
3. **Refused: prompt-level counter-instructions** ("ignore your harness persona…") — untestable, adds an arms-race layer inside the prompt, and would itself differ per CLI, breaking prompt parity (I3).

### Q7 — v2 resolver ranking (DRAFT-GATED(EPI-2))

The audit's ADR-13 annotation is **confirmed as shaped**: v2 crux-resolver candidates ranked by witnessed sandbox posture — **codex first** (`--sandbox read-only` witnessed CX-1/CX-2), **claude second** (tools-off + scratch-cwd; tools-off is not isolation — CL-3 — so scratch-cwd carries the weight), **grok third** (sandbox flag present in help, unexercised — SKIPPED cell; would require probing before any admission).

Two annotations, shape-level only, nothing beyond:
- A resolver seat inherits the full seat-equivalence contract (I1–I6) *plus* ADR-13's own rule that resolver output enters Round 2 as an anonymous, tool-derived evidence block — identity-logged-or-no-resolver applies exactly as identity-logged-or-no-seat.
- The resolver is the one place a CLI's agentic harness is a *feature* (an agent can run a check; an endpoint can only assert one — audit's framing), which is why the ranking keys on sandbox posture rather than on the debate-seat criteria.

Status: `DRAFT-CLI-4`, gated on the EPI-2 ruling (Epic B baseline). Stops at shape, per seam contract §C.3.

---

## 4. Draft ADRs

### DRAFT-CLI-1 — Seat-Equivalence Contract for CLI-backed seats
**Status: DRAFT**

- **Decision (functional):** a CLI-backed seat is admitted to a debate only under invariants I1–I6 (§3.Q1: identity witnessed, context isolation, prompt parity, output normalization, anonymization unaffected, failure equivalence), enforced by a pre-admission gate — no response enters a round before its served identity is verified. Fallback semantics per §3.Q3: visible degradation, same-seat API retry, rounds never retroactively invalidated; exit codes unchanged.
- **Relation to ADR-12:** operationalizes §3's safety invariants into a testable contract with named observable checks; adds no new invariant class, sharpens "identity-logged-or-no-seat" into "identity-verified-before-admission".
- **Acceptance test:** the observable-check column of the I1–I6 table, verifiable by a non-implementer.

### DRAFT-CLI-2 — Quota-Competition Policy (asymmetric static defaults)
**Status: DRAFT**

- **Decision (functional):** in `standard` profile — codex seat defaults to CLI (idle subscription); claude seat's CLI backend is per-run opt-in (API default everywhere absent explicit opt-in); grok excluded until F3 resolves. Quota exhaustion is a fallback cause, never a blocker, in either direction (council never stalls on quota; operator never stalls on council). `fallback_events[cause=quota]` frequency is the policy's revision trigger.
- **Rejected:** time windows (unpredictable schedule, no witnessed failure prevented); per-run quota probes (quota state not reliably readable — witnessed); dynamic accounting (already rejected, ADR-12).

### DRAFT-CLI-3 — ADR-12 §5 Flip Amendment (pre-draft, evidence slot EMPTY)
**Status: DRAFT-GATED(CLI-4 parity-run results)**

- **Shape:** `standard` defaults to the CLI lane per ADR-12 §5, ratified iff the §3.Q2 parity run passes: n = 12 blinded pairs (pick/judge stratified), rubric-scored, non-inferiority margin 1/12 per item with **zero margin on items 2 and 4**.
- **Evidence slot:** `[EMPTY — CLI-4 results: pass/fail per rubric item, sealed-key blinding record, scorer, date]`
- **Kill condition (named now):** items 2/4 regression persisting across two parity attempts separated by a containment fix ⇒ this amendment is retired unratified and `backend: cli` remains per-seat opt-in indefinitely.

### DRAFT-CLI-4 — v2 Crux-Resolver ranking annotation
**Status: DRAFT-GATED(EPI-2 ruling)**

- Confirms the audit's ranking (codex > claude > grok, by witnessed sandbox posture); adds the two shape-level annotations of §3.Q7 (resolver inherits I1–I6 + anonymous-evidence rule; agentic harness is a feature only in this role). Nothing beyond shape; ratification waits on Epic B.

---

## 5. Refined forks (F3, F4, F5 — this lane's)

| Fork | Refined statement | Recommendation | Deciding evidence (sharpened) |
|---|---|---|---|
| **F3** grok seat timing | Ship API-billed now vs wait for subscription OAuth. Sharpened: an API-billed grok CLI seat is *strictly dominated* by the existing grok API seat (same billing, plus subprocess/parse surface, plus harness tax) — "ship now" isn't a lesser option, it's a negative-value one. | **Wait.** Defer the grok adapter entirely until OAuth exists; v1 stays claude+codex per ADR-12. | Operator runs `grok login`; re-probe must witness **both** the subscription auth lane active (auth.json present, API-key header gone) **and** the identity channel intact post-login (session `events.jsonl` layout unchanged), **and** an env-key shield for grok invocations mirroring the claude key-strip pattern. Three checks, one 5-minute session. |
| **F4** codex model policy | The recon posed pin-via-`-m` vs accept-codex-default. Sharpened: ADR-12 §4 already ratified the per-call pin rule, so the fork as stated is *half-closed* — the live residue is only whether a codex-default (`gpt-5.5`) arm is worth testing as a separate cost/quality experiment. | Pin always (ratified). A codex-default arm is **out of the parity run** (it would confound backend with model identity — the parity run must vary transport *only*). | If pinned parity passes and the flip lands: a separate, smaller paired comparison (pinned vs codex-default on the codex seat only) decides whether the default is admissible as a cheap tier. Until then, nothing. |
| **F5** identity-channel fragility | Banner/session-file/JSON-key parsing is version-fragile. The recon posed doctor-re-probe vs per-run hard-fail as alternatives. Sharpened: **they are not alternatives — they are layers.** The per-run hard-fail already exists by contract (I1: unreadable ⇒ not admitted ⇒ API fallback); the doctor re-probe is the early-warning that keeps the hard-fail from firing mid-debate. | Both, by construction: I1 is the floor (this lane's design); the re-probe after CLI version changes is this lane's **one** requirement contributed to L-DOC's doctor (seam §C.2 — stated as a need, not designed here). The `seats[].cli.version` field (§3.Q5) gives both layers their forensic anchor. | First CLI auto-update that breaks a channel parse: if the doctor re-probe catches it before any run degrades, the layering is validated; if a run's fallback event is the first detector, re-probe cadence (L-DOC's design space) needs tightening. |

---

## 6. Questions for the operator (genuine forks only)

1. **Quota policy asymmetry (DRAFT-CLI-2):** accept codex-default-CLI / claude-CLI-opt-in? The alternative worth naming: claude also default-CLI in `standard`, accepting quota competition with your own CC work in exchange for simpler routing. **Recommendation: the asymmetry** — the codex subscription is where the free win is; claude CLI spend should be a knowing per-run choice because it draws down your primary work tool.
2. **F3 sequencing:** run `grok login` (+ key shield) now so grok enters v2 planning with evidence, or defer OAuth until claude+codex adapters have proven the lane? **Recommendation: defer** — the re-probe is 5 minutes whenever it happens, and doing it now buys nothing while v1 adapters don't exist; sequencing evidence purchases to decision points is the recon's own discipline.
3. **Parity threshold (DRAFT-CLI-3):** accept n = 12 pairs with zero-margin on faithfulness items and the two-strikes kill condition? Cheaper (n = 8) weakens the non-inferiority claim below the rubric's own sample convention; richer (n = 25 pairs) doubles a one-time API spend for confidence the reversible-default decision doesn't need. **Recommendation: n = 12 as designed.**

---

## 7. Inputs forward for the technical architect

*Requirements, contracts, constraints — explicitly NOT backlog items.*

1. **The seat-equivalence contract (I1–I6) is the acceptance frame** for any CliProvider implementation — each invariant's observable check should become an automated or checklist test; the canary test (I2), identity-gate test (I1), and stdin/timeout test (I6) are the non-negotiable three.
2. **Admission gate placement constraint:** identity verification happens between transport return and round entry — upstream of critique fan-out and of anonymization. Nothing downstream may ever receive unverified CLI content.
3. **Metrics extension is additive:** `seats[]` per §3.Q5 lands alongside the existing `calls[]`/`synthesis` blocks; the `synthesis` namespace is reserved for L-EPI — do not populate it from this lane. API seats get uniform entries (`identity_channel: api-echo`).
4. **One shared vocabulary:** the error-classifier extension categories and `fallback_events[].cause` are the same five-token set (`quota|timeout|parse|identity-unreadable|process-error`) — a single source, not two lists to drift.
5. **Version capture at run start** (`seats[].cli.version`) — required by F5's layering and by the doctor re-probe seam.
6. **Witnessed invocation shapes are the spec** (ADR-12 §Context): claude `-p --output-format json --tools "" --model <id>`; codex `exec --sandbox read-only --skip-git-repo-check -m <id>` with stdin closed (plain-mode stderr banner is the identity source — the `--json` stream carries none); scratch cwd always. Any deviation discovered at build time is a contract change, not an implementation detail.
7. **Parity-run tooling needs (when CLI-4 is scheduled):** a blinding-prep step that withholds sidecars, strips banners/cost lines, randomizes pair labels, and writes a sealed mapping file; a scoring sheet keyed to the five rubric items per pair.
8. **Constraints inherited, restated:** no provider-class merging (CLAUDE.md §5.7); all config in `settings.yaml` (backend axis is config-only); `timeout_sec` reused as the subprocess hard kill; hub is never a destination; exit-code semantics untouched.

---

**Done-contract check (lane frame §A):** charter entry state verified with one flagged delta ✔ · functional target state validated-readable by a non-implementer ✔ · Q1–Q7 each answered explicitly (none deferred) ✔ · four draft ADRs, two DRAFT / two DRAFT-GATED with named gates, no real ADR numbers claimed ✔ · F3/F4/F5 refined with deciding evidence ✔ · three operator questions with recommendations ✔ · inputs-forward contains zero backlog items ✔ · seam contracts respected: `synthesis` namespace untouched (L-EPI), doctor consumed not designed with exactly one contributed requirement (L-DOC), flip evidence owned here (§C.4) ✔ · zero source/config changes ✔
