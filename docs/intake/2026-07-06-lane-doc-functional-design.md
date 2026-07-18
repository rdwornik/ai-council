# L-DOC Functional Design — Liveness / Doctor

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — `council doctor` #25 (`6e0782e`) + secrets rule #30 (`7e6a5e3`) shipped; open: #32, #52. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-06 · **Lane:** L-DOC (pillars 2+3 — "życie projektu") · **Mode:** FUNCTIONAL design — no code, no config edits, no backlog items
**Author:** Fable 5 (one of five parallel lane sessions; integration serialized through the primary checkout)
**Sources:** fleet-recon audit 2026-07-05 (§3 liveness matrix, §4 currency table, §6 P1 spec, §8 forks F1/F2/F5/F12), Fable architecture audit 2026-07-04, ratified state at `5c81e71` (ADR-11/ADR-12 Accepted, `COUNCIL_INVOCATION_CONTRACT.md`), `src/ai_council/healthcheck.py`, `src/ai_council/cli.py`, `config/settings.yaml`
**Seam contracts honored:** §C.2 (L-DOC owns the doctor; L-INT consumes it only as optional Lane-A pre-flight; L-CLI contributes exactly one requirement — identity-channel re-probe after CLI version changes). Nothing here designs metrics namespaces (L-CLI/L-EPI), parity methodology (L-CLI), or enforcement (L-GOV).

---

## 1. Lane charter & current state (verified live 2026-07-06)

| Charter entry-state claim | Live verdict | Evidence |
|---|---|---|
| P1 spec complete (recon §6) | **CONFIRMED** | `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` §6 P1, committed at `2593075` |
| `liveness.py` seed in session scratch "may be gone" | **DELTA — still present** | `%LOCALAPPDATA%\Temp\claude\C--Users-1028120-Documents-Dev-ai-council\9fd449a5-…\scratchpad\liveness.py` exists alongside `probes/`. Scratch is volatile — preservation is operator question Q-3 (§6) |
| Witnessed baseline 6/6 keys, 9/9 pings, one stale pin | **CONFIRMED as recorded** (recon §3/§4; not re-run — would spend API calls the design does not need) | Stale pin still live: `config/settings.yaml` research block pins `grok-4.20-reasoning` (located by content under `research.providers.grok`) — F12 unfixed by design |
| `cli.py:388 override=False` hazard open | **CONFIRMED** | `load_dotenv(_global_env, override=False)` at `cli.py:388`, followed by `load_dotenv(override=False)` at `:389` — both lines carry the hazard |
| (context) run-time health gate exists | CONFIRMED | `cli.py:430-441` — mode-scoped target selection, blocking filter for debate seats, summarizer soft-check; `healthcheck.py:58 run_health_checks()` + `classify_error` taxonomy (9 classes incl. `billing`/`auth`) |
| (context) no doctor artifact location exists yet | CONFIRMED | no `output/health/` present |

Also verified: ADR-12 §3 ratifies the **identity-logged-or-no-seat** invariant and names the three per-CLI identity channels (claude `.modelUsage` in-band; codex plain-mode stderr banner; grok session `events.jsonl`) — the doctor's CLI checks in Q5 build on these as *ratified inputs*, not new design. CONTRACT §8 step 3 already names "Pre-flight (optional) — health/liveness check before an important run" — the Lane-A seam this doctor fills.

**Worktree note:** this branch started behind main (missing `2593075`..`5c81e71`); fast-forwarded before design so all verification above is against ratified state.

---

## 2. Functional target state

**One line:** one command — `council doctor` — prints a one-screen GREEN/YELLOW/RED truth table over keys → seats → pins → CLI fleet, writes a dated machine-readable verdict under `output/health/`, never blocks anything by itself, and feeds a pin-upgrade lifecycle that makes model currency somebody's job instead of nobody's.

### 2.1 What the operator sees

Running `council doctor` (from any cwd, like every council command) produces, in under ~60 s of wall time and **zero LLM spend beyond the seat pings**:

```
COUNCIL DOCTOR — 2026-07-06T09:41 — verdict: YELLOW

KEYS        6/6 present (names only, values never printed)
            ! ANTHROPIC_API_KEY is SET-BUT-EMPTY in this shell (env shadowing)
              — treated as absent; global secrets used; run from a normal shell to clear
SEATS       debate 6/6 PASS · synthesizer (gemini) PASS · research 4/4 PASS
PINS        11 checked: 10 current · 1 STALE — research.grok grok-4.20-reasoning
              serves via unlisted alias; listed successor grok-4.20-0309-reasoning
              (advisory first seen 2026-07-05, age 1 doctor-run)
            3 superseded-available: claude-opus-4-7→4-8, gpt-5.4→5.5, sonnet-4-6→5
CLI FLEET   codex 0.141.0 auth=ChatGPT-subscription OK · claude 2.1.200 key-strip OK
            grok 0.2.82 auth=API-KEY (no OAuth — cost lane inactive)
            ! codex version changed since last doctor run → identity channel UNVERIFIED
              (run `council doctor --smoke codex` to re-verify; 1-token spend)

verdict: YELLOW — usable; 2 advisories, 1 stale pin, 1 unverified identity channel
record: output/health/doctor-2026-07-06T0941.json
```

The table order is fixed and is the epistemic-load order: **keys** (can anything run) → **seats** (does the panel exist — debate seats and the synthesizer named separately, research seats after) → **pins** (is the panel who we think it is) → **CLI fleet** (the cost lane's health). One screen, no scrolling, ASCII only (Windows cp1252 anti-pattern honored).

### 2.2 What a machine consumes (the §C.2 contract)

Every run writes `output/health/doctor-<ts>.json` and rewrites `output/health/doctor-latest.json` (a full copy, not a link — Windows). The Lane-A pre-flight contract is exactly what seam §C.2 promises: **one command, machine-readable GREEN/YELLOW/RED + per-seat detail.** Functional shape (fields, not schema):

- `schema_version` — the doctor record is a committed surface once L-INT consumes it; version it from day one
- `generated_at`, `verdict` (GREEN | YELLOW | RED)
- `checks[]` — one entry per check with: `class` (from the closed taxonomy in DRAFT-DOC-1), `subject` (key name / seat name / pin / CLI), `status` (PASS | ADVISORY | FAIL), `detail` (human string), and for seat pings the `classify_error` category verbatim
- `seats{}` — per-seat roll-up (debate / synthesizer / research flagged by role) so a caller can ask "are the seats *my profile* needs green?" without parsing prose
- `cli_fleet{}` — per-CLI: version, auth-lane, identity-channel state (VERIFIED | UNVERIFIED-SINCE-VERSION-CHANGE | NO-CHANNEL), last-verified timestamp
- `advisories[]` — currency items with `first_seen` (persisted across runs — this is what makes advisories age; see §3 Q4)

Doctor's own exit code mirrors the verdict in ADR-08 spirit: **0 = GREEN, 3 = YELLOW (degraded-but-usable), 1 = RED**. A Lane-A caller that only reads the exit code still gets the right coarse signal; `doctor-latest.json` has the detail.

### 2.3 How it behaves on failure

- Doctor **never refuses to run and never blocks a subsequent council run mechanically.** It is a diagnosis, not a gate — enforcement stays where it already lives (the run-time health gate at `cli.py:430-441`, which filters/aborts per mode) and in caller obligations (§3 Q2). A doctor that can block becomes a thing people `--skip`; a doctor that is always cheap to run and always honest gets run.
- A check that itself errors (list endpoint unreachable, CLI binary hangs) is reported as that check's FAIL with the cause — it never crashes the whole doctor or masks other rows. Partial truth beats no truth.
- Doctor loads secrets with **override=True** and *separately* reports any empty-but-set key as an env-shadowing advisory (§3 Q6) — the doctor must see reality even from inside a poisoned shell, and must tell you the shell is poisoned.

### 2.4 The lifecycle around it (flows)

1. **On demand** — operator or any session runs `council doctor` whenever provider/key/CLI state is in doubt.
2. **Lane-A pre-flight** — a foreign caller optionally runs it before an important commission (CONTRACT §8 step 3); consumes exit code + `doctor-latest.json`; on RED for the seats its profile needs, it does not commission (obligation, not mechanism — §3 Q2).
3. **Periodic** — weekly, matching the observed drift rate (one stale pin in ~7 weeks). Manual at first; mechanization is an implementation choice, not a design requirement.
4. **After change** — any provider/key/pin/CLI change warrants a run; a CLI **version** change specifically flips that CLI's identity-channel state to UNVERIFIED until a smoke re-probe (the L-CLI seam requirement, §3 Q5).
5. **Pin-upgrade loop** — currency advisories age across doctor runs; aged advisories reach an operator decision point with evidence proportionate to the seat's epistemic load; decisions land as config commits; the next doctor run confirms (§3 Q4, DRAFT-DOC-2).

A session references health state by reading `doctor-latest.json` **with an age check**: a verdict older than the cadence window (7 days) is reported as STALE-VERDICT, not as its color. An old GREEN is not green.

---

## 3. Design answers (charter Q1–Q6)

### Q1 — Doctor UX: the one-screen truth table + machine output

Answered in §2.1/§2.2. The load-bearing choices, stated for validation:

1. **Fixed row order = epistemic-load order** (keys → debate seats + synthesizer → research seats → pins → CLI fleet). The operator's eye learns one geography; the most run-fatal facts are highest.
2. **The synthesizer seat is named on its own line**, never folded into "debate 6/6". It is the highest-epistemic-load seat (ADR-12 §3) and the seat whose identity history is already contaminated (Wave-0 C1) — its health is never allowed to hide in an aggregate.
3. **Key checks are by NAME only** — values never printed, never written to the JSON record (recon discipline preserved).
4. **Human table and JSON record are generated from the same check results** — the screen is a rendering of the record, so they cannot diverge.
5. **Zero-spend posture is visible**: rows that would need LLM spend to firm up (identity re-probe) say so and name the flag, instead of silently claiming health they didn't verify.
6. The machine contract is §2.2 — one command, `verdict` + `seats{}` + `cli_fleet{}` detail, versioned. That is the whole §C.2 seam; L-INT consumes, L-DOC owns the shape.

### Q2 — RED semantics: advise, don't block — with one designed exception path and real teeth via obligations

**Stance: the doctor is universally advisory; enforcement lives in the two places that already have authority** — the run-time health gate (mechanical, per-run, already blocks/filters) and the invocation contract (caller obligations, Lane A). The doctor adds truth and a record; it does not add a third enforcement point that would compete with the existing gate and invite `--skip-doctor` culture.

Per failure class (this table is DRAFT-DOC-1's core):

| Failure class | Doctor verdict | Who enforces, and how |
|---|---|---|
| Missing key (any configured seat) | **RED** | Run-time: provider simply doesn't build (existing behavior); panel shrinks or run aborts. Doctor's added value: you learn it *before* commissioning, with the key named |
| Debate-seat or synthesizer ping FAIL | **RED** | Run-time health gate already blocks/filters (mode-scoped, `cli.py:430-441`). Lane-A obligation: a caller whose profile needs that seat MUST NOT commission on a RED pre-flight (CONTRACT-level wording, see below) |
| Research successes below the ADR-08 `min_successful_providers` analog | **RED** (research-scoped) | Run-time: existing exit-3 degradation alarm at run time. Doctor lets a research commission fail 60 s early instead of mid-run |
| Empty-but-set key (env shadowing) | **RED**, distinct class | Doctor reports; the council run's own stance is DRAFT-DOC-3 (Q6) — treat-as-absent-loudly, so the run self-heals while naming the poison |
| Stale pin (serves via unlisted alias) | **YELLOW** | Nobody blocks; feeds the pin-upgrade loop (Q4) with a deprecation clock ticking |
| Superseded-available (newer model exists, pin listed & serving) | **YELLOW advisory** (aging) | Nobody blocks; pure Q4 input |
| CLI auth-lane drift (e.g. grok silently API-billed), CLI version-change with unverified identity channel | **YELLOW** | Nobody blocks — CLI seats have same-seat API fallback (ADR-12 §4); a YELLOW here means "the cost lane is degraded / trust-but-unverified", never "the debate is at risk" |
| Doctor's own check errored (endpoint down, CLI hang) | that row FAIL, verdict ≥ YELLOW | Reported, never masked |

**The teeth, precisely:** when the doctor ships, CONTRACT §8 step 3 should be amended (one sentence, L-INT's document but L-DOC's requirement — recorded here as an input forward, §7) from "Pre-flight (optional)" to: *optional to run; if run and RED for a seat class the commission's profile requires, the caller MUST either not commission or record the override and the RED detail in whatever ADR/decision it derives* — the exact shape exit-3 obligations already have (ADR-08/CONTRACT §4). Advisory tool, contractual consumption.

**Does this need an ADR?** Yes — but not because RED blocks (it doesn't). DRAFT-DOC-1 is warranted because the **verdict taxonomy and the record become a committed, versioned surface** the moment a foreign repo consumes them (same logic that made the invocation surface ADR-11's business). The stance "doctor never mechanically blocks" is itself a decision worth making supersede-only.

### Q3 — Cadence & trigger design

| Trigger | What fires it | Scope |
|---|---|---|
| On demand | operator judgment; any session in doubt | full |
| Lane-A pre-flight | foreign caller, before an important/delegated run (CONTRACT §8.3) | full (cheap enough not to scope down) |
| Periodic — weekly | operator habit first; mechanization optional later | full |
| Post-change | any key/pin/provider/CLI change; specifically a CLI version change → identity-channel UNVERIFIED until smoke | full + targeted smoke |
| Pre-important-run (operator lane) | before any `important`-profile run | full |

- **Weekly is evidence-based, not habit-based:** the witnessed drift rate is one stale pin in ~7 weeks (recon §4). Weekly gives ~7× margin over the observed drift without making the doctor a chore. Revisit the cadence if two consecutive months of weekly runs produce zero new advisories (loosen) or if a mid-week breakage is ever missed (tighten).
- **Where results live:** `output/health/` — operational telemetry, NOT `docs/audits/` (review artifacts) and never the hub (ADR-09/ADR-10 both honored). `doctor-<ts>.json` per run + `doctor-latest.json` rewritten.
- **Retention:** keep 90 days or the last 15 runs, whichever is more. Enough to see an advisory age and to answer "when did this seat last pass"; short enough that `output/health/` never becomes an archive. The `advisories[]` `first_seen` field carries the long memory, so old files can be dropped without losing advisory age.
- **How a session references the latest verdict:** read `doctor-latest.json`; report `verdict` + `generated_at`; **if older than 7 days, report STALE-VERDICT instead of the color.** This one rule prevents the classic failure where a green light from three weeks ago licenses today's run.

### Q4 — Pin-upgrade lifecycle (DRAFT-DOC-2's content)

Today: the doctor (nobody, actually) says "newer exists" and nothing happens — F12 has been known-stale since 2026-07-05 with no owner. The designed loop:

1. **Surface** — the currency sweep classifies every pin: `current` · `superseded-available` (newer listed; pin listed & serving) · `stale-alias` (pin absent from the list endpoint but still serving — the grok-4.20 pattern) · `dead` (pin refused; this is a RED seat failure, not a currency item). Perplexity has no list endpoint: ping-only, classified `unverifiable`, never `current`.
2. **Age** — each advisory persists across doctor runs with `first_seen`. The screen shows age. Nothing nags on first sight; a fresh "newer exists" is noise, an eight-week-old one is negligence.
3. **Decision trigger** — an advisory reaches the operator's decision queue when: it is `stale-alias` (deprecation clock is ticking — decide within 2 doctor runs), or a `superseded-available` advisory is >30 days old, or the operator pulls it early (a new model generation they want).
4. **Decide — operator, always** (hub doctrine: cost/judgment gates live in operator judgment; the doctor never auto-upgrades). Evidence proportionate to the seat's epistemic load:
   - **Debate seat / synthesizer pin:** one paired mini-run — same brief, old pin vs new pin, scored against `protocols/SYNTHESIS_QUALITY_RUBRIC.md`. This *reuses the rubric*, exactly as ADR-12 §5 does; it is NOT the CLI parity experiment and does not touch the flip (L-CLI seam untouched). Plus release-notes review. A regression keeps the old pin and the advisory is marked `held(<date>, <reason>)` — a held advisory stops nagging but stays visible.
   - **Research seat pin:** release notes + one smoke ping of the successor. Lower epistemic load, lower evidence bar.
   - **Stale-alias class:** upgrade by default — the pin already serves the successor's lineage via alias, the alias can vanish without notice, and the change is a one-line config edit. Evidence bar: successor ping PASS. (F12 is this class — see §5.)
5. **Record** — the upgrade is a config commit on a branch, Conventional-Commit form `chore(config): <seat> pin <old> -> <new> — <evidence one-liner>`, merged `--no-ff`; JOURNAL entry per normal session discipline. **No ADR per pin bump** — a pin change inside the same provider is configuration, and git history is the changelog (ADR-48/49 doctrine). **Escalates to an ADR only when** the change moves a seat to a different provider, changes the default synthesizer, or changes panel composition — those touch ADR-01/02 territory.
6. **Confirm** — the next doctor run shows the pin `current` and the advisory gone. The loop is closed by the same instrument that opened it.

**Who owns it:** the operator decides; the doctor is the process's memory (surfacing + aging + confirming). That is the answer to "today it's nobody's job" — the job is split into a mechanical half (doctor) and a judgment half (operator), and the mechanical half never forgets.

### Q5 — CLI fleet health without spend

**"CLI healthy" for a seat-grade CLI (claude, codex; grok when OAuth lands) means all of:**

1. **Present + versioned** — binary found, version string read (`<cli> --version` class checks; zero spend).
2. **Auth-state good, on the intended lane** — per-CLI, all witnessed channels from the recon: codex `login status` says ChatGPT-subscription; claude's key-strip/empty-key guard intact so the subscription lane is actually in use; grok's auth surface names OAuth (until then the doctor reports `auth=API-KEY — cost lane inactive`, a standing YELLOW that is also the F3 re-probe trigger for L-CLI). Auth-lane *drift* — a CLI silently billing an API key when the subscription lane was intended — is a first-class YELLOW, because it defeats the cost lane silently.
3. **Identity channel verified for the current version** — the §C.2 seam requirement (L-CLI's F5 lean, accepted): the doctor record remembers each CLI's last-verified version; when the observed version differs, that CLI's identity-channel state flips to **UNVERIFIED-SINCE-VERSION-CHANGE (YELLOW)** until a smoke re-probe confirms the channel still parses (claude `.modelUsage` key present; codex banner carries `model:`; grok session file carries `model_id`). The re-probe costs ~1 token per CLI and fires **only on version change or explicit `--smoke`** — so the default doctor run stays zero-spend while identity trust is never silently stale.

**Zero-spend default, and the F2 escalation criterion (sharpened):** the default doctor run spends nothing on CLIs (checks 1–2 only; check 3 only when a version change forces it). Escalate to a routine smoke (per-doctor-run 1-token probes) **only when a witnessed incident occurs in which checks 1–2 were green but a CLI seat failed at debate time for a cause the smoke would have caught** (e.g. auth token expired but `login status` lied, or output format changed without a version bump). Until such an incident exists, routine smoke spend buys nothing the version-change trigger doesn't already cover. One incident = flip the default; zero incidents = zero-spend stands. (Refines F2 from "a month of checks" to an event-triggered criterion — see §5.)

**Non-seat CLIs** (agy — excluded; deepcode — no headless): the doctor does not probe them. It may list them as `excluded (ADR-12 §2)` for completeness, but their health is not the council's health. Re-admission evidence is ADR-12's business, not the doctor's.

### Q6 — Override stance: closing the `cli.py:388` hazard

The witnessed hazard: `load_dotenv(_global_env, override=False)` + a CC-session shell where the harness injects `ANTHROPIC_API_KEY=""` → the empty string wins, the claude seat fails auth, and the failure masquerades as a provider outage.

The fork was posed as binary (force-override vs refuse-loudly). Failure stories for each, then the recommendation:

- **Force-override (`override=True`) everywhere:** the poisoned-shell case self-heals — but a *genuinely intended* session key (a test key, an alternate billing key deliberately exported for one run) is silently trampled by the global secrets file. Silent, and it mis-bills. This violates "no silent decisions about whose key is used".
- **Refuse-loudly on empty-but-set:** honest, never mis-bills — but it makes every council run from inside a CC session fail at startup for a reason the operator will meet dozens of times, each time paying friction for a shell state they didn't choose. The doctor especially must not refuse: a diagnostic tool that won't run in a sick environment is useless precisely when needed.
- **Recommended — treat-empty-as-absent, loudly (a third stance):** an **empty-string env value is never a valid credential and never a meaningful intent** — treat any set-but-empty key as absent (so the global secrets value loads), and print one loud, non-fatal notice naming the variable: `ANTHROPIC_API_KEY was set-but-empty in this shell — ignored; global secrets used`. A **non-empty** session key keeps today's `override=False` semantics — a real value deliberately exported still wins. This preserves the only legitimate use of session keys, self-heals the only witnessed hazard, and is silent about neither.
- **Doctor-specific addition:** the doctor loads with full `override=True` for its *checks* (it must measure the real global credentials regardless of shell state) and reports the shadowing itself as the env-shadowing RED class (§Q2 table) — the doctor both works around the poison and diagnoses it.

This is DRAFT-DOC-3. It changes run behavior (the secrets loader), so it is flagged as the one item in this design that touches code the council run itself executes — still functional here; the implementation lands with the doctor build.

---

## 4. Draft ADRs

> Per master discipline: no real numbers (operator assigns at ratification); statuses are DRAFT. None of these is EPI-2-gated — the doctor is orthogonal to the synthesizer baseline.

### DRAFT-DOC-1 — Doctor semantics: advisory verdicts, committed record surface

**Status: DRAFT**

- **Decision:** the council ships a doctor (vehicle: `council doctor` subcommand — F1 resolution, §5) producing a three-color verdict (GREEN/YELLOW/RED) over the closed check taxonomy in §3 Q2's table, rendered as a one-screen human table and a versioned machine record (`schema_version`, `verdict`, `checks[]`, `seats{}`, `cli_fleet{}`, `advisories[]` with `first_seen`), written to `output/health/doctor-<ts>.json` + `doctor-latest.json`; exit code 0/3/1 mirrors GREEN/YELLOW/RED (ADR-08-consistent).
- **The doctor never mechanically blocks a run.** Enforcement remains with the existing run-time health gate and with Lane-A caller obligations (CONTRACT §8.3 amendment: a caller that ran the pre-flight and got RED for a required seat class must not commission, or must record the override — exit-3-style obligation).
- **The record is a committed surface:** once any foreign repo consumes `doctor-latest.json`, field removals/renames are breaking changes governed the same way as the invocation surface (ADR-11 §5 pattern).
- Verdict staleness rule: a record older than the cadence window reads as STALE-VERDICT, not as its color.
- **Rejected:** doctor-as-gate (invites skip-culture, duplicates the run-time gate's authority); doctor-as-standalone-script (fails the §C.2 "one command" seam for foreign callers — they already invoke `council`).

### DRAFT-DOC-2 — Pin-upgrade lifecycle: surface → age → decide → commit → confirm

**Status: DRAFT**

- **Decision:** model currency is a closed loop owned half-mechanically, half-by-judgment: the doctor classifies every pin (`current` / `superseded-available` / `stale-alias` / `dead` / `unverifiable`), persists advisories with `first_seen`, and escalates by age (stale-alias: decide within 2 runs; superseded: >30 days) to an operator decision. Evidence proportionate to epistemic load: debate/synthesizer pin — paired old-vs-new mini-run scored on SYNTHESIS_QUALITY_RUBRIC + release notes; research pin — release notes + successor smoke ping; stale-alias — upgrade by default on successor ping PASS. Upgrades land as `chore(config)` commits (branch → `--no-ff`, JOURNAL) — **no ADR per pin bump**; ADR required only when a seat changes provider, the default synthesizer changes, or panel composition changes. The next doctor run confirms closure. Regressions mark the advisory `held(<date>, <reason>)` — visible, not nagging.
- **Rejected:** auto-upgrade (violates operator-judgment doctrine); ADR-per-bump (config noise in the decision record; git history is the changelog per ADR-48/49 doctrine); "upgrade when convenient" (that is the current process; it is how F12 stayed open).

### DRAFT-DOC-3 — Secrets-loading stance: empty-is-absent, loudly (mini)

**Status: DRAFT**

- **Decision:** across all council entry points, a set-but-**empty** environment credential is treated as absent — the global secrets value loads — accompanied by one loud non-fatal notice naming the variable. A set-and-**non-empty** session credential retains precedence (today's `override=False` intent). The doctor additionally loads with full override for measurement and reports shadowing as a RED-class finding.
- Closes the witnessed `cli.py:388` false-auth-failure hazard without silently trampling deliberately-exported keys and without adding startup friction to every CC-session-launched run.
- **Rejected:** blanket `override=True` (silently mis-bills over intentional session keys); refuse-loudly (repeated friction for an environmental condition the operator didn't choose; unacceptable for a diagnostic tool in particular).

---

## 5. Refined forks

| Fork | Refinement | Deciding evidence — sharpened |
|---|---|---|
| **F1** doctor vehicle | **Recommend resolving NOW: `council doctor` subcommand.** The recon's deciding evidence ("do foreign repos need pre-flight?") has effectively arrived without waiting for P3 usage: seam contract §C.2 fixes the consumer contract as *one command*, and CONTRACT §8.3 already names the pre-flight step for callers that, by design, only know how to invoke `council`. A standalone script would force a second executable into the caller's world; extending the pre-run gate can't serve on-demand/periodic use. Nothing remains for usage data to decide. | None outstanding (evidence = ratified CONTRACT + §C.2 seam). Ratifies with DRAFT-DOC-1 |
| **F2** doctor CLI smoke-spend | Refined from time-boxed ("a month of auth-only checks") to **event-triggered**: default zero-spend (version+auth checks; smoke fires only on CLI version change or explicit `--smoke`); escalate to routine per-run smoke on the **first witnessed incident where auth/version checks were green but a CLI seat failed at debate time for a smoke-catchable cause**. | The incident itself. Zero incidents = zero-spend stands indefinitely; the version-change trigger already covers the known drift vector (F5's accepted lean) |
| **F12** stale research pin | **Disposition: DRAFT-DOC-2's inaugural case, stale-alias class → upgrade-by-default.** Evidence bar already met in the recon (successor `grok-4.20-0309-reasoning` is listed; alias still serves, so zero urgency but a ticking clock). Execute as a one-line `chore(config)` commit in the first implementation session after ratification — or immediately by the operator; this design lane makes no config edits. | None needed (recon verdict stands). The only open question is *when*, and DRAFT-DOC-2's stale-alias rule ("within 2 doctor runs of the class being assigned") answers it |

---

## 6. Questions for the operator (genuine forks only)

1. **RED teeth — accept the obligation-based stance?** DRAFT-DOC-1 makes the doctor purely advisory, with enforcement via the existing run-time gate plus a one-sentence CONTRACT §8.3 amendment (RED pre-flight ⇒ don't commission or record the override). The alternative is a mechanical block (doctor RED ⇒ `council --file` refuses without `--force`). **Recommendation: obligation-based** — a blocking doctor duplicates the run-time gate's authority and trains `--skip` habits; the exit-3 obligation pattern already works in this ecosystem. *(This is the ruling the lane most needs.)*
2. **Retention/location:** `output/health/`, 90 days / last-15-runs, `doctor-latest.json` as the session-referenced pointer — fine as specified? **Recommendation: yes** (operational telemetry doctrine; long memory lives in `advisories[].first_seen`, not old files).
3. **`liveness.py` seed preservation:** the P1 prototype still sits in *volatile* session scratch (`…\9fd449a5-…\scratchpad\liveness.py`). Copy it now to a non-volatile operator location as the implementation seed, or accept the risk of re-deriving it (the recon §3/§6 spec is sufficient to rebuild)? **Recommendation: copy now** — one file, zero cost, and it encodes the verbatim-pin and override=True lessons already debugged. (Not copied by this session: the design mode is repo-read-only + one output doc, and the seed's home should be the operator's choice, not `docs/`.)

---

## 7. Inputs forward for the technical architect

*(Requirements, contracts, constraints — explicitly NOT backlog items.)*

1. **Reuse, don't rebuild:** the doctor's seat pings are `run_health_checks()` + `classify_error` (existing, mode-proven); pins come only from `config/settings.yaml` (single source, verbatim — the recon's amendment-4 discipline is a requirement, not a preference); currency uses provider list endpoints (perplexity: none → ping-only/`unverifiable`); the scratch `liveness.py` (if preserved per Q-3) is the working seed for the sweep + override handling.
2. **Contract to honor (§C.2, fixed):** one command; machine-readable GREEN/YELLOW/RED + per-seat detail; fields per §2.2 with `schema_version` from day one; exit codes 0/3/1. Field removals/renames are breaking once L-INT consumes — treat like the ADR-11 §5 surface.
3. **CONTRACT §8.3 amendment (one sentence, lands with the doctor, via L-INT's document):** pre-flight stays optional; if run and RED for a profile-required seat class, the caller must not commission or must record the override — exit-3-style obligation wording.
4. **State the doctor must persist between runs:** `advisories[].first_seen` (aging), per-CLI last-verified version + identity-channel status (the §C.2 re-probe trigger), `held(<date>,<reason>)` markers. Everything else is per-run.
5. **Identity-channel re-probe (L-CLI's requirement, accepted):** smoke fires only on version change or `--smoke <cli>`; verifies channel *parseability* (claude `.modelUsage` present; codex banner `model:`; grok session `model_id`), ~1 token each; all CLI probes obey the ADR-12 §3 invariants (scratch cwd, stdin closed, read-only flags) — these are ratified, not restated here.
6. **Secrets loader change (DRAFT-DOC-3):** empty-string env credential ⇒ treated as absent with one loud notice; non-empty session values keep precedence; doctor measures with full override and reports shadowing. Touches the `cli.py:388-389` loader; the notice must be ASCII (cp1252 anti-pattern).
7. **Failure containment requirement:** every doctor check is isolated — a hanging CLI or dead endpoint fails its own row (with cause) and never the run; total wall time bounded (existing per-provider timeout caps in `healthcheck.py` are the pattern).
8. **Constraints inherited:** zero LLM spend on the default path beyond seat pings; keys by name only, never values, in screen or record; `output/health/` never `docs/` and never the hub (ADR-09/10); one-screen ASCII human output; runnable from any cwd like every council command.
9. **Out of scope, by seam:** metrics sidecar fields (L-CLI/L-EPI namespaces); parity-run methodology and the ADR-12 §5 flip (L-CLI); enforcement mesh (L-GOV); anything hub-side.

---

**Done-contract check:** entry state verified live with one delta flagged (`liveness.py` survives) ✔ · charter Q1–Q6 each answered explicitly, none deferred ✔ · three draft ADRs, none numbered, none EPI-gated ✔ · F1/F2/F12 refined with deciding evidence ✔ · operator questions are genuine forks with recommendations ✔ · zero code/config/backlog content ✔ · seam contracts §C.2/C.1/C.4 respected ✔
