# GOV-1 Operator Rulings Register — RULED

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — 15 rulings applied; ADR-09/10 Accepted, DRAFT-GOV-1 ratified as ADR-14 (`85a692f`). No open remainder. [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-17 · **Status:** RULED · **Branch:** `docs/gov1-consolidation` · **Gate:** G1 → G2 (#31)

**Acceptance scope (operator, 2026-07-17):** *ACCEPT ALL over the lane-doc §6 primary set as
quoted here; this supersedes the technical-architect-intake §5 labels, whose drift is noted
below.* Each ruling is the functional architect's §6 **Recommendation**, accepted verbatim and
flipped to **RULED**. Source of record = each lane doc's `§6` (never a handoff); the intake §5
register was a navigation aid only.

> This register closes gate **G1** (operator acceptance of the rulings) and is a #31 (GOV-1)
> artifact. It records **17 decisions**: the 15 lane-doc §6 rulings + the #1/#24 FORK_RULING +
> a PENDING placeholder for the EPI-1 ruling (deferred to its own Beat-1 mini-session).

---

## The 15 lane-doc §6 rulings (verbatim recommendations → RULED)

### L-EPI — `docs/intake/2026-07-06-lane-epi-functional-design.md` §6

- **OQ-1 — Shape of the #1 scoring run** → **RULED.**
  > **Recommendation: comparative** — gemini-vs-openai segment scoring (uses the contamination as
  > a free A/B experiment; same rubric, same corpus, materially stronger Branch A/B evidence).
  *(This ruling is what makes #24 the superset of #1 — see the FORK_RULING below.)*
- **OQ-2 — #19 default posture at ship** → **RULED.**
  > **Recommendation: default-on post-calibration** — an opt-in alarm protects nobody by default,
  > and the posture is already advisory-only.
- **OQ-3 — Scoring authority for EPI-1** → **RULED.**
  > **Recommendation: operator scores the ruling segments (≈2×10–20 syntheses, blind); LLM judge
  > admissible as a recorded second opinion only.**

### L-CLI — `docs/intake/2026-07-06-lane-cli-functional-design.md` §6

- **Q1 — Quota policy asymmetry (DRAFT-CLI-2)** → **RULED.**
  > **Recommendation: the asymmetry** — the codex subscription is where the free win is; claude CLI
  > spend should be a knowing per-run choice because it draws down your primary work tool.
- **Q2 — F3 sequencing** → **RULED.**
  > **Recommendation: defer** — the re-probe is 5 minutes whenever it happens, and doing it now buys
  > nothing while v1 adapters don't exist; sequencing evidence purchases to decision points is the
  > recon's own discipline.
- **Q3 — Parity threshold (DRAFT-CLI-3)** → **RULED.**
  > **Recommendation: n = 12 as designed.**

### L-DOC — `docs/intake/2026-07-06-lane-doc-functional-design.md` §6

- **Q1 — RED teeth / doctor stance (DRAFT-DOC-1)** → **RULED.**
  > **Recommendation: obligation-based** — a blocking doctor duplicates the run-time gate's authority
  > and trains `--skip` habits; the exit-3 obligation pattern already works in this ecosystem.
- **Q2 — Retention / location** → **RULED.**
  > **Recommendation: yes** (operational telemetry doctrine; long memory lives in
  > `advisories[].first_seen`, not old files).
- **Q3 — `liveness.py` seed preservation** → **RULED.**
  > **Recommendation: copy now** — one file, zero cost, and it encodes the verbatim-pin and
  > override=True lessons already debugged. *(Seed home is the operator's choice, not `docs/`; not
  > copied by a design session.)*

### L-INT — `docs/intake/2026-07-06-lane-int-functional-design.md` §6

- **Q1 — Verdict-package vehicle (DRAFT-INT-1)** → **RULED.**
  > **Recommendation: (a).** The JSON is the machine truth for agent callers and future gates; the
  > mirror block keeps Lane B/operator reads self-contained.
- **Q2 — Exit-0-with-degradation obligation** → **RULED.**
  > **Recommendation: oblige it.** The shrunk-panel-on-exit-0 case is the one degradation the caller
  > currently cannot see at all; an obligation keyed only to exit codes re-inherits ADR-08's coarseness.

### L-GOV — `docs/intake/2026-07-06-lane-gov-functional-design.md` §6

- **Q1 — Pause disposition** → **RULED.**
  > **Recommendation: O2** — lift after GOV-1 executes and `main` is pushed. *(Applied this session.)*
- **Q2 — DRAFT-GOV-1 ratification + the ADR-09/10 flip** → **RULED.**
  > **Recommendation: yes;** the flip is overdue by the convention's own expiry rule and needs only
  > your ruling. *(Applied this session: DRAFT-GOV-1 → ADR-14; ADR-09/10 → Accepted.)*
- **Q3 — Watch location** → **RULED.**
  > **Recommendation: BACKLOG** — living file, survives the story-map schema, zero new files.
  *(Location RULED; the `## Watches` block + seed watches are DRAFT-GOV-2's implementation, not
  ratified this session.)*
- **Q4 — Push-cadence interim rule** → **RULED.**
  > **Recommendation: yes** — it is the minimum that would have prevented §1.2, at zero mechanism
  > cost. *(Adopted: push `main` at every currency-pass close-out and after any ratification merge.)*

---

## FORK_RULING — #1/#24 evidence-method reconciliation → **RULED (a)**

**Operator ruling 2026-07-17: (a) #24 is authoritative for #2; #1 is absorbed.** #24's full-corpus
comparative archaeology is THE evidence path for #2 (the Branch A/B synthesizer decision); #1
(~15-transcript sampling) is absorbed and **leaves per ADR-65**, with its absorption noted in the
`BACKLOG.md` grooming log. Rationale: OQ-1 (above) ruled #1 "comparative" — the same method #24
runs at full-corpus scale, so #24 is the superset. #2's dependency re-points from #1 → #24.

## EPI-1 ruling — **PENDING (addendum on Beat-1 completion)**

Beat 1 (the #24 EPI-1 report + operator ruling = the Epic B event) is **deferred**: the blind
scoring is an operator-time task not yet done. When the operator reports scoring complete, Beat 1
re-arms as its own mini-session; the EPI-1 Branch A/B ruling then enters **this register as an
addendum entry**, and un-gates the Epic-B items (#18/#19/#9 + the v2 resolver). Not done here.

---

## Note — intake §5 label drift (superseded, per the acceptance scope above)

The `docs/intake/archive/2026-07-06-technical-architect-intake.md` §5 "Operator rulings register" paragraph
listed 14 topic **labels** that drifted from the actual §6 rulings: the labels *"F10 visible-author
close"*, *"corpus scope"*, and *"secrets rule"* have **no §6 operator-ruling counterpart** (they
name, respectively, an EPI functional requirement, a settled scope item, and a DOC inputs-forward
item at §3(Q6)/§7 — not §6), while the genuine **L-DOC §6 trio** and **L-CLI §6 Q2** were
under-labeled. Recording the 15 verbatim from §6 (above) resolves the drift by construction. The
§5 labels are superseded by this register for the purpose of the rulings record.
