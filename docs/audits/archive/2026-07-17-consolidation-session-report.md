# Consolidation Session Report — 2026-07-17

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — GOV-1 #31 delivered; ADR-09/10 Accepted; DRAFT-GOV-1 ratified as ADR-14. No open remainder (#31 struck). [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Scope:** Day session following the 2026-07-16 night batch. Executed **Beat 2** (GOV-1 consolidation,
gate G1→G2), **Beat 3** (Leg C unlocks #29/#30), and the **EPI-1 workspace relocation**. **Beat 1**
(EPI-1 report + ruling) was **deferred** by operator decision. Plan of record:
`~/.claude/plans/day-session-shiny-wirth.md` (approved). This report is the session's standing record
(major-batch reports persist audits-class; precedent `2026-07-09-qa-lived-exercise.md`).

## Operator decisions (this session)

- **Beat 1 DEFERRED.** The blind scoring sheet was untouched (0/40); per ruling **r3** CC cannot score.
  Rationale: G2/pause-lift gates *all* build work; EPI-1 gates only the Epic-B trio — an operator-time
  dependency (scoring) must not block a ready mechanical unblock. Beat 1 re-arms as its own mini-session
  when scoring completes; the EPI-1 ruling then enters the rulings register as an **addendum**.
- **FORK_RULING (#1/#24) = (a):** #24's full-corpus archaeology is authoritative for #2; **#1 absorbed**.
- **EPI-1 relocation:** approved to `docs/audits/2026-07-17-epi1-archaeology/`, gitignored-while-active.
- **Codex producer-lane (#30):** CC implements Codex's design + terra read-only review (see §Beat 3).

## Beat 2 — GOV-1 consolidation (#31, G1→G2) · merge `34be9aa` (pushed)

| Item | Result |
|---|---|
| 15 lane-doc §6 rulings + FORK_RULING(a) → RULED | `docs/intake/archive/2026-07-17-gov1-rulings-register.md` (`ec406a4`) |
| ADR-09/10 → Accepted (header **and** index, same commit) | `85a692f` |
| DRAFT-GOV-1 ratified → **ADR-14** (ADR-13 reserved) | `85a692f` |
| CLAUDE.md §11 extended ADR-08 → 09/10/11/12+14; re-stamped | `8bde22d` |
| VISION:25 reconciled to ADR-43/ADR-10; re-stamped | `2ae0b99` |
| CONTRIBUTING: re-read caught real staleness (ADR status-values vs ADR-14); fixed + re-stamped | `4629d63` |
| BACKLOG: #1 absorbed into #24; #2 re-pointed #1→#24 | `8ed4012` |
| **Feature-work pause LIFTED (O2)**; `main` pushed | JOURNAL `5e76705`; merge `34be9aa` |

**Closure:** #31 Done-when met; rulings register carries 15 + fork (EPI-1 = PENDING addendum); ADR-09/10
Accepted in header+index; ADR-14 exists+indexed; canonical re-stamps genuine (all 2026-07-17);
`validate-backlog` OK; ship-gate green modulo the pre-existing #20 mypy exclusion (444 passed).

## Beat 3 — Leg C unlocks (post-pause)

- **#29** (F12 stale pin) · merge `b12666b`, `closes [#29]`. `config/settings.yaml:428` +
  `research/providers/grok_research.py:32` default → `grok-4.20-0309-reasoning`. **Live health check:
  RESOLVED** (model responded at the x.ai API); 11 grok research tests pass.
- **#30** (DOC-3 secrets rule) · merge `220e79a`, `closes [#30]`. Empty API-key env var now reads as
  **absent, LOUDLY**, and reloads `.env`/config — closing the `cli.py` `override=False` hazard where an
  empty var shadowed the real `.env` value. `_strip_empty_api_keys(config)` derives expected key-envs
  from debate models + research providers. **2 unit tests** on the hazard path (446 passed total; no new
  mypy errors; ruff clean). **Producer-lane note:** the plan named Codex as producer, but Codex is a
  **read-only reviewer** on this machine by a hub-owned global policy (`~/.codex/AGENTS.md`) — and its
  Windows write-sandbox also fails. Per operator ruling, #30 ran as **CC-implements-Codex's-design +
  terra read-only review** (`codex exec review`: **no Critical/High**; one non-blocking coverage note).
  Machine gotcha recorded; hub feedback filed (`docs/intake/2026-07-17-hub-feedback-codex-producer-lane.md`).

## EPI-1 workspace relocation (housekeeping)

Moved out of the runtime `output/` dir to its governance home
`docs/audits/2026-07-17-epi1-archaeology/` (+ sealed-key & judge as siblings), **gitignored while
scoring is active** (blind-integrity + audit immutability). `.gitignore` block: `02ead5f`. **Seal
finalization:** once scored and the #24 report records the un-blinding, the seal expires — key + judge
are un-ignored and committed with the finalized evidence bundle.

## What's next (operator)

1. **Beat-1 mini-session:** score the 40 blind items per
   `docs/audits/2026-07-17-epi1-archaeology/OPERATOR-SCORING-README.md` → run #24 (report + ruling =
   the Epic-B event, un-gating #18/#19/#9 + the v2 resolver). The EPI-1 ruling enters the rulings
   register as an addendum.
2. **Hub:** carry `docs/intake/2026-07-17-hub-feedback-codex-producer-lane.md` to the hub (EPIC-H).

## Session SHAs

Beat 2: `ec406a4 85a692f 8bde22d 2ae0b99 4629d63 8ed4012 5e76705` → merge `34be9aa`.
Beat 3: `da8549b`→`b12666b` (#29); `7e6a5e3`→`220e79a` (#30).
Close-out: `02ead5f` (relocation) · `321fa76` (hub feedback) · this report · JOURNAL anchor.
