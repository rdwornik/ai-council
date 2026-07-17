# ADR-14: ADR lifecycle states

**Date:** 2026-07-17
**Status:** Accepted (2026-07-17)

## Context

ai-council's ADR statuses drifted: ADR-09 and ADR-10 sat `Proposed` from 2026-07-02 while both
were long since implemented and load-bearing (`protocols/` shipped; `--return-dir` shipped). No
convention governed (a) when `Proposed` must flip to `Accepted`, (b) the line between a
`Revised (dated)` amendment and a `Superseded` replacement, or (c) keeping the ADR file header in
sync with the `docs/decisions/README.md` index row. This ADR ratifies **DRAFT-GOV-1** (functional
design in `docs/intake/2026-07-06-lane-gov-functional-design.md` §4), the lifecycle convention
that closes those gaps. Ratified per the operator's GOV-1 ruling (rulings register
`docs/intake/2026-07-17-gov1-rulings-register.md`, L-GOV §6 Q2); the in-place status-line edits it
authorizes are permitted by ADR-94 (ratification status-line exception).

## Decision

**States:** `Proposed` → `Accepted` → (`Revised (dated)`)\* → `Superseded (by ADR-X)`.

- **Proposed** — shape ratified for authoring, not yet load-bearing. **Expiry rule:** once the
  decision is implemented and has survived one review cycle, the status **must** flip at the next
  currency pass. `Proposed` is a waiting room, not a resting state. (Witnessed failure mode:
  ADR-09/10, `Proposed` since 2026-07-02 while long since load-bearing.)
- **Accepted** — the operator's ratification ruling, recorded in a JOURNAL entry naming the
  ratification SHA (codifies the witnessed ADR-11/12 pattern).
- **Revised (dated)** — a dated amendment **appended** to the same file; the original decision text
  stays intact. Legal only for parameter-level changes that do **not** invert the decision (panel
  size, defaults, cost tables). This reconciles the live `Revised` statuses (ADR-01/02/06) with
  CLAUDE.md §5 "ADRs are immutable — supersede, never edit": *immutable* means the decision record
  is never rewritten; it does not forbid dated, append-only amendments. Grandfathered: ADR-01/02/06
  conform as-is.
- **Superseded (by X)** — the decision is inverted or replaced; requires a new ADR; the old file
  gains only the status line + pointer (ADR-07 → ADR-43 is the conforming precedent).

**Sync invariant.** Status lives in exactly two places — the ADR file header **and** its
`docs/decisions/README.md` index row — and any status change touches **both in the same commit**.
(Testable; a future mechanization candidate, not mechanized by this ADR.)

**Pre-number discipline.** Unratified designs carry `DRAFT-<lane>-<n>` inside intake/audit
documents; real numbers are assigned by the operator at ratification only (ADR-13 stays reserved by
the crux-resolver draft; this ADR took the next free number, ADR-14).

## Consequences

- **Immediate disposition (this ratification):** ADR-09 and ADR-10 flip `Proposed` → `Accepted` at
  GOV-1 execution — both implemented, load-bearing, and past one review cycle (operator ruling; no
  new evidence needed). Header + index rows flipped in the same commit per the sync invariant.
- The expiry rule makes `Proposed` a bounded state: any future implemented-and-reviewed ADR flips at
  the next currency pass rather than drifting.
- The `Revised (dated)` state resolves the apparent conflict between the live `Revised` ADRs and the
  immutability rule without any rewrite.
- DRAFT-GOV-2 (recurring currency pass + watch protocol) is **not** ratified here — it remains a
  draft in the L-GOV lane doc §4.

## Related

- **DRAFT-GOV-1** — `docs/intake/2026-07-06-lane-gov-functional-design.md` §4 (the ratified design)
- **ADR-94** (hub) — ratification status-line-in-place exception (authorizes the header flips)
- Rulings register — `docs/intake/2026-07-17-gov1-rulings-register.md` (L-GOV §6 Q2 = the ruling)
- ADR-09, ADR-10 — the two ADRs flipped to `Accepted` under this ADR's immediate disposition
