# ADR-09: protocols/ as the invocation surface

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `protocols/*.md` shipped (`0966c2a`, #12). No open remainder. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-02
**Status:** Accepted (2026-07-17)

## Context

ai-council has two outward-facing specification documents that tell an operator or a
delegating agent HOW to commission the Council: `council-question-guide.md` (how to frame a
question) and `synthesis-quality-rubric.md` (how synthesis quality is scored). They previously
sat loose at the root of `docs/`.

ADR-60 (hub) gives `docs/` a semantic-role taxonomy — for child code repos: `decisions/`
(OUTPUTS), `audits/` (OUTPUTS), `archive/` (ARCHIVED), optional `diagrams/`. Two consequences:

- These files are neither decisions, audits, archive, nor diagrams — they are invocation
  *specs*, an unmodeled role sitting outside any taxonomy folder.
- As ai-council becomes delegation-ready (commissioned by an external agent), its invocation
  surface should be a first-class, discoverable location — mirroring the hub's own
  `protocols/` directory (`AI_COUNCIL_PROCESS.md`, `PLAYBOOK.md`, `ESSENTIALS.md`).

## Decision

Create a top-level `protocols/` directory holding ai-council's outward-facing invocation
specs, named in SCREAMING_SNAKE to mirror the hub's `protocols/` convention:

- `docs/council-question-guide.md` → `protocols/COUNCIL_QUESTION_GUIDE.md`
- `docs/synthesis-quality-rubric.md` → `protocols/SYNTHESIS_QUALITY_RUBRIC.md`

Moves use `git mv` (history preserved). After this ADR, `docs/` holds **only** `decisions/`,
`audits/`, and `archive/` (+ optional `diagrams/`) per ADR-60.

Living-doc reference updates land in lockstep with the move (this ADR + BACKLOG #12 are one
atomic bundle):
- `.pre-commit-config.yaml` hub hooks `toc-freshness` / `toc-generate`:
  `^docs/council-question-guide\.md$` → `^protocols/COUNCIL_QUESTION_GUIDE\.md$`.
- `ARCHITECTURE.md` Folder Governance table — add a `protocols/` row; `docs/` row notes specs
  live in `protocols/`.
- `CLAUDE.md` §10 anti-pattern reference to `docs/council-question-guide.md`.
- `BACKLOG.md` #10 reference to the rubric path.
- Any other living cross-references (grep both slugs first).

Immutable/point-in-time records that mention the old paths (`docs/audits/*`, `JOURNAL.md`)
are **not** rewritten (ADR-60 Rule 5 — accurate when written).

## Reconciliation with ADR-60 (hub, immutable)

ADR-60 governs the semantic roles of folders **inside** `docs/`. It has no "eject" clause —
moving a file **out** of `docs/` into a new top-level directory is **unmodeled, not
forbidden**. This ADR codifies that ejection locally for ai-council. It does **not** edit,
amend, or supersede ADR-60; the hub `docs/` taxonomy is untouched. A universal
`protocols/`-for-child-repos convention, if ever wanted, is a hub ADR to author separately.

## Alternatives considered

- **Keep the files at `docs/` root.** Rejected: leaves an unmodeled role loose in a taxonomy'd
  folder and gives the invocation surface no first-class home.
- **Use a `docs/protocols/` subfolder.** Rejected: adding a fifth role inside `docs/` would
  edit the ADR-60 child-repo taxonomy. A sibling top-level `protocols/` keeps it intact.
- **Rename in place, no move.** Rejected: naming is not the problem; location and role are.

## Consequences

- ai-council gains a first-class, discoverable invocation surface mirroring the hub.
- `docs/` conforms cleanly to ADR-60 (decisions/audits/archive only).
- One pre-commit hook path + a handful of living cross-refs move in lockstep with the files (this
  bundle), so the hub `toc-*` hooks keep firing at the new path.
- SCREAMING_SNAKE filenames are an intentional, ADR-recorded exception to ADR-34's kebab-case
  markdown rule, matching the hub `protocols/` precedent and the ALL-CAPS governance-doc
  convention (ARCHITECTURE "Key conventions").

## Related

- ADR-60 (hub) — docs/ folder taxonomy (unchanged by this ADR)
- ADR-34 (hub) — file naming (SCREAMING_SNAKE governance-doc exception noted here)
- ADR-67 (hub) — Council process; the `/council-question` template consuming
  `COUNCIL_QUESTION_GUIDE` lives on this surface
- BACKLOG #12 — the implementation task (bundled with this ADR)
