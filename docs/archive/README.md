# archive/ — Pending-Classification Zone

Per ADR-60 amendment 2026-05-27 (in `.dev-knowledge/docs/decisions/`).

Holding zone for artifacts whose destination isn't yet decided. Reviewed periodically; each item is either:

- deleted (git history retains it), or
- promoted to `decisions/`, `audits/`, `diagrams/`, or authored into an ADR.

Not a dumping ground — a triage queue. If something sits here across two reviews with no decision, default to deletion.

## Child-repo taxonomy reminder

This is a child code repo. Its `docs/` carries `decisions/` + `audits/` + `archive/` (+ `diagrams/` where applicable). It does **not** carry `handoffs/` (centralized in `.dev-knowledge/docs/handoffs/`), `research/`, or `council-questions/`.

## Current contents

(Empty — folder seeded for the pending-classification zone at the 2026-05-27 taxonomy-simplification session.)

## Naming convention

`YYYY-MM-DD-{descriptive-slug}.md`

## How to review

1. Open the file. Skim for what's still actionable.
2. If actionable → `git mv` to the correct live folder (preserves history).
3. If superseded / one-shot value already extracted → `git rm`.
4. If still genuinely "don't know" after two passes → `git rm` (the periodic-review threshold).
