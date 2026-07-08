# docs/intake/

Forward-looking **intake and functional-design** artifacts — the "what to build"
layer that precedes implementation. Distinct from `docs/audits/` (retrospective
"what is / what happened" analysis) and `docs/decisions/` (ratified ADRs).

Seeded 2026-07-08 during Wave-1 onboarding (fleet-consistency census, finding f.6):
the technical-architect intake doc plus the five ADR-98 lane functional-designs were
relocated here from `docs/audits/`, where they had been misfiled. Genuine audit
artifacts (e.g. the 2026-07-05 fleet-recon reconciliation, the code-quality-audit's
refactoring-guide companion) stay under `docs/audits/`.

Per ADR-60 (docs/ taxonomy) and the Phase-1 universal `docs/intake/` convention
(hub #280). Note: ai-council has **no `docs/handoffs/`** — per ADR-60/ADR-42, handoffs
centralize in `.dev-knowledge`, not in child repos. The census's `docs/handoffs`
target for ai-council is filed back to the hub as NEEDS-RULING (see the Wave-1
runbook gap-notes), not created here.

## Contents
- `YYYY-MM-DD-technical-architect-intake.md` — technical-architect entry-point document
- `YYYY-MM-DD-lane-*-functional-design.md` — per-lane (ADR-98) functional designs
