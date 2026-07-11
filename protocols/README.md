# protocols/ — project-local Council-domain docs

**Scope marker (hub BACKLOG #314).** This folder holds **project-local,
Council-domain** protocol docs — operational contracts for the ai-council CLI
tool itself:

- `COUNCIL_INVOCATION_CONTRACT.md` — CLI invocation lanes / flags
- `COUNCIL_QUESTION_GUIDE.md` — how to write a good Council question (TOC-gated)
- `SYNTHESIS_QUALITY_RUBRIC.md` — synthesis scoring rubric

**Methodology protocols are NOT here.** The universal methodology protocols
(`ESSENTIALS.md`, `PLAYBOOK.md`, and the rest of the hub `protocols/` set) are
**hub-pointer only** — read them at `../.dev-knowledge/protocols/`, never copied
into this repo. `CLAUDE.md` §1 points at them directly by design.

This is the "marked" half of #314's hub-pointer-vs-local split: the folder a
reader lands in first now states, in-band, that it is intentionally
domain-scoped and carries no methodology-protocol copies.

> Note: the hub's `protocols/AI_COUNCIL_PROCESS.md` is a **different** artifact —
> ecosystem governance for the six-step gated Council *process* (ADR-67) — not a
> copy of these tool-operational docs. No content overlap.
