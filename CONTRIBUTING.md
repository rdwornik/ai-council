---
last_reviewed: 2026-06-02
status: active
owner: Rob
---

# Contributing

<!-- scope: meta -->

Sole contributor: Rob Dwornik. Audience: future Rob + AI agents (Claude Code, Codex)
working on the `ai-council` CLI. Universal working style lives in
`../.dev-knowledge/protocols/ESSENTIALS.md` + `PLAYBOOK.md`; this file is the
repo-specific contribution contract.

## Branch naming

Off `main` (ADR-30): `feat/<topic>`, `fix/<topic>`, `docs/<scope>`, `chore/<scope>`.
One branch per unit of work; merge `--no-ff` after the pre-merge gate passes.

## Commit style

Conventional Commits — `type(scope): summary` (imperative mood; body for non-trivial
changes). Types: `feat / fix / docs / chore / refactor / test`. Keep commit message
text free of apostrophes/backticks when committing through a non-PowerShell shell.

### Backlog-id references (forward-only index)

`BACKLOG.md` follows the ADR-66 story-map; tasks carry a `[#id]`. When a commit closes
a backlog task, reference it in the message (`closes [#id]`) so the work is locatable by
id via `git log --grep "closes \[#"`. Git history is the implementation record (ADR-65);
done tasks leave the file.

## Pre-commit setup

`pip install pre-commit && pre-commit install`. Active hook:

- `normalize-headers` — `scripts/normalize_headers.py`; normalizes dated-log headers in
  `LESSONS.md` / `JOURNAL.md`.

## Validators

- **Pre-merge gate (manual):** `.\scripts\check.ps1` — `pytest` + `mypy` + `ruff`. Run
  before every merge (CLAUDE §5). Unit suite without API keys:
  `pytest tests/ -m "not integration and not envcheck" -v`.
- **Conformance (read-only, external):** `.dev-knowledge/scripts/audit.py` audits this
  repo against the seven-file canonical standard (ADR-38 A6) out-of-band; it never writes
  here (Layer-2 invariant, ADR-28).

## ADR process

Local tool-design decisions live in `docs/decisions/ADR-NN-topic.md` (ADR-NN numbering,
hyphen-named per ADR-34). ADRs are immutable — supersede with a new ADR; never edit in
place. Cross-repo/ecosystem decisions are authored in `.dev-knowledge`; a Council debate
that informs one routes its transcript here via `target-project` (ADR-43).

## Handoff process

Handoffs centralize in `.dev-knowledge/docs/handoffs/` (ADR-42/60) — this repo carries no
`docs/handoffs/`. Continuing a prior session: read the most recent bundle there.
