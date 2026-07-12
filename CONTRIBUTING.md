---
last_reviewed: 2026-07-13
reconciled_with: handoff-process@5.7
status: active
owner: Rob
---

# Contributing

<!-- scope: meta -->

Sole contributor: Rob Dwornik. Audience: future Rob + AI agents (Claude Code, Codex)
working on the `ai-council` CLI. Universal working style lives in the hub protocol set
`../.dev-knowledge/protocols/ESSENTIALS.md` + `PLAYBOOK.md` (hub-pointer, never copied
in); this file is the repo-specific contribution contract.

## Branch naming

<!-- scope: meta -->

```
feat/short-description
fix/short-description
docs/short-description
chore/short-description
```

Default branch: `main` (ADR-30).

Branch prefixes are `feat/ fix/ docs/ chore/` (these four only). Commit **types** follow Conventional Commits and additionally include `refactor` and `test` — commit types are **not** branch prefixes. Never commit directly to `main`: branch → `--no-ff` merge.

## Commit style

<!-- scope: meta -->

Conventional Commits. Format: `type(scope): short imperative sentence`

```
docs(adr): add ADR-31 file naming convention
docs(journal): record session close
feat(scope): add scope tag validator enforcement
fix(validator): handle missing HEAD baseline
chore: update dev dependencies
```

Scopes are optional but use the file/folder slug when it clarifies. See recent commits in `git log` for live examples.

**Local caveat (Windows):** keep commit message text free of apostrophes/backticks when committing through a non-PowerShell shell (Git Bash) — quoting mismatches otherwise mangle the message.

### Backlog-id references (forward-only index)

<!-- scope: meta -->

Commit messages **extend** Conventional Commits (they do not replace them) with an optional backlog/ADR reference, so a closure is locatable by id (ADR-65: git is the technical record, forward-indexed via this convention):

```
fix(audit): widen check-8 stamp regex [#42]      # touches backlog item 42
feat(scripts): add backlog validator, closes [#57]   # closing commit for item 57
docs(adr): ADR-65 done-item disposition           # ADR number is itself the index
```

- **Touching** a backlog item: append `[#<id>]` to the summary.
- **Closing** a backlog item: add `closes [#<id>]` (summary or body) — pairs with the item leaving `BACKLOG.md` in the same or a following commit.
- **`closes` vs `advances`:** use `closes [#<id>]` on the commit that **finishes** an item — not `advances [#<id>]`. `advances` records intermediate progress only: the item stays open in `BACKLOG.md` **and** invisible to the closure detector (which keys on `closes`), so it silently accumulates as done-but-open and must be closed manually (this is what forced the manual close of #73). A multi-commit arc may use `advances` along the way, but the commit that completes the work must use `closes`.
- `<id>` is the entry's stable `id:` field (monotonic, never reused — PLAYBOOK §10 schema).
- **Cross-repo references are repo-qualified.** A bare `[#<id>]` denotes a task in THIS repo only. To reference another fleet repo's backlog item, qualify it: `hub#<id>`, `ai#<id>`, `corp#<id>`. (Operator ruling, content-parity inventory D2 / #331 — qualified-refs chosen over a global allocator. Automated enforcement lands with #328; this is the convention it will check.)

This indexes commits **going forward only.** Git history is immutable — **historical commits are never rewritten** (ADR-65). Pre-convention closures are located via the SHAs already embedded in retired entries (preserved in the one-time migration JOURNAL map).

**Enforced by the carried `commit-msg` gate** (where installed): a commit that removes a `- [#id]` task from `BACKLOG.md` without referencing that id (`[#id]` or `closes [#id]`) is rejected. A reworded task (id present before and after) does not trigger. Install the commit-msg stage once per machine:

```
pre-commit install --hook-type commit-msg
```

In this repo the commit-msg gate is the **hub-sourced** `backlog-id-on-close` hook (`repo: ../.dev-knowledge`, pinned `rev`) — not a repo-local script.

**"What's been implemented" query.** Because done tasks **leave** `BACKLOG.md` (ADR-65) and git is the implementation record, the list of completed tasks with their implementing commits is:

```
git log --grep 'closes \[#'
```

This is the detailed implementation history the active file deliberately does not carry.

## Pre-commit setup

<!-- scope: meta -->

Install once per machine:

```
pip install pre-commit
pre-commit install
```

Run manually at any time:

```
pre-commit run --all-files
```

## Validators

<!-- scope: meta -->

Pre-commit hooks (`.pre-commit-config.yaml`) — the roster is repo-local; hub-only hooks do not ship here:

| Hook | Stage | What it does |
|------|-------|--------------|
| `normalize-headers` | pre-commit | Normalizes dated-log entry headers in `LESSONS.md` / `JOURNAL.md`. |
| `floor-hash-verify` | pre-commit | Verifies `.claude/CLAUDE-FLOOR.md` matches its `.sha256` sidecar. |
| `canonical_freshness` | pre-commit | `last_reviewed` A2 gate; FAIL blocks a commit on a canonical doc edited since its last review. |
| `validate-audit-casing` | pre-commit | ADR-101 R4 audit-filename casing gate (fleet ruling d1; casing-only carry). |
| `validate-backlog` | pre-commit | Validates the `BACKLOG.md` story-map structure (ADR-66); ADR-78 floor twin of the hub validator. |
| `toc-freshness` / `toc-generate` | pre-commit | TOC freshness for `protocols/COUNCIL_QUESTION_GUIDE.md` (hub-sourced, `repo: ../.dev-knowledge`, pinned `rev`). |
| `ruff` | pre-commit | Lint gate — `ruff check` E/F/I/W (consumer-owned, `astral-sh/ruff-pre-commit` pinned; config in `pyproject.toml`). Blocks on violations. |
| `backlog-id-on-close` | commit-msg | Requires `[#id]` / `closes [#id]` when a commit removes a `- [#id]` task (hub-sourced). |
| `block-ff-push` | pre-push | Refuses a direct-to-`main` / true-FF push; a `--no-ff` merge passes (hub-sourced). Activate once: `pre-commit install --hook-type pre-push`. |

**Pre-merge gate (manual, not wired to pre-commit):** `.\scripts\check.ps1` — `pytest` + `mypy` + `ruff`. Run before every merge (CLAUDE §5). Unit suite without API keys: `pytest tests/ -m "not integration and not envcheck" -v`.

**Conformance (read-only, external):** `.dev-knowledge/scripts/audit.py` audits this repo against the canonical standard (ADR-38 A6) out-of-band; it never writes here (Layer-2 invariant, ADR-28).

## ADR process

<!-- scope: meta -->

Decisions that bind future sessions live in `docs/decisions/ADR-NN-topic.md`.

- Numbering: next integer after highest existing ADR
- Filename: `ADR-NN-short-kebab-topic.md` (hyphen-named per ADR-34)
- Status values: `Accepted | Superseded | Withdrawn`
- Minor prescription drift → amend in-place (add dated `## Amendment YYYY-MM-DD` section)
- Intent change or reversal → new ADR or AI Council reopen

Local tool-design decisions live here (`docs/decisions/`); see ADR-01 through ADR-08 for style reference. Cross-repo/ecosystem decisions are authored in `.dev-knowledge`; a Council debate that informs one routes its transcript here via `target-project` (ADR-43).

## Handoff process

<!-- scope: meta -->

Protocol: the hub methodology protocol `HANDOFF_PROCESS.md` (read at the hub `../.dev-knowledge/protocols/` set; hub-pointer, never copied into a consumer) — **v5**. CC owns the handoff: it emits a lean **residual** + a **probe manifest** under `docs/handoffs/<slug>/`, and a fresh browser chat boots from the thin hub protocol `HANDOFF_BOOT.md`. The ADR-36 read-only contract holds: a handoff never writes to a target repo.

Handoffs **centralize in `../.dev-knowledge/docs/handoffs/`** (ADR-42/60) — this repo carries no local `docs/handoffs/`. Continuing a prior session: read the most recent bundle there.

`BACKLOG.md` (root): cross-session pending items per ADR-41. Universal mandate (ADR-38 amendment A5 — every repo, no tier gating). Review before chartering a new session.

## Definition of done (session close)

<!-- scope: meta -->

The hub methodology protocol `DEFINITION_OF_DONE.md` (read at the hub `.dev-knowledge/protocols/` set; hub-pointer, never copied into a consumer) is the single source of truth for what "done" means at session close (ADR-85): a session that produces commits adds a `JOURNAL.md` entry naming ≥1 commit SHA from this arc (**hard-gated**), and should update `BACKLOG.md` with a structural marker (**advisory** in v1). It is enforced **mechanically and deterministically** by the carried session-end Stop-hook (no LLM in the gate) — and the only escape is `/override [reason]`. The other living docs (`ARCHITECTURE`, `VISION`, `LESSONS`, this file) are "update when materially affected", not per-session-gated. Pointer only — the rules live in that file, not here (resident copies drift).

In this repo the concrete Stop-hook is `scripts/session_end_backpressure.py`.
