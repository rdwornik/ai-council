---
last_reviewed: 2026-06-02
status: active
owner: Rob
---

# CLAUDE.md — AI Council
> **Session contract for Claude Code in this repo.** Read on every session start (auto). Single canonical agent-instruction file (≤200 lines). Per ADR-53.
>
> **For universal rules:** read `../.dev-knowledge/protocols/ESSENTIALS.md` and `../.dev-knowledge/protocols/PLAYBOOK.md`.

## 1. First read (session start)

In order, read:
1. This file (you're here)
2. `../.dev-knowledge/protocols/ESSENTIALS.md` — Rob's universal working style
3. `../.dev-knowledge/protocols/PLAYBOOK.md` — universal protocols (only sections relevant to current task)
4. Most recent handoff under `.dev-knowledge/docs/handoffs/` if continuing prior session
5. Last 5 entries of `JOURNAL.md`

If ESSENTIALS or PLAYBOOK are unavailable, proceed with this file alone but flag it.

## 2. Repo identity

- **Name:** `ai-council`
- **Status:** `active`
- **Purpose:** Multi-model AI debate and research CLI tool; produces binding ADRs governing the `Dev/` ecosystem.
- **Owner:** Rob
- **Critical paths:** `src/ai_council/`, `tests/`, `docs/decisions/`, `config/settings.yaml`

## 3. Architecture

See `ARCHITECTURE.md` for the structural model; read it before structural changes (required per ADR-51 — mandatory for every repo).

## 4. Conventions

- **Naming:** snake_case Python; kebab-case markdown; `ADR-NN-topic.md` future ADRs (existing ADRs hyphen-named per ADR-34)
- **Commits:** Conventional Commits — `type(scope): summary` (imperative; body for non-trivial changes)
- **Branches:** `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<scope>`
- **Testing:** `pytest tests/ -m "not integration and not envcheck" -v` (unit suite, no API keys); `pytest -x --tb=short` (quick); `asyncio_mode = auto` in `pyproject.toml`
- **Linting:** `ruff check src/ tests/ --fix`; pre-merge: `.\scripts\check.ps1` (pytest + mypy + ruff)

**Out of scope for this repo:**
- Client/pre-sales data → Obsidian vault
- Cross-ecosystem lessons → `.dev-knowledge/LESSONS.md`
- Curated Council transcripts → `.dev-knowledge/docs/decisions/transcripts/`

## 5. Critical rules

1. Read `.claude/rules/` before making code changes: `code-standards.md`, `python-env.md`, `testing.md`
2. API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `PERPLEXITY_API_KEY`) live in `C:\Users\1028120\Documents\.secrets\.env` — never add keys to a repo-local `.env`
3. Run `.\scripts\check.ps1` (pytest + mypy + ruff) before every merge
4. `LESSONS.md` is append-only — never edit old entries (ADR-29)
5. ADRs are immutable — supersede with a new ADR; never edit in place
6. Config strings (models, prompts, personas, timeouts) live in `config/settings.yaml` — never hardcode
7. Do NOT merge `xai.py` and `deepseek.py` into a single provider — keep separate

## 6. Session start protocol

1. `/boot` (loads skills, memory, recent commits)
2. `git status` — clean working tree?
3. `git log --oneline -5` — recent context
4. Read most recent handoff under `.dev-knowledge/docs/handoffs/` if continuing
5. `pytest --collect-only -q` — test discovery sanity check
6. Run `.\scripts\check.ps1` when ready to merge
7. Wait for Rob's prompt — never improvise

## 7. Slash commands available

User-level (`~/.claude/commands/`):
- `/boot` — load context (skills, memory, recent commits)
- `/session-summary` — generate token-efficient session summary
- `/evolve` — evolution audit: promote/prune/graduate learned rules
- `/codex-review` — invoke Codex review on a staged code diff

Repo-level: none — this repo has no `./.claude/commands/` directory (`.claude/` holds `rules/` only).

## 8. Skills active

User-level (`~/.claude/skills/`):
- `gotchas` — universal dev gotchas (encoding, shell safety, test pitfalls)
- `verify` — domain verification scripts for the ecosystem (run after pytest)

(`boot`/`session-summary`/`evolve`/`codex-review` are **commands**, not skills — see §7.)

Repo-level (`./.claude/rules/`):
- `code-standards.md` — ecosystem code standards
- `python-env.md` — venv, install, async-first guidance
- `testing.md` — pytest + pytest-asyncio standards

Code review: Codex via `/codex-review`; threshold 3+ files for a full review.

## 9. Hooks active

Pre-commit (`.pre-commit-config.yaml`):
- `normalize-headers` — `scripts/normalize_headers.py`; normalizes dated-log headers in `LESSONS.md`/`JOURNAL.md`

Manual pre-merge gate:
- `.\scripts\check.ps1` — pytest + mypy + ruff (run before every merge; not wired to pre-commit)

## 10. Anti-patterns specific to Claude Code in this repo

- **Windows cp1252**: Do not print Unicode chars in Rich progress callbacks — ASCII only
- **google-genai event loop**: `genai.Client(api_key=...)` must be created INSIDE the async method, NOT in `__init__`
- **Interactions API warnings**: suppress `UserWarning` from `client.aio.interactions` at the call site
- **MockProvider ABC**: `async def generate` must exist in class body AND be shadowed by `AsyncMock` in `__init__`
- **pytest-asyncio**: `asyncio_mode = auto` required in `pyproject.toml`
- **Critique template**: Uses `{previous_responses_anonymized}`, not `{previous_responses}`
- **Inbox loop parity**: Features added to interactive CLI must be explicitly mirrored into inbox loop
- **`_anonymize_responses()` shuffle**: Part of blind-voting contract — do not change without an ADR
- **`make_cache_key()` location**: In `src/ai_council/research/merger.py`, NOT `src/ai_council/research/cache.py`
- **Windows /dev/null**: Use `io.StringIO()` for Console mocking in tests, not `open("/dev/null", "w")`

Do NOT:
- Re-add scope-tag enforcement — withdrawn under ADR-46; `validate_scope_tags.py` deleted
- Recreate `CHANGELOG.md` or `BACKLOG_ARCHIVE.md` — removed per ADR-49
- Edit existing `LESSONS.md` entries — append-only per ADR-29
- Add API keys to a repo-local `.env` — global secrets only
- Change Council runtime behavior to fix question-quality problems — fix in `docs/council-question-guide.md`

## 11. Recent ADRs binding here

**Local (`docs/decisions/`):**
- ADR-01: Synthesizer Selection — non-participating model synthesizes; default gemini (Revised 2026-04-30)
- ADR-02: Default Panel Composition — full 5-model default; `--lite` for 3-model (Revised 2026-05-11)
- ADR-03: Blind Voting in Round 2 — `_anonymize_responses()` shuffles; hides provider identity
- ADR-04: Mode System — pick/ideas/judge/research with aliases and auto-detection
- ADR-05: Research Mode Integration — parallel-research code path, file cache, `--deep` opt-in
- ADR-06: Cost Optimization — per-provider tracking; Qwen trial deferred (Revised 2026-05-11)
- ADR-07: Dual Output Paths — superseded by ADR-43 (opt-in target-project routing)
- ADR-08: Research Degradation Alarm — <3 research providers succeed → exit code 3 + alarm banner

**Ecosystem (`.dev-knowledge/docs/decisions/`) binding here:**
- ADR-29: append-only LESSONS; ADR-34: filename conventions; ADR-38: `src/ai_council/` namespace
- ADR-42: handoffs centralized in `.dev-knowledge`; ADR-43: cross-project transcript routing
- ADR-48/49: no CHANGELOG/BACKLOG_ARCHIVE; Conventional Commits; JOURNAL/LESSONS structure
- ADR-51: ARCHITECTURE.md convention (universal); ADR-53: CLAUDE.md as single canonical instruction file
- ADR-59: universal visual pattern (dot-prefix configs, ALL-CAPS canonical, `.code-workspace` sort) — repo conforms; ADR-60: docs/ folder taxonomy (decisions/ + audits/ + archive/, README-seeded)
- ADR-67: AI-Council process operationalization — six-step gated loop; downstream `/council-question` template + gate + `council.return_dir` are ai-council's to implement (not yet built)

> **BACKLOG schema (ADR-41/47 → ADR-64/65/66):** whether the ADR-64/65/66 story-map layout binds child repos or is `.dev-knowledge`-scoped is unresolved upstream (`.dev-knowledge` BACKLOG #20 open). This repo's `BACKLOG.md` retains the ADR-41/47 stream schema pending that decision.

## 12. Section history

- v1.0 (pre-ADR-53) — technical reference document (architecture, commands, design decisions)
- v2.1 (2026-05-19) — ADR-53: retire AGENTS.md; CLAUDE.md becomes substantive single canonical agent-instruction file; technical depth moved to ARCHITECTURE.md
- v2.2 (2026-06-02) — universalization conformance audit: add `last_reviewed` frontmatter (resolves audit.py check #10 WARN); fix §header PLAYBOOK path; reconcile §7/§8 to actual `~/.claude/` + `.claude/` state (`/save` repo-command and `handoff`/`save` skills do not exist; +`/evolve`/`/codex-review`; +`verify` skill; `/review`→`/codex-review`); §10 namespace path `src/research/`→`src/ai_council/research/`; §11 +local ADR-08, +ecosystem ADR-59/60/67, note unresolved backlog-schema scope

---

**Last updated:** 2026-06-02
**Maintained by:** Rob
