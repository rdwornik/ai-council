# AGENTS.md

> **Canonical governance contract for this repo.** Cross-tool standard — read by Claude Code, Codex, Cursor, Aider, and any other LLM-based agent operating here. LLMs advise; hooks/tests enforce.
>
> **Template version:** 2026-04-24 — see `.dev-knowledge/templates/AGENTS-md-template.md` for source.

## 1. Read first

Before doing anything in this repo, read:
- `C:\Users\1028120\Documents\Dev\.dev-knowledge\protocols\ESSENTIALS.md` — Rob's daily working style + protocols
- `C:\Users\1028120\Documents\Dev\.dev-knowledge\protocols\PLAYBOOK.md` — universal rules: prompt format, file types, Council lifecycle, etc.

This AGENTS.md covers **only** what's specific to this repo. Universal rules live in `.dev-knowledge/`.

If `.dev-knowledge/` is unavailable, proceed with this AGENTS.md alone but flag missing-universal-rules in session output.

## 2. Repo identity

- **Name:** `ai-council`
- **Scale tier:** `M`
- **Purpose:** Multi-model AI debate and research CLI tool; produces binding ADRs governing the `Dev/` ecosystem.
- **Status:** `active`
- **Owner:** `Rob`
- **Critical paths:** `src/ai_council/`, `tests/`, `docs/decisions/`, `config/settings.yaml`

## 3. Architecture

`cli.py` (Click entry `council`) → `orchestrator.py` / `runner.py` → `debate.py` + `providers/` (five debate providers: Claude, Gemini, GPT, Grok, DeepSeek) → `synthesis.py`. A separate `research/` subsystem runs five research providers in parallel for `--mode research`. `mode_detector.py` routes between pick / ideas / judge / research; `metrics.py` tracks per-call token counts and cost; `output.py` handles the dual canonical + opt-in target-project output paths. Config is loaded via `config/config_loader.py` from `config/settings.yaml` (single source of truth).

**Key dependencies:**
- `click`, `rich` — CLI framework and console output
- `pyyaml`, `python-dotenv`, `python-frontmatter` — config and inbox parsing
- `anthropic`, `openai`, `google-genai` — provider SDKs (Grok and DeepSeek use the OpenAI-compatible client with a custom `base_url`)

**Enforcement (what's mechanically guarded):**
- pre-commit hook: `scripts/normalize_headers.py` normalizes dated-log entry headers in `LESSONS.md` / `JOURNAL.md`
- `scripts/check.ps1` — pre-merge check: pytest + mypy + ruff (must pass before merge)

**Advisory (LLM should respect but not enforced by tooling):**
- Single-responsibility modules; namespace package under `src/ai_council/` (per ADR-38)
- Dataclasses over raw dicts for data shapes
- Click CLI, Rich for console output
- Logging not `print` (except Rich console output in the CLI layer)
- Config strings (models, prompts, personas, panels, timeouts) live in `config/settings.yaml`; never hard-code
- Do NOT merge the OpenAI-compatible providers (`xai.py`, `deepseek.py`) — keep separate

## 4. Conventions

**Filenames:**
- snake_case for Python; kebab-case + lowercase for markdown
- Future ADRs use underscore convention `ADR-NN_topic.md` (per ADR-34); the 7 existing kebab-case ADRs (`ADR-01-synthesizer-selection.md` …) are grandfathered per ADR-29 — do not rename

**Branches:**
- `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<scope>`

**Commits:**
- Conventional Commits — `type(scope): summary` (imperative; body for non-trivial changes); per ADR-48/49 the JOURNAL `Changes:` line plus git history is the change record

**Testing:**
- `pytest` + `pytest-asyncio` (`asyncio_mode = auto`)
- Unit suite: `pytest tests/ -m "not integration and not envcheck" -v` (362 unit tests, no API keys needed)
- Quick local: `pytest -x --tb=short`
- Integration tests require 2+ provider API keys in `.env`

**Linting:**
- `ruff check src/ tests/ --fix`
- Pre-merge: run `.\scripts\check.ps1` (pytest + mypy + ruff together)

## 5. Tools active in this repo

**Code review:**
- Codex (OpenAI) — invoked via `/review`; threshold 3+ files for a full review.
- For full Codex review configuration (severity tiers, review modes, output format), see `.dev-knowledge/templates/codex-review-config-template.md` — Scale M section applies here.

**Architecture enforcement:**
- No Tach in this repo. Module boundaries are advisory and reviewed by humans + LLM agents.

**Pre-commit hooks:**
- `normalize-headers` — local hook running `scripts/normalize_headers.py` on `LESSONS.md` / `JOURNAL.md` (converts `## YYYY-MM-DD` dated entries to `###`; idempotent)
- No scope-tag validator. Scope-tag enforcement was withdrawn under the ADR-46 demotion (Council Simplification 2026-05-16); `validate_scope_tags.py` has been deleted — do not re-introduce.

**Other:**
- API keys loaded globally from `C:\Users\1028120\Documents\.secrets\.env` via the PowerShell profile (`keys list` / `keys set` / `keys reload`). Do not add keys to a repo-local `.env`.

## 6. Things this repo gets wrong (gotchas)

**Rules location:** `.claude/rules/` — this repo uses repo-rule files rather than a `.claude/skills/gotchas/` skill. Read these before changes:
- `.claude/rules/code-standards.md` — ecosystem code standards
- `.claude/rules/python-env.md` — venv, install, async-first guidance
- `.claude/rules/testing.md` — pytest + pytest-asyncio standards

Recurring traps (see CLAUDE.md "Gotchas" section for the full list):
- Windows cp1252: never print Unicode in Rich progress callbacks — ASCII only
- `google-genai`: `genai.Client(api_key=...)` must be constructed *inside* the async method, not in `__init__`, or it binds to the wrong event loop
- Interactions API: emits `UserWarning` on every access — suppress at the call site
- `pytest-asyncio` requires `asyncio_mode = auto` in `pyproject.toml`
- `_anonymize_responses()` shuffle order is part of the blind-voting contract — do not change without an ADR
- Inbox loop is a separate code path from the interactive CLI; features added to the interactive path must be explicitly added to the inbox loop too

## 7. Council decisions binding here

ADRs in `docs/decisions/`. Active list (one-liner each):

- **ADR-01** Synthesizer Selection — non-participating model synthesizes; default gemini (Revised 2026-04-30).
- **ADR-02** Default Panel Composition — full 5-model panel is the default; `--lite` for 3 models (Revised 2026-05-11).
- **ADR-03** Blind Voting in Round 2 — `_anonymize_responses()` shuffles + labels "Proposal A/B/C" to hide provider identity.
- **ADR-04** Mode System — pick / ideas / judge / research with aliases and auto-detection.
- **ADR-05** Research Mode Integration — separate parallel-research code path, file cache, `--deep` opt-in.
- **ADR-06** Cost Optimization Strategy — per-provider cost tracking; Qwen trial deferred/abandoned (Revised 2026-05-11).
- **ADR-07** Dual Output Paths — superseded by ADR-43 (opt-in `target-project` routing replaces always-on secondary write).

See `docs/decisions/README.md` for the full index. Debate transcripts in `docs/decisions/transcripts/`.

Ecosystem ADRs in `.dev-knowledge/docs/decisions/` also bind this repo as a child:
- **ADR-29** (append-only LESSONS), **ADR-34** (filename conventions), **ADR-38** (`src/ai_council/` namespace), **ADR-42** (handoffs centralized in `.dev-knowledge`), **ADR-43** (cross-project transcript routing), **ADR-48 / ADR-49** (no CHANGELOG; no BACKLOG_ARCHIVE; Conventional Commits; JOURNAL / LESSONS structure), **Council #28** (AGENTS.md required at Scale M+).

## 8. Out of scope

Things that explicitly do NOT belong in this repo:
- Client-specific data → Obsidian vault
- Dev methodology / cross-repo lessons → `.dev-knowledge`
- Curated Council transcripts → `.dev-knowledge/docs/decisions/transcripts/`. Operational metrics and per-run transcripts stay in `ai-council/output/` (gitignored).

## 9. Session start checklist

When starting a Claude Code session here, check:

1. `git status` — clean working tree?
2. `git log --oneline -5` — recent context
3. Read `JOURNAL.md` — last few entries
4. Read most recent handoff under `.dev-knowledge/docs/handoffs/` (handoffs are centralized there per ADR-42; this repo has no `docs/handoffs/`)
5. `pytest --collect-only -q` — test discovery works (sanity)
6. Run `.\scripts\check.ps1` for the full pre-merge check (pytest + mypy + ruff) when ready to merge

If any check fails → stop and ask Rob before proceeding.

## 10. Do NOT

Things tried and explicitly rejected (with rationale):

- **Re-add scope-tag enforcement** — withdrawn under the ADR-46 demotion (Council Simplification 2026-05-16); `validate_scope_tags.py` and its pre-commit hook are deleted, consistent with ADR-48.
- **Recreate `CHANGELOG.md`** — removed per ADR-49; git history (Conventional Commits) plus JOURNAL `Changes:` lines is the change record.
- **Recreate `BACKLOG_ARCHIVE.md`** — removed per ADR-49; significant abandoned items become lightweight decision-notes in `docs/decisions/`.
- **Edit existing `LESSONS.md` entries** — append-only per ADR-29.
- **Amend ADR-48 or ADR-49 without a dedicated Council debate** — amendments require their own debate.
- **Change Council runtime behavior to fix a question-quality problem** — author-facing fixes belong in `docs/council-question-guide.md`, not in `src/ai_council/`.
- **Merge `xai.py` and `deepseek.py` into a single "OpenAI-compatible" provider** — keep separate per CLAUDE.md.

Anti-patterns specific to this repo:
- Adding API keys to a repo-local `.env` — keys live in the global secrets file only
- Printing Unicode in Rich progress callbacks on Windows (cp1252 trap)
- Constructing `genai.Client` in `__init__` — must be inside the async method
- Adding features to the interactive CLI path without mirroring them into the inbox loop

---

**Last updated:** 2026-05-17
**Maintained by:** Rob
