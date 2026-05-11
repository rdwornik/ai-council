# Journal — ai-council

## 2026-05-11 — Cross-project transcript routing (feat/transcript-routing)

**Did:**
- Implemented opt-in, config-driven per-invocation transcript routing for all 4 modes
- Added `target_projects` map to `config/settings.yaml` + `AppConfig` loader with validation
- Created `src/ai_council/routing.py`: `TargetResolver` + `RoutingError` (fail-loud on unknown names)
- Extended `inbox.py` `parse_file` to accept optional resolver, resolve `target-project` frontmatter at parse time
- Added `target_paths: list[Path]` parameter to `save_to_file` and `save_research_to_file` — auto-mkdir, best-effort mirror
- Added `--target-project` Click flag (multiple=True) to CLI, wired through RunRequest → orchestrator → output
- 6 commits on branch `feat/transcript-routing`; 349 tests pass; ruff at pre-existing 17 errors baseline

**Architecture decisions:**
- Names dynamic (frontmatter / flag), paths static (settings.yaml) — two-layer model per spec
- Single `TargetResolver` called from both CLI flag path and inbox frontmatter path — no forked logic
- Canonical write always first (hard); mirror writes best-effort with logging
- Existing `secondary_dir` behavior unchanged — coexists with new `target_paths`

**Next:**
- `.dev-knowledge/protocols/ESSENTIALS.md` "Council output convention" section update — separate `.dev-knowledge` session
- Await operator confirmation to merge `feat/transcript-routing` → main

## 2026-05-09 — Audit-sync governance closure (F-01, F-02)

**Did:**
- Verified prior commit `62c1f7d` (config/settings.yaml grok model `grok-4.20 → grok-4.3`) matches Stage 3 expected pattern; commit was made by a prior session, not this one
- Created `VISION.md` (tier M, ADR-33 Lite: Mission / Scope / Relationships / Lifecycle)
- Configured lessons discovery in `CLAUDE.md` (`DEV_KNOWLEDGE_PATH` env var per ADR-35)
- Updated CHANGELOG

**Result:** F-01 + F-02 closed. Baseline 310/310 tests passing. Branch `docs/audit-sync-2026-05-09` ready for review and merge (3 commits ahead of main).

**Next:** return `09_EXECUTION_EVIDENCE.md` to .dev-knowledge for review. Await ADR-40 recalibration before tackling F-03 (BACKLOG.md) and F-04 (ARCHITECTURE.md).

### 2026-04-30 | ADR-38 migration: src/ → src/ai_council/
- Moved all 34 source files under `src/ai_council/` via `git mv` (history preserved); rewrote 73 internal imports in src/, 83 imports + 56 mock.patch string literals in tests/
- Updated pyproject.toml: added `[build-system]` (`setuptools.build_meta`), `where=["src","."]` for packages.find, new entry points, coverage paths; deleted pytest.ini (consolidated into `[tool.pytest.ini_options]`)
- 310 unit tests pass, identical to pre-migration baseline; zero functional changes

### 2026-04-24 | Fix research providers (Gemini 404, OpenAI mini 400)
- Gemini research: `gemini-2.5-pro-preview-05-06` → `gemini-2.5-pro` (preview was not yet released)
- OpenAI mini: added `tools=[{"type": "web_search_preview"}]` to Responses API call (deep research models require at least one search tool)
- Full smoke test: Perplexity + Gemini both completed; OpenAI mini job accepted + completes (~3min for simple queries, may be transient-fail on complex topics)
- 255 tests passing

### 2026-03-29 | Sonnet 4.6 synthesizer + mypy CI
- Added `claude-sonnet` provider; set as default synthesizer (5x cheaper than Opus)
- mypy CI enforcement via `scripts/check.ps1` (pytest + mypy + ruff, 0 errors)
- Archived code review reports to `docs/archive/`
- 255 tests

### 2026-03-28 | Retry logic + graceful degradation
- Error classification (`classify_error()`), `was_retry` tracking
- Specific healthcheck messages per provider failure mode
- `RunPolicy` (retry_on patterns, min_panel_size) decoupled from debate logic
- 231 → 255 tests after provider unit tests + orchestrator extraction
- Next: Sonnet synthesizer, Qwen trial

### 2026-03-25 | Research mode
- Shipped 4 research providers: Perplexity sonar-pro, o4-mini-deep-research, o3-deep-research, Gemini+Search
- Progressive Rich display, file cache (7-day TTL), result merger + LLM summarizer
- `--deep` flag for o3-deep (45 min, $10+); `--no-cache` bypass
- 35 new research unit tests

### 2026-03-22 | Mode system (pick/ideas/judge)
- Four debate modes with per-mode prompts and persona directives
- Auto-detection via cheap LLM call with 5s interactive confirm
- `-M` short flag (was `-m`, conflicted with `python -m`)
- 37 new mode unit tests

### 2026-03-20 | Default panel update + prompt upgrades
- Default panel: Claude + Gemini + OpenAI (was Claude + Gemini + DeepSeek)
- Round 1: structured decision framework; Round 2: steelmanning + hidden assumptions
- Synthesis: argument quality weighting + blind spot detection
- Fixed Gemini event loop crash (fresh `genai.Client()` per call)

### 2026-03-15 | Phase 1 foundation
- Multi-model debate pipeline: Claude, Gemini, GPT, Grok, DeepSeek
- Panel system, persona injection, blind voting (Round 2 anonymization)
- Non-participating synthesizer selection
- Inbox batch mode with frontmatter overrides
- Health checks at startup; cost tracking per debate
- 72 tests; CHANGELOG v1.0.0
