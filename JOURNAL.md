# Journal — ai-council

## 2026-05-12 — Scrum-master review implementation (.dev-knowledge strażnik)

**Did:**
- Implemented 9 of 10 findings from `.dev-knowledge` scrum-master review (2026-05-12)
- C1: retired `tasks/todo.md` (255 vs 362 test stale + March 2026 checklist); surviving items migrated to BACKLOG.md
- I1: created `BACKLOG.md` per ADR-41 schema (8 streams, 11 items seeded)
- I2+I3: `README.md` architecture section updated to `src/ai_council/` namespace layout + test count to 362
- I4+I5: `docs/COUNCIL_QUESTION_GUIDE.md` → `docs/council-question-guide.md` and `docs/SYNTHESIS-QUALITY-RUBRIC.md` → `docs/synthesis-quality-rubric.md` (ADR-34 hyphen+lowercase)
- I6: `2026-03-15_CODE_REVIEW_REPORT.md` + `2026-03-26_CODE_REVIEW_REPORT.md` archived to `docs/audits/archive/legacy/`
- M1: `VISION.md` `last_reviewed` bumped 2026-05-09 → 2026-05-12
- M3: 4 lessons appended to `tasks/lessons.md` (target resolver fail-loud, inbox parity 3rd instance, ADR-43 schema DRYness, observability field design)
- M2 (AGENTS.md addition) deferred per strażnik own "low urgency" framing; tracked in BACKLOG.md P3 Governance

**Result:** ai-council fully aligned with strażnik audit findings except deferred M2. Audit pattern validated — I5 fresh violation caught and fixed same-day. CHANGELOG + commits = audit trail per single-round-trip principle.

**Next:** Step 5 smoke test (operator-driven, BACKLOG P1 Phase 2).

---

## 2026-05-12 — Phase 1 + ADR-34 hyphen combined

**Did (Phase 1):**
- Per-synthesis observability emitted: latency, transcript size, timeout flag, output tokens, error class — `DebateResult.synthesis_metrics` + `_metrics.json` synthesis block
- Created `docs/SYNTHESIS-QUALITY-RUBRIC.md` (5-point operator checklist)
- ADR-06 Qwen trial closed-out: deferred/abandoned with reopen trigger (DeepSeek round-blocking >2%)
- Gemini synthesizer version check: Case A — already on `gemini-3.1-pro-preview` (3.x), no upgrade action

**Did (ADR-34):**
- Council CLI emitter format flipped to hyphen per `.dev-knowledge` cycle 2 ratified mandate: `council_out_*` → `council-out-*`
- Downstream patterns updated (tests + docs aligned); no historical transcript rename (pre-decision artifacts)

**Result:** Observability foundation in place for smoke test (Phase 2). Cross-repo cycle 2 Change 1 implementation complete.

**Next:** Turn 4 delivery report to `.dev-knowledge` for cycle 2 closure; then Phase 2 smoke test operator-driven execution once baseline reads accumulated.

---

## 2026-05-11 — ADR governance sweep + HANDOFF cleanup

**Did:**
- Audit ADR-01..07 status headers against current ecosystem state
- ADR-07: file status flipped to "Superseded by ADR-43" — was index-only before today; file is source of truth
- ADR-01: status date updated to 2026-04-30 (Gemini synthesizer revision); header had captured only the 03-29 Sonnet revision
- ADR-02: revised to reflect 5-model default panel; original "3-model default" was factually wrong per current CLAUDE.md and code
- ADR-05: provider count corrected 3→4 (Grok/XAI added post-ADR, undocumented in ADR body)
- ADR-06: Qwen trial marked deferred (not pending); Gemini synthesizer change cross-referenced to ADR-01
- ADR-03, ADR-04: verified current, no changes
- decisions/README.md: index re-synced with ADR-01, ADR-02, ADR-06 updated statuses
- HANDOFF.md: deleted — handoff process owned by `.dev-knowledge` per ADR-42; pointer file adds noise not value

**Result:** ADR status headers are now authoritative in files; index mirrors them. Governance docs internally consistent.

**Candidates for future work (from audit):**
- ADR-01 Synthesizer selection: Gemini default still operative; model landscape has evolved (Claude 4.7, Gemini 3.x era). Candidate for meta-debate: should default panel + synthesizer refresh for 2026 model landscape?
- ADR-06 Cost optimization: Qwen trial deferred indefinitely; OpenRouter hedge not implemented. If DeepSeek reliability degrades again, Qwen/OpenRouter question will resurface.

---

## 2026-05-11 — Docs hygiene sweep

**Did:**
- Five-file docs internal-alignment pass post today's feature work
- HANDOFF.md: replaced pre-ADR-42 feature status doc with pointer to .dev-knowledge-owned handoff process
- COUNCIL_QUESTION_GUIDE.md: added `target-project` frontmatter + `--target-project` CLI flag section
- decisions/README.md: complete index (ADR-01 through ADR-07 with status) + cross-repo ADR-43 reference
- docs/archive/ consolidated into docs/audits/ with git history preserved via `git mv`
- docs/audits/README.md: new convention doc

**Result:** Internal docs reflect current state across all feature work shipped today. No code, test, or config changes.

---

## 2026-05-11 — ADR-43 amendment cycle 1 implementation

**Did:**
- Refactored `target_projects` schema per `.dev-knowledge`-approved ADR-43 amendment: `dev_root` + opt-in name list, paths computed as `<dev_root>/<name>/docs/decisions/transcripts/`
- Updated `TargetResolver` constructor signature and path computation; updated cli.py caller
- Adjusted ~10 existing test cases; added 5 new validation tests (dev_root required, dir validation, dict migration error, duplicate names, path computation) — 359 total
- Updated README.md + CLAUDE.md with new schema examples and ADR-43 reference
- Archived `.dev-knowledge` cycle closure note for symmetric audit trail
- Codex `/review` pending

**Result:** Schema is DRY; ecosystem root declared once; new repos join routing via single-line list addition.

**Next:** Codex `/review`; then generate delivery report for `.dev-knowledge` (Turn 4 implicit closure of cycle 1 handshake). Operator decides `git push` timing.

---

## 2026-05-11 — Post-routing cleanup

**Did:**
- Disabled `secondary_output_enabled` default — resolves architectural overlap with new `target_paths` per-invocation routing
- Added README Transcript Routing section (closes acceptance-criteria miss from previous session)
- Fixed CLAUDE.md test count drift (349 → 354)

**Result:** Clean post-routing state. No double-write to `.dev-knowledge` when `--target-project .dev-knowledge` used; README documents the feature for users.

**Next:** `.dev-knowledge` ESSENTIALS update (separate session). `git push` when ready (currently 21+ commits ahead of origin).

---

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
