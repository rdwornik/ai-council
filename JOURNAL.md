# Journal — ai-council

### 2026-05-17 — Question-framing bias audit

Audited 21 past curated Council debate questions against the question-framing bias rubric in `docs/council-question-guide.md`; 9 research-mode questions scored in full, 12 pick/judge headlines scored at title-only (full prompt not preserved in transcript). Asker-leakage, loaded terminology, and anchoring dominate; report is evidence for an operator decision, not a recommendation.

**Changes:** `docs/audits/2026-05-17-question-framing-bias-audit.md` (new audit report).

---

### 2026-05-17 — Question-framing bias-elimination section

**Did:** Added a cross-mode question-framing bias-elimination section to `docs/council-question-guide.md`, covering seven framing biases, a pre-flight self-check, and a research-mode sharpener.

**Changes:** `docs/council-question-guide.md` (new bias-elimination section).

---

### 2026-05-17 — Research-mode question guide + AGENTS.md

**Did:** Added a "Research-mode questions" section (recognition test + formulation rules + breadth-over-depth trap) to `docs/council-question-guide.md`; created `AGENTS.md` at repo root from the canonical ecosystem template (`.dev-knowledge/templates/AGENTS-md-template.md`) per Council #28.

**Result:** 362 tests green. Branch `docs/research-mode-guide-and-agents-md` ready for review.

**Changes:** `docs/council-question-guide.md` (new research-mode section); `AGENTS.md` (new file).

---

### 2026-05-17 — Documentation simplification rollout (ADR-48/49/50)

**Did:**
- Created branch `feat/docs-simplification-rollout`
- Removed `CHANGELOG.md` and `BACKLOG_ARCHIVE.md` per ADR-49
- Copied `scripts/normalize_headers.py` from `.dev-knowledge`; ran it over LESSONS.md (no-op — already H3 pipe schema) and JOURNAL.md (H2 → H3 dated entries)
- Added `.pre-commit-config.yaml` wiring normalize_headers as a local pre-commit hook
- Added "Documentation conventions" section to `CLAUDE.md` (no CHANGELOG, no BACKLOG_ARCHIVE, Conventional Commits standard, JOURNAL/LESSONS structure)
- Added transcript-to-ADR workflow step to `docs/council-question-guide.md`

**Result:** 362 tests green. Branch `feat/docs-simplification-rollout` ready for review. Not merged, not pushed.

**Changes:** CHANGELOG.md deleted; BACKLOG_ARCHIVE.md deleted; JOURNAL.md header levels H2→H3; CLAUDE.md +11 lines; council-question-guide.md +7 lines; scripts/normalize_headers.py added; .pre-commit-config.yaml added.

**Abandoned:** Step 4 (LESSONS ordering) — already reverse-chronological, no action needed.

**Next:** Operator reviews branch and merges if satisfied. Then apply same rollout to `corp-ops` and `corp-sca-time-automation`.

---

### 2026-05-15 — ADR-46+47 compliance cleanup (cross-repo handoff)

**Did:**
- LESSONS.md: migrated `## Session: Phase 1 Foundation (2026-02-21)` → `## 2026-02-21` + Session label in body
- JOURNAL.md: moved 2026-05-12 addendum entry to correct reverse-chrono position
- BACKLOG.md: [blocked] → [open] + Blocked annotation on Step 6; Status field added to all 11 entries; BACKLOG_ARCHIVE.md created
- Driven by .dev-knowledge cross-repo audit (2026-05-15-ecosystem-audit.md) + handoff bundle
- LESSONS.md H3 entries re-ordered to reverse-chrono (follow-on: 2026-05-12/2026-05-11 entries appeared after April entries)

**Result:** ai-council compliant with ADR-46 + ADR-47. Re-audit from .dev-knowledge expected to clear all 5 FAIL checks.

**Next:** Operator runs `python scripts/audit.py run` in .dev-knowledge to confirm. Stream B P1 items flip to [done] on clean audit.

---

### 2026-05-13 — P3 BACKLOG entry captured for ADR-34 timestamp-underscore case

**Did:** Added P3 BACKLOG entry naming the specific case (council-out filename `YYYYMMDD_HHMMSS` timestamp underscore) and the methodology question (ISO timestamp exempt from ADR-34?); cross-linked to existing P2 CI enforcement entry.

**Failed:** —

**Next:** Methodology decision on ADR-34 ISO-timestamp exemption — can be addressed when ADR-45 implementation surfaces it OR sooner if convenient.

---

### 2026-05-12 — Scrum-master review implementation (.dev-knowledge strażnik)

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

### 2026-05-12 — Phase 1 + ADR-34 hyphen combined

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

### 2026-05-12 — Scrum-master addendum implementation (I7 + I8)

**Did:**
- I7: moved `tasks/lessons.md` → `LESSONS.md` at repo root; retired `tasks/` folder entirely
- I8: renamed `docs/handoffs/_archive/` → `docs/handoffs/archive/`
- CLAUDE.md updated (Lessons Discovery bullet + Folder Governance `tasks/` entry replaced with `LESSONS.md`)
- VISION.md lessons path reference updated
- BACKLOG.md: no separate LESSONS.md-absent item existed; AGENTS.md M2 remains open (deferred)
- LESSONS.md: architect-side lesson captured on local-config-defense failure mode

**Process:** Both findings caught by operator post main-review implementation. Single-branch, 4 commits. Historical entries in CHANGELOG/JOURNAL left immutable.

**Result:** ai-council fully aligned with ecosystem convention on lessons location + archive folder naming. Original 10 findings + 2 addendum findings = all addressed except AGENTS.md (M2 from main review, still deferred per strażnik "low urgency").

---

### 2026-05-11 — ADR governance sweep + HANDOFF cleanup

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

### 2026-05-11 — Docs hygiene sweep

**Did:**
- Five-file docs internal-alignment pass post today's feature work
- HANDOFF.md: replaced pre-ADR-42 feature status doc with pointer to .dev-knowledge-owned handoff process
- COUNCIL_QUESTION_GUIDE.md: added `target-project` frontmatter + `--target-project` CLI flag section
- decisions/README.md: complete index (ADR-01 through ADR-07 with status) + cross-repo ADR-43 reference
- docs/archive/ consolidated into docs/audits/ with git history preserved via `git mv`
- docs/audits/README.md: new convention doc

**Result:** Internal docs reflect current state across all feature work shipped today. No code, test, or config changes.

---

### 2026-05-11 — ADR-43 amendment cycle 1 implementation

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

### 2026-05-11 — Post-routing cleanup

**Did:**
- Disabled `secondary_output_enabled` default — resolves architectural overlap with new `target_paths` per-invocation routing
- Added README Transcript Routing section (closes acceptance-criteria miss from previous session)
- Fixed CLAUDE.md test count drift (349 → 354)

**Result:** Clean post-routing state. No double-write to `.dev-knowledge` when `--target-project .dev-knowledge` used; README documents the feature for users.

**Next:** `.dev-knowledge` ESSENTIALS update (separate session). `git push` when ready (currently 21+ commits ahead of origin).

---

### 2026-05-11 — Cross-project transcript routing (feat/transcript-routing)

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

### 2026-05-09 — Audit-sync governance closure (F-01, F-02)

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
