# Changelog — ai-council

## 2026-05-11

### Changed (docs hygiene)
- `docs/HANDOFF.md`: aligned with .dev-knowledge-owned handoff process (pointer to ADR-42 + Playbook)
- `docs/COUNCIL_QUESTION_GUIDE.md`: added `target-project` frontmatter and `--target-project` CLI flag section
- `docs/decisions/README.md`: complete ADR index with status (ADR-01 through ADR-07); cross-repo ADR-43 reference
- `docs/archive/` consolidated into `docs/audits/` (git history preserved via `git mv`)
- `docs/audits/README.md`: new file documenting audit archive convention

### Changed (breaking, per ADR-43 amendment cycle 1)
- `config/settings.yaml`: `target_projects` schema refactored from `dict[name, full_path]` to `dev_root: str` + `target_projects: list[name]`. Paths computed as `<dev_root>/<name>/docs/decisions/transcripts/`. Old config fails loud at load with migration hint.

### Added
- `docs/handoffs/_archive/2026-05-11_dev-knowledge-cycle-closure.md`: `.dev-knowledge` cycle 1 closure note (cross-repo handshake audit trail)

### Changed
- `config/settings.yaml`: `secondary_output_enabled` default flipped to `false` — `target_paths` per-invocation routing supersedes always-on global mirror
- `README.md`: added Transcript Routing section
- `CLAUDE.md`: corrected test count (349 → 354)

### Added
- Cross-project transcript routing via `target-project` frontmatter field (inbox mode)
- `--target-project` Click flag for direct CLI invocation (multi-target via repeated flag)
- `config/settings.yaml`: `target_projects` map for target name → path resolution
- `src/ai_council/routing.py`: `TargetResolver` + `RoutingError` — fail-loud on unknown names
- `RunRequest.target_paths`: list of resolved mirror dirs forwarded through orchestrator to `save_to_file`
- `output.py` / `research/output.py`: `target_paths` parameter — auto-mkdir, best-effort writes
- `config/config_loader.py`: `AppConfig.target_projects: dict[str, str]`
- 7 new tests across test_config, test_routing, test_inbox, test_output, test_cli, test_runner (349 total)

### Added (audit trail)
- `docs/handoffs/_archive/`: outgoing delivery report (to `.dev-knowledge`) and inbound press-back (from `.dev-knowledge`) for cross-repo transcript-routing feature audit trail

## 2026-05-09

### Changed
- `config/settings.yaml`: update grok model string to grok-4.3 (commit 62c1f7d)

### Added
- `VISION.md` — created per ADR-33 (tier M, Lite schema)
- `CLAUDE.md` — Lessons Discovery section configuring `DEV_KNOWLEDGE_PATH` per ADR-35; ADR naming note (future underscore per ADR-34, existing 7 kebab-case grandfathered per ADR-29)

## [Unreleased]

### Changed
- **ADR-38 compliance:** migrated from flat `src/` layout to `src/ai_council/` namespace package. All Python sources now live under `src/ai_council/`; entry points are `ai_council.cli:main`; imports rewritten from `from src.X` to `from ai_council.X` across 27 source modules and 17 test files.
- Added `[build-system]` block to `pyproject.toml` with `setuptools.build_meta` backend (was missing).
- `[tool.setuptools.packages.find]` now uses `where = ["src", "."]` / `include = ["ai_council*", "config*"]` to keep `config/` (top-level package, sibling of `src/`) discoverable alongside `ai_council`.
- `[tool.coverage.run]` source/omit paths updated to `src/ai_council/`.
- Consolidated pytest configuration into `pyproject.toml`'s `[tool.pytest.ini_options]`; deleted `pytest.ini` (contents were duplicated).
- Updated `python -m src.cli` → `python -m ai_council.cli` in CLAUDE.md, README.md, docs/COUNCIL_QUESTION_GUIDE.md.
- All 310 unit tests pass post-migration (matches pre-migration baseline; zero functional changes).

### Added
- Mode system: pick/ideas/judge/research with auto-detection via cheap LLM call
- Research mode: 4 providers (Perplexity sonar-pro, o4-mini-deep-research, o3-deep-research, Gemini+Search)
- `--deep` flag for o3-deep-research; `--no-cache` to bypass file cache
- Retry logic with error classification (`classify_error()`, `was_retry` tracking)
- Graceful provider skip with specific healthcheck messages per failure mode
- Cost tracking per debate: per-call token counts + USD estimates in `DebateMetrics`
- Orchestrator extraction: `CouncilRunner` in `runner.py`, `RunPolicy` in `policy.py`
- Provider unit tests with mocked SDKs (24 tests)
- `pytest-cov`, `mypy` enforcement via `scripts/check.ps1`
- `--format json` output flag
- `--modes` flag to list available debate modes
- Progressive research display with Rich Live + `asyncio.wait()`
- File cache for research results (`~/.ai-council/research_cache/`, 7-day TTL)
- Claude Sonnet 4.6 provider + set as default synthesizer
- Grouped flags and usage examples in `--help` epilog

### Changed
- Default panel: Claude + Gemini + OpenAI (was: Claude + Gemini + DeepSeek)
- Synthesizer: Claude Sonnet 4.6 (was: Claude Opus 4.6, 5x cost reduction)
- CLI short flag `-M` for mode (was `-m`, conflicted with `python -m`)
- Upgraded Round 1 prompt: structured decision framework
- Upgraded Round 2 prompt: steelmanning + hidden assumption identification
- Upgraded synthesis prompt: argument quality weighting + blind spot detection

### Fixed
- Gemini event loop crash: fresh `genai.Client()` per call, not cached in `__init__`
- Windows cp1252 encoding: `Console(legacy_windows=False)` permanent fix
- `FAILED_` prefix inherited in output filenames (slug stripping)
- `--full` flag ignored when inbox frontmatter had `models` field
- Gemini research model: updated to `gemini-2.5-pro` (preview slug returned 404)
- OpenAI mini research: added required `web_search_preview` tool to Responses API call (deep-research endpoint rejects calls without a search tool)

## [1.0.0] — 2026-03-15

- Multi-model debate: Claude, Gemini, GPT, Grok, DeepSeek
- Panel system with customizable model selection
- Persona injection per provider
- Blind voting with anonymized proposals
- Non-participating synthesizer for final verdict
- Inbox batch mode with frontmatter overrides
- Health check before debate start
- 72 tests passing
