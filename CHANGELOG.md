# Changelog — ai-council

## [Unreleased]

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
