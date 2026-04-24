# Journal — ai-council

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
