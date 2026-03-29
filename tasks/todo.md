# AI Council — Task Tracking

## Current Status

**Phase 1 complete.** All providers, debate pipeline, modes, research, and tests implemented.
199 unit tests passing (6 deselected: integration + envcheck).

## Open Items

### Provider Unit Tests
- [ ] `tests/test_providers.py` — unit tests for individual providers (anthropic, gemini, openai_provider, xai, deepseek) with mocked SDK clients
  - Currently only covered by integration test (live keys required)
  - See CLAUDE.md "Known issues"

### Quality / Tooling
- [ ] Add `pytest-cov` and configure coverage reporting
- [ ] Add `mypy` for static type checking in CI
- [ ] `--help` improvements — group flags by mode, surface mode aliases more clearly

### Phase 2 Candidates
- [ ] Orchestrator extraction — pull `CouncilRunner` coordination logic into a standalone `orchestrator.py` (currently mixed into `runner.py`)
- [x] Retry logic — `classify_error()` in base.py, `was_retry` on `ModelResponse`, 1x retry with 1.5x timeout in `_call_provider`; provider notes in output
- [x] DeepSeek graceful skip — healthcheck returns specific messages (auth/timeout/unreachable) via `classify_error()`

### Research Mode
- [ ] `openai_deep_research.py` — o3-deep-research (~45 min timeout) is wired but untested end-to-end; add integration test
- [ ] Research output: consider `--format json` flag for structured citation output

## Completed (archived)

Phase 1 Waves 1–5: all foundation, config, providers, debate pipeline, CLI, tests — done.
Research mode (Perplexity, o4-mini, Gemini + grounding, o3-deep, cache, merger, display) — done.
Mode system (pick/ideas/judge/research + auto-detection) — done.
Inbox batch mode — done.
Cost tracking + metrics — done.
Healthcheck — done.
