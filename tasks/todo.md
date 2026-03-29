# AI Council — Task Tracking

## Current Status

**Phase 1 + Phase 2 complete.** All providers, debate pipeline, modes, research, orchestrator extraction, and tests implemented.
255 unit tests passing (6 deselected: integration + envcheck).

## Open Items

### Quality / Tooling
- [ ] `mypy` — installed in dev deps; no CI enforcement yet; run manually with `mypy src/`
- [ ] `--help` improvements — flag grouping by mode (Click limitation; `--modes` flag already exists)

### Research Mode
- [ ] `openai_deep_research.py` — o3-deep-research (~45 min timeout) is wired but untested end-to-end; add integration test

## Completed (archived)

Phase 1 Waves 1–5: all foundation, config, providers, debate pipeline, CLI, tests — done.
Research mode (Perplexity, o4-mini, Gemini + grounding, o3-deep, cache, merger, display) — done.
Mode system (pick/ideas/judge/research + auto-detection) — done.
Inbox batch mode — done.
Cost tracking + metrics — done.
Healthcheck — done.
