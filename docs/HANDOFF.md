# AI Council — Handoff

**Last updated:** 2026-03-29

## Status
Feature-complete. 255 tests. 5 debate + 4 research providers. 4 modes (pick/ideas/judge/research).

## Recent
- Sonnet 4.6 added as default synthesizer (claude-sonnet-4-6 provider, `pick_synthesizer()` updated)
- Retry logic + error classification shipped
- Orchestrator extracted (runner.py/policy.py)
- Research mode with 4 providers (Perplexity, o4-mini, o3-deep, Gemini)
- Mode system with auto-detection (pick/ideas/judge/research)
- Cost tracking per debate

## Open
- Qwen 3.5 shadow trial vs DeepSeek (via OpenRouter)
- o3-deep integration test (blocked — $10+ per run)
- No pytest-cov coverage report configured

## Key context
- Gemini event loop: fresh `genai.Client()` per call, not cached in `__init__`
- Windows cp1252: `Console(legacy_windows=False)` permanent fix
- Default panel: Claude + Gemini + OpenAI
- Synthesizer: Claude Sonnet 4.6 (switched from Opus, 5x cost reduction)
- Critique template uses `{previous_responses_anonymized}`, not `{previous_responses}`
