# AI Council — Handoff

**Last updated:** 2026-04-21

## Status
Feature-complete. 255 unit tests (6 deselected). 5 debate + 4 research providers. 4 modes (pick/ideas/judge/research). mypy + ruff + pytest-cov all enforced via `scripts/check.ps1`. **8 commits ahead of origin/main — not pushed.**

## Recent
- Playbook compliance shipped: docs/, 6 ADRs, JOURNAL.md, CHANGELOG.md, folder governance
- `scripts/check.ps1` with mypy enforcement (0 errors)
- Provider unit tests added — all 5 providers covered with mocked SDKs (24 tests)
- pytest-cov installed and configured in pyproject.toml
- CouncilRunner extracted to `src/orchestrator.py`; `runner.py` re-exports for backward compat
- Sonnet 4.6 as default synthesizer (5x cost reduction vs Opus)

## Open
- Qwen 3.5 shadow trial vs DeepSeek R1 via OpenRouter
- OpenRouter as fallback routing layer (ADR-recommended, Week 1-2)
- o3-deep integration test (blocked — $10+ per run)
- Push 8 local commits to origin/main

## Key context
- Gemini event loop: fresh `genai.Client()` per call, not cached in `__init__`
- Windows cp1252: `Console(legacy_windows=False)` permanent fix
- Default panel: Claude + Gemini + OpenAI
- Synthesizer: Claude Sonnet 4.6 (switched from Opus, 5x cost reduction)
- Critique template uses `{previous_responses_anonymized}`, not `{previous_responses}`
- CouncilRunner lives in `src/orchestrator.py`; `src/runner.py` re-exports it (backward compat)
