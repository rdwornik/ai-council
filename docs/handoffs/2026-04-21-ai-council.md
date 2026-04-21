# Handoff — AI Council
**Date:** 2026-04-21
**Branch:** main (8 commits ahead of origin/main, not pushed)

---

## Verified state

Verified against `git log`, current files, and running test suite on 2026-04-21.

### Tests
- **255 unit tests pass** (`pytest tests/ -m "not integration and not envcheck"`)
- 6 deselected (integration + envcheck)
- mypy, ruff, pytest-cov all enforced via `scripts/check.ps1`

### Architecture
`CouncilRunner` lives in **`src/orchestrator.py`** (extracted in `95e4035`). `src/runner.py` re-exports it for backward compat. CLAUDE.md architecture section updated to reflect this.

### Providers
- **Debate (5):** Claude, Gemini, GPT, Grok, DeepSeek
- **Research (4):** Perplexity sonar-pro, o4-mini-deep-research, o3-deep-research (--deep only), Gemini Deep Research

### Modes
`pick` / `ideas` / `judge` / `research` — auto-detected via cheap LLM call.

---

## Discrepancies found and corrected

| Location | Was | Now |
|---|---|---|
| CLAUDE.md → Architecture | `runner.py` listed as home of `CouncilRunner` | Added `orchestrator.py`; fixed `runner.py` description |
| CLAUDE.md → Known Issues | "No unit tests for individual providers" | Removed — fixed in `5bc6842` (24 mocked SDK tests) |
| CLAUDE.md → Known Issues | "mypy not installed" | Removed — enforced since `4fce22c` |
| CLAUDE.md → Known Issues | "No pytest-cov configured" | Removed — configured in `pyproject.toml` |
| docs/HANDOFF.md | "2 commits ahead of origin" | Updated to 8 commits; open tasks refreshed |
| Previous HANDOFF doc | Claimed stale items cleaned up | They were still in CLAUDE.md — now actually removed |

---

## Commits in this session

None — this handoff only corrects documentation.

---

## Open tasks (as of 2026-04-21)

- [ ] Push 8 local commits to origin/main (non-destructive, just blocked by habit)
- [ ] Qwen 3.5 shadow trial vs DeepSeek R1 via OpenRouter
- [ ] OpenRouter as fallback routing layer (ADR-recommended)
- [ ] o3-deep integration test (blocked — $10+ per run)

---

## Key invariants

- **Gemini event loop:** `genai.Client(api_key=...)` must be instantiated inside the async method, never in `__init__`
- **Windows cp1252:** `Console(legacy_windows=False)` — do not use Unicode chars in Rich progress callbacks
- **Critique template:** `{previous_responses_anonymized}`, not `{previous_responses}`
- **Default panel:** Claude + Gemini + OpenAI
- **Synthesizer:** Claude Sonnet 4.6 (non-participating; fallback allows participant if none available)
- **Do NOT merge provider files** — `xai.py` and `deepseek.py` stay separate despite OpenAI-compatible interface
