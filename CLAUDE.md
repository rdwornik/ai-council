# CLAUDE.md — AI Council

## What this repo does

Multi-model AI debate tool. Pose an architectural question; a panel of AI models (Claude, Gemini, GPT, Grok, DeepSeek) debate in parallel rounds with anonymized critique; a non-participating model synthesizes the final verdict into a structured decision document.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1  # Windows
pip install -e ".[dev]"
cp .env.example .env  # add API keys
python -m src.cli "Should we use REST or GraphQL?" --rounds 1
pytest tests/ -m "not integration" -v
```

## Architecture

```
src/
  cli.py               — Click entry point; PROVIDER_CLASSES dict; builds RunRequest, delegates to CouncilRunner
  runner.py            — CouncilRunner.run(); build_all_providers(); determine_panel(); pick_synthesizer()
  debate.py            — run_debate() → DebateOutcome; persona injection; blind voting via _anonymize_responses()
  synthesis.py         — synthesize(); builds transcript; calls non-participating synthesizer → DebateResult
  output.py            — save_to_file(); print_round_summary(); print_synthesis(); print_cost_summary()
  models.py            — Dataclasses: Question, ModelResponse, Round, DebateOutcome, DebateResult, RunRequest, DebateMetrics
  policy.py            — RunPolicy: min_panel_size, retry_on patterns, should_retry()
  metrics.py           — build_call_metrics(); build_debate_metrics(); per-provider cost rates
  healthcheck.py       — run_health_checks(); pings all providers before debate starts
  inbox.py             — Inbox folder scanning, frontmatter parsing, auto-archive
  providers/
    base.py            — AIProvider ABC + ProviderError
    anthropic.py       — Claude (Anthropic SDK, asyncio.to_thread)
    gemini.py          — Gemini (google-genai, native async)
    openai_provider.py — GPT (AsyncOpenAI)
    xai.py             — Grok (OpenAI-compatible, base_url)
    deepseek.py        — DeepSeek (OpenAI-compatible, base_url)
config/
  settings.yaml        — Models, prompts, personas, panels, defaults (single source of truth)
  config_loader.py     — YAML -> typed dataclasses; API key detection at startup
tests/                 — 121 unit tests + 1 integration test
```

## Dev standards

- Python 3.12+, `pyproject.toml` as single dependency source
- `ruff` for linting/formatting, `pytest` + `pytest-asyncio` for testing
- Feature branches, no deletions without asking
- Logging not print, dataclasses not dicts
- Click CLI, Rich console output
- Do NOT merge OpenAI-compatible providers (xai.py, deepseek.py) — keep separate

## Key commands

```bash
# Default 3-model panel (claude, gemini, deepseek)
python -m src.cli "Should we use REST or GraphQL?" --rounds 1

# Full 5-model panel
python -m src.cli "Monorepo vs polyrepo?" --rounds 2 --full

# Custom models + custom synthesizer
python -m src.cli "SQL or NoSQL?" --models claude,openai --synthesizer gemini

# From file
python -m src.cli --file question.md --rounds 3

# Inbox batch mode (reads council_inbox/*.md with optional frontmatter overrides)
python -m src.cli --inbox

# Debug logging
python -m src.cli "question" --rounds 1 --verbose
```

## Key design decisions

- **Panel system**: `determine_panel()` in runner.py; `--models` wins over `--full` wins over default
- **Persona injection**: Per-provider personas in `settings.yaml`; injected via `{persona}` placeholder in prompt templates
- **Blind voting**: `_anonymize_responses()` shuffles + labels as "Proposal A/B/C"; provider names hidden
- **Non-participating synthesizer**: `pick_synthesizer()` picks a model outside the panel; falls back with `is_participant=True` if none available
- **Config source of truth**: All model strings, timeouts, max_tokens, prompts, personas in `settings.yaml`
- **RunPolicy**: Retry logic (`retry_on` patterns, `min_panel_size`) decoupled from debate logic; passed into `run_debate()`
- **DebateOutcome**: `run_debate()` returns `DebateOutcome` (rounds + degradation fields); not `list[Round]`
- **Graceful degradation**: Round 2+ all-fail → `DebateOutcome(degraded=True)` with partial rounds; round 1 all-fail → `RuntimeError`
- **provider_statuses**: Dict tracking per-provider `"ok"` | `"failed"` — surfaced in output and saved to markdown
- **Cost tracking**: `metrics.py` builds `DebateMetrics` with per-call token counts and estimated USD costs

## Test suite

```bash
pytest tests/ -m "not integration and not envcheck" -v   # 121 unit tests, no API keys needed
pytest tests/ -m envcheck -v             # verify API keys are in environment
pytest tests/test_integration.py -v      # requires 2+ API keys in .env
```

Coverage: cli, config, debate, healthcheck, inbox, models, output, synthesis

## Dependencies

- `click` — CLI framework
- `rich` — Console output formatting
- `pyyaml` — Config loading
- `python-dotenv` — .env file loading
- `anthropic`, `openai`, `google-genai` — AI provider SDKs
- `python-frontmatter` — Inbox file parsing

## API Keys

Keys loaded globally from `Documents/.secrets/.env` via PowerShell profile.
Do NOT add API keys to local `.env`.
Check: `keys list` | Update: `keys set KEY value` | Reload: `keys reload`

This repo uses: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`

## Integration points

ai-council is fully standalone. It is used for architectural decision-making across the ecosystem.

- AI Council debates produce binding decisions (see ECOSYSTEM.md)
- 5 LLM providers: Claude, Gemini, GPT, Grok, DeepSeek

## Related repos

- [ECOSYSTEM.md](../ECOSYSTEM.md) — full ecosystem overview, AI Council binding decisions
- [corp-by-os](../corp-by-os/) — root orchestrator
- [corp-os-meta](../corp-os-meta/) — shared schema

## Gotchas

- **Windows cp1252**: Do not print Unicode chars in Rich progress callbacks. Use ASCII only.
- **MockProvider ABC**: `async def generate` must exist in class body AND be shadowed by `AsyncMock` in `__init__`
- **pytest-asyncio**: Needs `asyncio_mode = auto` in `pytest.ini`
- **Critique template**: Uses `{previous_responses_anonymized}`, not `{previous_responses}`
- **google-genai async**: `client.aio.models.generate_content()` — native async, NOT `asyncio.to_thread`

## Known issues

- **No unit tests for individual providers** — `providers/anthropic.py`, `gemini.py`, `openai_provider.py`, `xai.py`, `deepseek.py` are only covered by the integration test (requires live API keys). Unit tests with mocked SDK clients would improve coverage.
- No `pytest-cov` configured for coverage reporting
- `mypy` not installed — no static type checking in CI
- DeepSeek API key may not be available in current environment


## Global Skills
Before modifying code, consult ~/.claude/skills/gotchas/ for known ecosystem traps.
After pytest passes, check ~/.claude/skills/verify/ for verification scripts.

