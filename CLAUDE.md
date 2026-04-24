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
  orchestrator.py      — CouncilRunner.run(); debate lifecycle coordination (extracted from runner.py)
  runner.py            — build_all_providers(); determine_panel(); pick_synthesizer(); re-exports CouncilRunner
  debate.py            — run_debate() → DebateOutcome; persona injection; blind voting via _anonymize_responses()
  synthesis.py         — synthesize(); builds transcript; calls non-participating synthesizer → DebateResult
  output.py            — save_to_file(); print_round_summary(); print_synthesis(); print_cost_summary()
  models.py            — Dataclasses: Question, ModelResponse, Round, DebateOutcome, DebateResult, RunRequest, DebateMetrics
  policy.py            — RunPolicy: min_panel_size, retry_on patterns, should_retry()
  metrics.py           — build_call_metrics(); build_debate_metrics(); per-provider cost rates
  healthcheck.py       — run_health_checks(); pings all providers before debate starts
  inbox.py             — Inbox folder scanning, frontmatter parsing, auto-archive
  mode_detector.py     — detect_mode(); _pick_cheapest(); auto-classifies question via cheap LLM call
  providers/
    base.py            — AIProvider ABC + ProviderError
    anthropic.py       — Claude (Anthropic SDK, asyncio.to_thread)
    gemini.py          — Gemini (google-genai, native async)
    openai_provider.py — GPT (AsyncOpenAI)
    xai.py             — Grok (OpenAI-compatible, base_url)
    deepseek.py        — DeepSeek (OpenAI-compatible, base_url)
  research/
    __init__.py        — Package init
    models.py          — Dataclasses: Source, ResearchResult, MergedResearchReport
    provider.py        — ResearchProvider ABC + ResearchProviderError
    display.py         — run_research_with_display(); Rich Live progress table; asyncio.wait() spinner loop
    merger.py          — make_cache_key(); merge_results(); summarize_report(); _deduplicate_sources()
    cache.py           — cache_get(); cache_put(); cache_invalidate(); file-based TTL cache
    runner.py          — build_research_providers(); run_research(); full pipeline orchestration
    output.py          — save_research_to_file(); print_research_summary()
    providers/
      perplexity.py         — Perplexity sonar-pro (OpenAI-compatible, base_url)
      openai_mini_research.py — o4-mini-deep-research (Responses API + background polling)
      openai_deep_research.py — o3-deep-research (--deep only, 45 min timeout)
      gemini_research.py    — Gemini + Google Search grounding
config/
  settings.yaml        — Models, prompts, personas, panels, defaults (single source of truth)
  config_loader.py     — YAML -> typed dataclasses; API key detection at startup
tests/                 — 199 unit tests + 1 integration test
```

## Dev standards

- Python 3.12+, `pyproject.toml` as single dependency source
- `ruff` for linting/formatting, `pytest` + `pytest-asyncio` for testing
- Feature branches, no deletions without asking
- Logging not print, dataclasses not dicts
- Click CLI, Rich console output
- Do NOT merge OpenAI-compatible providers (xai.py, deepseek.py) — keep separate
- Before merging any branch, run: `.\scripts\check.ps1` (pytest + mypy + ruff)

## Key commands

```bash
# Default 3-model panel, pick mode (default) — auto-detected or explicit
python -m src.cli "Should we use REST or GraphQL?" --rounds 1

# Ideas mode — brainstorm
python -m src.cli -M i "What features am I not using in my auth system?"

# Judge mode — evaluate a proposal
python -m src.cli -M j "Is this microservices architecture production-ready?"

# Full 5-model panel
python -m src.cli "Monorepo vs polyrepo?" --rounds 2 --full

# Custom models + custom synthesizer
python -m src.cli "SQL or NoSQL?" --models claude,openai --synthesizer gemini

# From file
python -m src.cli --file question.md --rounds 3

# Inbox batch mode (reads council_inbox/*.md with optional frontmatter overrides)
python -m src.cli --inbox

# Research mode — parallel web research (Perplexity + Gemini + o4-mini)
python -m src.cli -M research "Best HTAP databases in 2026"
python -m src.cli -M r "LLM inference hardware comparison"

# Research mode — include slow deep providers (o3-deep-research, ~45 min)
python -m src.cli -M research "LLM inference hardware" --deep

# Research mode — skip cache read/write
python -m src.cli -M r "Redis vs Valkey" --no-cache

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
- **Mode system**: `pick`/`ideas`/`judge`/`research` via `--mode`/`-M`; aliases `p`/`i`/`j`/`r`; auto-detected from question text; mode-specific prompt templates and `persona_mode_directives` in settings.yaml; `pick` uses existing `prompts.*` for backward compat; `RunRequest.mode` and `DebateResult.mode` carry mode through pipeline
- **Research mode**: Separate code path — bypasses debate pipeline entirely; runs parallel providers via `asyncio.wait()`+`as_completed` progressive display; merges results; summarizes via LLM; writes `{ts}_{slug}_research.md`; file cache under `~/.ai-council/research_cache/` with 7-day TTL

## Debate modes

| Mode | Aliases | Default rounds | Purpose |
|------|---------|---------------|---------|
| `pick` | `p`, `pick`, `d`, `decide` | 2 | Choose between options. **(default)** |
| `ideas` | `i`, `ideas` | 1 | Brainstorm. Surface unknowns and divergent ideas. |
| `judge` | `j`, `judge` | 2 | Evaluate a proposal or claim. Get a verdict. |
| `research` | `r`, `research` | — | Multi-source web research report with citations. |

- `pick` uses `prompts.initial`/`prompts.critique`/`prompts.synthesis` from settings.yaml (backward compat).
- `ideas`/`judge` use per-mode `round1_header`/`round1_instruction`/`round1_structure`/`round2_instruction`/`synthesis_output` from `modes:` block in settings.yaml.
- `persona_mode_directives` in settings.yaml injects a `CRITICAL INSTRUCTION:` override at top of each persona for `ideas`/`judge`.
- Mode auto-detected from question text via cheap LLM call (deepseek > gemini > others); 5s interactive confirm.
- Inbox frontmatter can specify `mode:` — CLI `--mode` overrides it.
- `resolve_mode()` in `config_loader.py` maps aliases; raises `ValueError` for unknown modes.
- `research` mode routes to `src/research/runner.py:run_research()` before the debate pipeline; `--deep` adds o3-deep-research; `--no-cache` bypasses file cache.

## Research providers

| Provider | Key env var | Default | --deep only | Notes |
|----------|-------------|---------|------------|-------|
| `perplexity` | `PERPLEXITY_API_KEY` | yes | no | sonar-pro; OpenAI-compatible |
| `openai_mini` | `OPENAI_API_KEY` | yes | no | o4-mini-deep-research; Responses API |
| `gemini` | `GEMINI_API_KEY` | yes | no | Gemini + Google Search grounding |
| `openai_deep` | `OPENAI_API_KEY` | no | yes | o3-deep-research; ~45 min timeout |

Missing API keys are silently skipped — remaining providers still run.

## Test suite

```bash
pytest tests/ -m "not integration and not envcheck" -v   # 255 unit tests (6 deselected), no API keys needed
pytest tests/ -m envcheck -v             # verify API keys are in environment
pytest tests/test_integration.py -v      # requires 2+ API keys in .env
```

Coverage: cli, config, debate, healthcheck, inbox, models, mode_detector, output, synthesis, research (models/cache/merger/display/runner/provider)

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

This repo uses: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `PERPLEXITY_API_KEY`

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
- **google-genai event loop**: `genai.Client(api_key=...)` must be created INSIDE the async method, NOT in `__init__` — otherwise it binds to the wrong event loop
- **Research make_cache_key location**: `make_cache_key()` lives in `src/research/merger.py`, NOT `src/research/cache.py`
- **Windows /dev/null**: Use `io.StringIO()` for Console mocking in tests, not `open("/dev/null", "w")`

## Known issues

- DeepSeek API key may not be available in current environment
- o3-deep-research integration test not run (blocked — $10+ per run)


## Folder governance
- `src/` — all Python source code
- `tests/` — all tests
- `config/` — settings.yaml and config_loader.py
- `scripts/` — check.ps1 and utility scripts
- `docs/` — HANDOFF.md, decisions/ (ADRs), handoffs/ (session handoffs), archive/ (frozen snapshots)
- `output/` — gitignored; debate transcripts and research reports
- `council_inbox/` — gitignored; drop .md files for batch processing
- `eval/` — evaluation data (eval_history.jsonl)
- `tasks/` — todo.md, lessons.md (local task tracking)
- Do not create files outside these directories without updating this section

## Global Skills
Before modifying code, consult ~/.claude/skills/gotchas/ for known ecosystem traps.
After pytest passes, check ~/.claude/skills/verify/ for verification scripts.

