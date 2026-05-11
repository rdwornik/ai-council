# CLAUDE.md — AI Council

## What this repo does

Multi-model AI debate tool. Pose an architectural question; a panel of AI models (Claude, Gemini, GPT, Grok, DeepSeek) debate in parallel rounds with anonymized critique; a non-participating model synthesizes the final verdict into a structured decision document.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1  # Windows
pip install -e ".[dev]"
cp .env.example .env  # add API keys
council "Should we use REST or GraphQL?" --rounds 1
pytest tests/ -m "not integration and not envcheck" -v
```

## Architecture

Per ADR-38, all source modules live under `src/ai_council/` (namespace package).

```
src/ai_council/
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
  inbox.py             — Inbox folder scanning, frontmatter parsing, auto-archive, Downloads detection
  routing.py           — TargetResolver; resolves target-project names to transcript dirs; RoutingError on unknown
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
      gemini_research.py    — Gemini Deep Research (Interactions API, autonomous agent, ~5-20 min)
config/                — top-level package, sibling of src/ (NOT under src/ai_council/)
  settings.yaml        — Models, prompts, personas, panels, defaults (single source of truth)
  config_loader.py     — YAML -> typed dataclasses; API key detection at startup
scripts/
  check.ps1            — pytest + mypy + ruff pre-merge check
  council-ask.ps1      — helper script for quick CLI invocations
tests/                 — 354 unit tests + integration tests
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
# Default: full 5-model panel, pick mode (default) — auto-detected or explicit
council "Should we use REST or GraphQL?" --rounds 1
python -m ai_council.cli "Should we use REST or GraphQL?" --rounds 1  # also works

# 3-model panel (lite mode)
council --lite "Quick question" --rounds 1

# Ideas mode — brainstorm
council -M i "What features am I not using in my auth system?"

# Judge mode — evaluate a proposal
council -M j "Is this microservices architecture production-ready?"

# Full 5-model panel (no-op — default is already full)
council "Monorepo vs polyrepo?" --rounds 2 --full

# Custom models + custom synthesizer
council "SQL or NoSQL?" --models claude,openai --synthesizer gemini

# From file
council --file question.md --rounds 3

# Inbox batch mode (reads council_inbox/*.md with optional frontmatter overrides)
council --inbox

# Research mode — parallel web research (Perplexity + Gemini + o4-mini)
council -M research "Best HTAP databases in 2026"
council -M r "LLM inference hardware comparison"

# Research mode — include slow deep providers (o3-deep-research, ~45 min)
council -M research "LLM inference hardware" --deep

# Research mode — skip cache read/write
council -M r "Redis vs Valkey" --no-cache

# Debug logging
council "question" --rounds 1 --verbose
```

## Key design decisions

- **Panel system**: `determine_panel()` in runner.py; `--models` wins over `--full`/`--lite` wins over default. Full 5-model panel is now the default; `--lite` uses the 3-model panel; `--full` is a no-op kept for backward compat
- **Persona injection**: Per-provider personas in `settings.yaml`; injected via `{persona}` placeholder in prompt templates
- **Blind voting**: `_anonymize_responses()` shuffles + labels as "Proposal A/B/C"; provider names hidden
- **Non-participating synthesizer**: `pick_synthesizer()` picks a model outside the panel; default synthesizer is `gemini` (`gemini-3.1-pro-preview`); falls back with `is_participant=True` if none available
- **Config source of truth**: All model strings, timeouts, max_tokens, prompts, personas in `settings.yaml`
- **RunPolicy**: Retry logic (`retry_on` patterns, `min_panel_size`) decoupled from debate logic; passed into `run_debate()`
- **DebateOutcome**: `run_debate()` returns `DebateOutcome` (rounds + degradation fields); not `list[Round]`
- **Graceful degradation**: Round 2+ all-fail → `DebateOutcome(degraded=True)` with partial rounds; round 1 all-fail → `RuntimeError`
- **provider_statuses**: Dict tracking per-provider `"ok"` | `"failed"` — surfaced in output and saved to markdown
- **Cost tracking**: `metrics.py` builds `DebateMetrics` with per-call token counts and estimated USD costs
- **Mode system**: `pick`/`ideas`/`judge`/`research` via `--mode`/`-M`; aliases `p`/`i`/`j`/`r`; auto-detected from question text; mode-specific prompt templates and `persona_mode_directives` in settings.yaml; `pick` uses existing `prompts.*` for backward compat; `RunRequest.mode` and `DebateResult.mode` carry mode through pipeline
- **Research mode**: Separate code path — bypasses debate pipeline entirely; runs parallel providers via `asyncio.wait()`+`as_completed` progressive display; merges results; summarizes via LLM; writes `{ts}_{slug}_research.md`; file cache under `~/.ai-council/research_cache/` with 7-day TTL

## Transcript Routing

Opt-in, per-invocation mirroring of debate transcripts to named target project directories.

**Two-layer model:**
- **Names** (e.g., `.dev-knowledge`) come from frontmatter or `--target-project` flag — dynamic per invocation
- **Paths** (e.g., `C:/Users/.../Dev/.dev-knowledge`) live in `config/settings.yaml` under `target_projects` — never hardcoded

**Frontmatter (inbox mode):**
```yaml
---
mode: judge
target-project: .dev-knowledge          # single string
# OR
target-project: [.dev-knowledge, foo]   # multi-target (rare)
---
```

**CLI flag (direct mode):**
```bash
council --target-project .dev-knowledge "question"
# Multi-target — repeat the flag:
council --target-project .dev-knowledge --target-project foo "question"
```

**Config (`config/settings.yaml`):**
```yaml
target_projects:
  ".dev-knowledge": "C:/Users/1028120/Documents/Dev/.dev-knowledge"
  # "corp-monorepo": "C:/Users/1028120/Documents/Dev/corp-monorepo"
```

**Behavior:**
- Transcript written to `<target_root>/docs/decisions/transcripts/<filename>` (auto-mkdir)
- Canonical `output/` is always written first (hard requirement; failure is hard error)
- Mirror writes are best-effort: failure logs a warning, canonical is never affected
- Unknown target name → `RoutingError` at parse time (before debate runs), listing known names
- No `target-project` specified → canonical only, unchanged behavior
- All 4 modes (pick / ideas / judge / research) route through the same plumbing

**Key files:**
- `src/ai_council/routing.py` — `TargetResolver` + `RoutingError`
- `config/settings.yaml` — `target_projects` map (single source of truth for paths)
- `config/config_loader.py` — `AppConfig.target_projects: dict[str, str]`

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
| `grok` | `XAI_API_KEY` | yes | no | grok-3; Responses API; x_search + web_search; unique X/Twitter signal |
| `openai_mini` | `OPENAI_API_KEY` | yes | no | o4-mini-deep-research; Responses API |
| `gemini` | `GEMINI_API_KEY` | yes | no | Interactions API (`deep-research-preview-04-2026`); autonomous agent; ~5-20 min |
| `openai_deep` | `OPENAI_API_KEY` | no | yes | o3-deep-research; ~45 min timeout |

Missing API keys are silently skipped — remaining providers still run.

## Test suite

```bash
pytest tests/ -m "not integration and not envcheck" -v   # 266 unit tests (6 deselected), no API keys needed
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

## Downloads scanning

`--inbox` auto-scans `~/Downloads/*.md` before the council_inbox folder. Detection: any YAML frontmatter key matching `mode`, `rounds`, `models`, `synthesizer`, or `full` (case-insensitive). Config in `settings.yaml` under `inbox.scan_downloads`, `inbox.downloads_dir`, `inbox.council_frontmatter_keys`. Detected files are archived to `council_inbox/archive/`.

## Gotchas

- **Windows cp1252**: Do not print Unicode chars in Rich progress callbacks. Use ASCII only.
- **MockProvider ABC**: `async def generate` must exist in class body AND be shadowed by `AsyncMock` in `__init__`
- **pytest-asyncio**: Needs `asyncio_mode = auto` in `pytest.ini`
- **Critique template**: Uses `{previous_responses_anonymized}`, not `{previous_responses}`
- **google-genai async**: `client.aio.models.generate_content()` — native async, NOT `asyncio.to_thread`
- **google-genai event loop**: `genai.Client(api_key=...)` must be created INSIDE the async method, NOT in `__init__` — otherwise it binds to the wrong event loop
- **Interactions API experimental warnings**: `client.aio.interactions` emits `UserWarning: Interactions usage is experimental` on every access — suppress with `warnings.catch_warnings()` + `warnings.simplefilter("ignore", UserWarning)` in the call site
- **Interactions API agent IDs**: The SDK type hint only knows `"deep-research-pro-preview-12-2025"` as of google-genai 1.73. Agent ID is configured in `settings.yaml` under `research.providers.gemini.model` — update there to switch agents
- **Research make_cache_key location**: `make_cache_key()` lives in `src/research/merger.py`, NOT `src/research/cache.py`
- **Windows /dev/null**: Use `io.StringIO()` for Console mocking in tests, not `open("/dev/null", "w")`

## Known issues

- o3-deep-research integration test not run (blocked — $10+ per run)
- gemini deep-research integration test not run (blocked — takes 5-20 min per call)


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

## Lessons Discovery

Set `DEV_KNOWLEDGE_PATH` to point to the .dev-knowledge repo:

```powershell
$env:DEV_KNOWLEDGE_PATH = "C:/Users/1028120/Documents/Dev/.dev-knowledge"
```

**Where lessons go:**

- ai-council-local lessons (provider quirks, SDK gotchas, CLI patterns specific to this repo) → `tasks/lessons.md` (stays here)
- Cross-ecosystem lessons (universalizable methodology, governance patterns) → `$DEV_KNOWLEDGE_PATH/LESSONS.md`

**Criterion:** lesson applies only to ai-council code → stays local. Lesson applies across repos → flows to .dev-knowledge.

**ADR naming:** future ADRs use underscore convention (`ADR-NN_topic.md`) per ADR-34. The 7 existing kebab-case ADRs (`ADR-01-synthesizer-selection.md` etc.) are grandfathered per ADR-29 — do not rename.

## Global Skills
Before modifying code, consult ~/.claude/skills/gotchas/ for known ecosystem traps.
After pytest passes, check ~/.claude/skills/verify/ for verification scripts.

