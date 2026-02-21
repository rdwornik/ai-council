# AI Council — Architecture Reference

## Project Overview

Multi-model AI debate tool. A question is posed; multiple AI models argue in parallel rounds; a non-participating synthesizer produces the final decision summary.

## Phase 1 Checklist (complete)

- [x] Config loading (settings.yaml → typed dataclasses)
- [x] Provider abstraction (AIProvider ABC)
- [x] All providers: OpenAI, Anthropic, Gemini, xAI/Grok, DeepSeek
- [x] Debate pipeline (parallel async calls, critique rounds)
- [x] Persona injection (per-model adversarial roles)
- [x] Blind voting (anonymized critique prompts)
- [x] Non-participating synthesizer selection
- [x] Default 3-model panel + `--full` 5-model flag
- [x] Rich console output + markdown file save
- [x] Full unit test suite

## Architecture

```
src/
  cli.py              — Click entry point; panel/synthesizer selection
  debate.py           — run_debate(); persona injection; blind voting
  synthesis.py        — synthesize(); builds transcript; calls synthesizer
  output.py           — save_to_file(); print_round_summary(); print_synthesis()
  models.py           — Pure dataclasses: Question, ModelResponse, Round, DebateResult
  providers/
    base.py           — AIProvider ABC + ProviderError
    anthropic.py      — Claude (Anthropic SDK)
    gemini.py         — Gemini (google-genai SDK)
    openai_provider.py — GPT (openai SDK)
    xai.py            — Grok (OpenAI-compatible)
    deepseek.py       — DeepSeek (OpenAI-compatible)
config/
  settings.yaml       — All model configs, prompts, personas, defaults (single source of truth)
  config_loader.py    — YAML → typed dataclasses; API key detection at startup
tests/
  conftest.py         — MockProvider, shared fixtures
  test_config.py      — Config loading, panels, personas
  test_debate.py      — Anonymization, persona injection, round logic
  test_synthesis.py   — Synthesis output, panel_mode, is_participant fields
  test_output.py      — Markdown output format, header fields
  test_cli.py         — Panel determination, synthesizer selection logic
  test_models.py      — Dataclass field validation
  test_integration.py — End-to-end with real API calls (marked `integration`)
```

## Key Design Decisions

### Panel System
- **Default panel**: `["claude", "gemini", "deepseek"]` — 3 models, balanced perspectives
- **Full panel**: `["claude", "gemini", "deepseek", "openai", "grok"]` — all 5 models
- **Custom**: `--models claude,openai` — explicit override
- `_determine_panel()` in cli.py; `--models` always wins over `--full`

### Persona System
- Each provider has an adversarial persona defined in `settings.yaml` under `personas:`
- Personas are loaded into `PromptsConfig.personas` (dict: provider_name → text)
- Injected via `{persona}` placeholder in `initial` and `critique` prompt templates
- Fallback: `prompts.personas.get(p.name(), "")` — empty string if no persona defined

### Blind Voting (Anonymization)
- Critique rounds use `_anonymize_responses()` in `debate.py`
- Responses are shuffled randomly and labeled "Proposal A", "Proposal B", etc.
- Provider names and model strings are **not** included in the critique prompt
- Label→provider mapping is logged at DEBUG only (not stored in DebateResult)
- Template placeholder: `{previous_responses_anonymized}` (not `{previous_responses}`)

### Non-Participating Synthesizer
- `_pick_non_participant_synthesizer()` in cli.py selects a synthesizer outside the panel
- Preferred synthesizer (from config or `--synthesizer`) is used if available outside panel
- If no non-participant is available (all providers in panel), falls back to preferred with `is_participant=True`
- `DebateResult.synthesizer_is_participant` tracks this; displayed in output header

### Provider Pattern
- Each provider is a separate file (no shared base class beyond `AIProvider` ABC)
- OpenAI-compatible providers (xai.py, deepseek.py) are structurally identical — do NOT merge
- `PROVIDER_CLASSES` dict in cli.py maps name → class

### Config as Single Source of Truth
- All model strings, timeouts, max_tokens live in `settings.yaml`
- `prompts:` section holds all three templates; `personas:` holds per-model roles
- `defaults:` holds panel lists, round counts, synthesizer preference

## Provider SDK Notes

- **anthropic**: `client.messages.create()` via `asyncio.to_thread`
- **google-genai**: `client.aio.models.generate_content()` — native async, NOT `asyncio.to_thread`
- **openai**: `AsyncOpenAI` client, `chat.completions.create()`
- **openai-compatible** (xai, deepseek): same as openai but with `base_url`

## Gotchas

- **Windows cp1252**: Do not print Unicode chars (✓, →, etc.) in Rich progress callbacks. Use ASCII (`OK`, `>`, etc.)
- **MockProvider ABC**: `async def generate` must exist in the class body AND be shadowed by `AsyncMock` in `__init__` for ABC compliance
- **pytest-asyncio**: Needs `asyncio_mode = auto` in `pytest.ini`
- **Synthesis prompt**: `{rounds}` placeholder removed from synthesis template; `synthesize()` still passes `rounds=` kwarg (Python ignores extra kwargs in `.format()`)
- **Critique template rename**: Uses `{previous_responses_anonymized}`, not `{previous_responses}`

## CLI Usage

```bash
# Default 3-model panel
python -m src.cli "Should we use REST or GraphQL?" --rounds 1

# Full 5-model panel
python -m src.cli "Monorepo vs polyrepo?" --rounds 1 --full

# Custom models
python -m src.cli "SQL or NoSQL?" --rounds 1 --models claude,openai

# From file
python -m src.cli --file question.md --rounds 3

# Debug prompts
python -m src.cli "question" --rounds 1 --verbose
```

## Testing

```bash
# Unit tests only (no API keys needed)
pytest tests/ -m "not integration" -v

# Integration test (requires API keys)
pytest tests/test_integration.py -v
```
