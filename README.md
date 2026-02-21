# AI Council

Multi-model architectural debate tool. Sends a question to a panel of AI models in parallel, runs critique rounds where each model sees anonymized versions of others' answers, then a non-participating model synthesizes the final decision.

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### API Keys (`.env`)

```
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
XAI_API_KEY=your-key
DEEPSEEK_API_KEY=your-key
```

Any keys you provide will be used. Models with missing keys are skipped automatically. At least 2 panel models must be available.

> **Note:** Always run commands from inside the activated venv. The `venv/` directory is gitignored.

## Usage

```bash
# Default 3-model panel: claude, gemini, deepseek (2 rounds)
python -m src.cli "Should we use REST or GraphQL?"

# Full 5-model panel: all models
python -m src.cli "Monorepo vs polyrepo?" --full

# Custom model list
python -m src.cli "SQL or NoSQL?" --models claude,openai

# Read question from file
python -m src.cli --file question.md

# Custom rounds and output directory
python -m src.cli --rounds 3 --output ./debates "Should we use microservices?"

# Override synthesizer
python -m src.cli --synthesizer gemini "REST vs GraphQL?"

# Verbose logging (shows anonymization map, full prompts at DEBUG)
python -m src.cli --verbose "Test question"
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `QUESTION` | — | Question to debate (or use `--file`) |
| `--file PATH` | — | Read question from `.md` file |
| `--rounds N` | 2 | Number of debate rounds |
| `--full` | off | Use all 5 models instead of default 3-model panel |
| `--models LIST` | panel default | Comma-separated override: `claude,openai,grok` |
| `--output PATH` | `./output` | Output directory for transcripts |
| `--synthesizer NAME` | `openai` | Model that synthesizes (picked outside panel when possible) |
| `--verbose` | off | Enable DEBUG logging |

## Output

- **Console:** Round summaries (first ~50 words per model) + full synthesis via Rich
- **File:** `output/{timestamp}_{slug}.md` — full transcript with Panel/Mode/Synthesizer metadata header, all rounds, and synthesis

## Running Tests

```bash
# Activate venv first, then:

# Unit tests (no API keys needed)
pytest tests/ -m "not integration" -v

# Integration tests (requires 2+ API keys in .env)
pytest tests/test_integration.py -v
```

## Architecture

```
config/
  settings.yaml       — models, prompts, personas, panel lists (single source of truth)
  config_loader.py    — YAML -> typed dataclasses; API key detection at startup
src/
  models.py           — Question, ModelResponse, Round, DebateResult (pure dataclasses)
  debate.py           — parallel async rounds; persona injection; blind voting (anonymized critiques)
  synthesis.py        — builds transcript; calls non-participating synthesizer
  output.py           — Rich console + markdown file save
  cli.py              — Click CLI; panel/synthesizer selection logic
  providers/
    base.py           — AIProvider ABC + ProviderError
    anthropic.py      — Claude (Anthropic SDK)
    gemini.py         — Gemini (google-genai SDK)
    openai_provider.py — GPT (openai SDK)
    xai.py            — Grok (OpenAI-compatible)
    deepseek.py       — DeepSeek (OpenAI-compatible)
tests/
  conftest.py         — MockProvider, shared fixtures
  test_config.py / test_debate.py / test_synthesis.py / test_output.py / test_cli.py
  test_integration.py — end-to-end with real API calls (marked integration)
```
