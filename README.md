# AI Council

You have a hard architectural decision to make. Instead of asking one AI and hoping for the best, AI Council sends the question to a panel of top models simultaneously — Claude, Gemini, GPT, Grok, DeepSeek — lets them argue, critique each other's reasoning anonymously, then has a separate model synthesize the final verdict.

The result is a structured decision document: consensus points, unresolved disagreements, a recommended path forward, risks, and action items.

## Features

- **Multi-model debate** — parallel rounds with 2-5 AI models
- **Blind voting** — critique rounds use anonymized proposals to prevent bias
- **Adversarial personas** — each model has a specialized perspective (Systems, Security, Performance, Product, Contrarian)
- **Non-participating synthesizer** — a model outside the debate panel renders the final verdict
- **Inbox mode** — drop `.md` files into a folder for batch processing with YAML frontmatter overrides
- **Health checks** — providers are pinged before debate starts; unhealthy ones are skipped
- **Structured output** — markdown reports saved to `output/` with full transcripts and panel metadata

---

## How it works

1. **You ask a question** — on the command line, from a file, or by dropping `.md` files into an inbox folder
2. **The panel debates** — each model gives its position in parallel (Round 1), then critiques the others' anonymized answers (Round 2+)
3. **A non-participating model synthesizes** — a model that wasn't in the debate reads the full transcript and renders a verdict
4. **You get a markdown report** — saved to `output/` with the full transcript, panel metadata, and synthesis

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your keys
```

### API keys (`.env`)

```
ANTHROPIC_API_KEY=your-key
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key
XAI_API_KEY=your-key
DEEPSEEK_API_KEY=your-key
```

You don't need all five. Any models with missing keys are skipped. You need at least 2 panel models.

---

## Usage

### Single question

```bash
# Default panel: claude, gemini, deepseek (2 rounds)
python -m src.cli "Should we adopt a monorepo?"

# All 5 models
python -m src.cli "Microservices vs monolith?" --full

# Specific models
python -m src.cli "SQL or NoSQL?" --models claude,openai,grok

# Custom synthesizer
python -m src.cli "REST or GraphQL?" --synthesizer gemini

# From a markdown file
python -m src.cli --file question.md --rounds 3
```

### Inbox mode — batch processing

Drop `.md` files into `council_inbox/` and process them all at once:

```bash
python -m src.cli --inbox
```

Each file is archived to `council_inbox/archive/` after processing (prefixed `FAILED_` on error). Input files and archives are gitignored.

You can add YAML frontmatter to override settings per file:

```markdown
---
models: claude,openai
rounds: 1
---
Should we use Redis or Memcached for session caching?
```

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `QUESTION` | -- | Question to debate |
| `--file PATH` | -- | Read question from `.md` file |
| `--rounds N` | 2 | Number of debate rounds |
| `--full` | off | Use all 5 models |
| `--models LIST` | 3-model panel | Comma-separated: `claude,openai,grok` |
| `--synthesizer NAME` | `claude` | Model that writes the final verdict |
| `--output PATH` | `./output` | Where to save transcripts |
| `--inbox` | off | Process all files in `council_inbox/` |
| `--inbox-dir PATH` | `./council_inbox` | Override inbox folder |
| `--verbose` | off | Debug logging |

---

## Models

| Name | Provider | Default panel |
|------|----------|---------------|
| `claude` | Anthropic | yes |
| `gemini` | Google | yes |
| `deepseek` | DeepSeek | yes |
| `openai` | OpenAI | -- |
| `grok` | xAI | -- |

Each model has an adversarial persona baked in (Systems Architect, Security Architect, Performance Architect, etc.) to push disagreement and surface blind spots.

---

## Architecture

```
src/
  cli.py            — Click CLI entry point, panel/synthesizer selection
  debate.py         — Debate pipeline: parallel rounds, persona injection, blind voting
  synthesis.py      — Transcript assembly and synthesizer invocation
  output.py         — Rich console output + markdown file save
  models.py         — Data models (Question, ModelResponse, Round, DebateResult)
  healthcheck.py    — Provider health checks at startup
  inbox.py          — Inbox scanning, frontmatter parsing, auto-archive
  providers/        — One file per AI provider (Anthropic, OpenAI, Gemini, xAI, DeepSeek)
config/
  settings.yaml     — All model configs, prompts, personas (single source of truth)
  config_loader.py  — YAML to typed dataclasses
```

---

## Output

- **Console:** Round-by-round summaries + full synthesis rendered in the terminal
- **File:** `output/{timestamp}_{slug}.md` — full transcript with all rounds, panel metadata, and synthesis

---

## Tests

```bash
# Unit tests (no API keys needed)
pytest tests/ -m "not integration" -v

# Integration tests (requires 2+ keys in .env)
pytest tests/test_integration.py -v
```

72 unit tests covering all modules. Integration test runs a real debate with live API calls.

---

## Related repos

- corp-by-os — orchestrator
- corp-os-meta — shared schemas
- corp-knowledge-extractor — extraction engine
- corp-rfp-agent — RFP automation

## License

Internal use only — Blue Yonder Pre-Sales Engineering
