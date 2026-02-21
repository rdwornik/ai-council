# AI Council

Multi-model architectural debate tool. Sends a question to 2–4 AI models in parallel, runs critique rounds where each model sees and challenges others' answers, then synthesizes a final decision with consensus, disagreements, risks, and action items.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API keys
```

### API Keys (`.env`)

```
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
XAI_API_KEY=your-key
```

Any keys you provide will be used. Models with missing keys are skipped. At least 2 models must be available.

## Usage

```bash
# Basic debate (all available models, 2 rounds)
python -m src.cli "Should we use YAML or JSON for config?"

# Read question from file
python -m src.cli --file question.md

# Custom rounds and models
python -m src.cli --rounds 3 --models gemini,openai "Should we use microservices?"

# Custom output directory
python -m src.cli --output ./debates "What database should we use?"

# Choose synthesizer
python -m src.cli --synthesizer gemini "REST vs GraphQL?"

# Verbose logging
python -m src.cli --verbose "Test question"
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `QUESTION` | — | Question to debate (or use `--file`) |
| `--file PATH` | — | Read question from `.md` file |
| `--rounds N` | 2 | Number of debate rounds |
| `--models LIST` | all available | Comma-separated: `gemini,openai,claude,grok` |
| `--output PATH` | `./output` | Output directory for transcripts |
| `--synthesizer NAME` | `claude` | Model that synthesizes the final decision |
| `--verbose` | off | Enable DEBUG logging |

## Output

- **Console:** Round summaries (first ~50 words per model) + full synthesis via Rich
- **File:** `output/{timestamp}_{slug}.md` — full transcript with all rounds + synthesis

## Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/ -m "not integration" -v

# Integration tests (requires 2+ API keys in .env)
pytest tests/test_integration.py -v
```

## Architecture

```
config/config_loader.py  → loads settings.yaml into typed dataclasses
src/models.py            → Question, ModelResponse, Round, DebateResult
src/providers/           → one file per AI SDK (gemini, openai, anthropic, xai)
src/debate.py            → parallel async rounds with asyncio.gather
src/synthesis.py         → builds transcript, calls synthesizer
src/output.py            → Rich console + markdown file save
src/cli.py               → Click CLI, wires everything together
```
