# AI Council

You have a hard architectural decision to make. Instead of asking one AI and hoping for the best, AI Council sends the question to a panel of top models simultaneously — Claude, Gemini, GPT, Grok, DeepSeek — lets them argue, critique each other's reasoning anonymously, then has a separate model synthesize the final verdict.

The result is a structured decision document: consensus points, unresolved disagreements, a recommended path forward, risks, and action items.

---

## How it works

1. **You ask a question** — on the command line, from a file, or by dropping `.md` files into an inbox folder
2. **The panel debates** — each model gives its position in parallel (Round 1), then critiques the others' anonymized answers (Round 2+)
3. **A non-participating model synthesizes** — a model that wasn't in the debate reads the full transcript and renders a verdict
4. **You get a markdown report** — saved to `output/` with the full transcript, panel metadata, and synthesis

---

## Example

```bash
python -m src.cli "Should we use REST or GraphQL for our public API?" --rounds 2
```

**Console output:**
- Round summaries (each model's position at a glance)
- Full synthesis from the non-participating judge

**Saved to** `output/20260223_143012_should-we-use-rest-or-graphql.md`:
```
## Consensus
All participants agreed that REST is better suited for public APIs due to...

## Unresolved Disagreements
Claude and Grok disagreed on schema flexibility — Claude prioritized...

## Recommended Decision
Use REST. The team's operational maturity and existing tooling outweigh...

## Risks
GraphQL's flexibility may be needed as the API grows. Revisit if...

## Action Items
- Define versioning strategy for REST endpoints
- ...
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

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
| `QUESTION` | — | Question to debate |
| `--file PATH` | — | Read question from `.md` file |
| `--rounds N` | 2 | Number of debate rounds |
| `--full` | off | Use all 5 models |
| `--models LIST` | 3-model panel | Comma-separated: `claude,openai,grok` |
| `--synthesizer NAME` | `openai` | Model that writes the final verdict |
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
| `openai` | OpenAI | — |
| `grok` | xAI | — |

Each model has an adversarial persona baked in (Systems Architect, Security Architect, Performance Architect, etc.) to push disagreement and surface blind spots.

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
