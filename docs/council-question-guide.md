# How to Write an AI Council Debate Question

---

## The Format

Every debate question is a markdown file with three parts:

```markdown
---
models: claude,gemini,deepseek,grok
synthesizer: openai
rounds: 2
---

## Question: [one clear sentence]

### Current State
[concrete facts — numbers, specific problems, what exists today]

### Questions
1. **[sub-question]**
   - A: [option — one sentence, includes key trade-off]
   - B: [option]
   - C: [option]
   - D: [option]

2. **[sub-question]**
   - A: [option]
   - B: [option]
   - C: [option]

### Constraints
- [hard constraint]
- [hard constraint]
- [hard constraint]
```

That's it. Nothing else.

---

## YAML Frontmatter — the config

| Field | Options | When to change from default |
|-------|---------|---------------------------|
| models | `claude,gemini,deepseek,grok` | Default = all 4. Use 3 for simple questions: `claude,gemini,openai` |
| synthesizer | `gemini` | Default = gemini ($0.04). Use `openai` if gemini is on the panel |
| rounds | `2` | Default = 2. Use `1` for simple pick decisions. Never more than 2. |
| mode | `pick` / `ideas` / `judge` / `research` | Usually omit — auto-detected. Force `ideas` for brainstorming, `judge` for evaluating a proposal |
| target-project | project name string or list | Omit for local-only output. Set to mirror transcript to a project's `docs/decisions/transcripts/` dir. |

---

## Cross-project routing (`target-project`)

By default, transcripts are written only to `output/`. To also mirror to a named project directory:

**Inbox mode** — add `target-project:` to YAML frontmatter:

```markdown
---
mode: judge
target-project: .dev-knowledge        # single target
# or:
# target-project: [.dev-knowledge, corp-monorepo]
---
Question body...
```

**Direct CLI mode** — pass `--target-project NAME` flag (repeatable):

```bash
council --target-project .dev-knowledge --mode pick "..."
council --target-project .dev-knowledge --target-project corp-monorepo "..."
```

Target names must be declared in `config/settings.yaml` under `target_projects` with a single `dev_root` path. Unknown name → fail-loud at parse time, listing known names.

See `README.md` Transcript Routing section for full schema and config examples.

---

## The Question — one sentence

Good: *"How should we restructure the Obsidian vault after bulk ingestion of 587 messy extraction notes?"*

Good: *"How should Rob handle the transition from a full browser chat to a new one?"*

Bad: *"I need help deciding about the vault and also there are some quality issues and maybe we should think about..."*

**Rule:** If you can't state the question in one sentence, you're asking two questions. Split them into two debates.

---

## Current State — concrete facts, not opinions

This is the most important section. The panel can only reason as well as the context you give them.

**Include:**
- Numbers: "587 notes", "93.7% accuracy", "30/30 memory slots full"
- Specific problems: "193 notes have doc_type: general (33%)"
- What exists today: paths, tools, file counts, current config
- What changed recently: "since the bulk ingest, we improved X, Y, Z"
- Real examples when they help

**Don't include:**
- Your opinion on what the answer should be (that biases the panel)
- Long narrative about how you got here
- Instructions on how to debate (the system handles that)

---

## Questions — structured options, not open-ended

**Good question:**
```
1. **Revert or clean incrementally?**
   - A: Delete all 587 notes, re-extract with improved pipeline. Clean start, $5-10 API cost.
   - B: Keep notes, improve incrementally. Old notes stay dirty.
   - C: Delete all, re-ingest ONLY v3 outputs. V2 goes to quarantine.
   - D: Delete all, re-ingest v3, then re-EXTRACT v2 through new pipeline (most expensive, highest quality).
```

**Bad question:**
```
1. What do you think we should do about the vault? Consider the trade-offs
   and think about ADHD implications and also what's the 6-month view?
```

**Rules for questions:**
- 2-4 options per question (A/B/C/D). Never more than 5.
- Each option is one sentence with the key trade-off baked in
- Options should be genuinely different approaches, not variations of the same thing
- Include the "do nothing" option when it's viable
- Include cost/effort hints inline: "(most expensive)", "($5-10 API cost)", "(zero config needed)"
- 3-7 questions per debate. Under 3 = probably too simple for Council. Over 7 = split into two debates.

---

## Constraints — hard boundaries, not preferences

**Good constraints:**
```
- Solo developer, ADHD
- Memory full (30/30 slots)
- Zero Obsidian plugins (Council binding decision)
- Re-extraction costs $0.001-0.04 per file
```

**Bad constraints:**
```
- I'd prefer something simple
- It should probably be easy to use
- Would be nice if it worked across platforms
```

**Rule:** Constraints are things that ELIMINATE options. "Solo developer, ADHD" kills anything requiring daily manual discipline. "Zero plugins" kills any Obsidian plugin solution. If a constraint doesn't eliminate at least one option, it's not a constraint — it's a preference. Remove it.

---

## What NOT to Put in a Council Debate

| Don't include | Why | That belongs in... |
|---------------|-----|-------------------|
| `Model / Mode / Effort` table | That's Claude Code prompt format | Claude Code prompts |
| `Read CLAUDE.md first` | Panel doesn't read your files | nowhere — remove it |
| `UNDERSTAND section` | That's prompt engineering | Claude Code prompts |
| `WHAT NOT TO DO` | You're telling the panel how to think | nowhere — let them debate |
| `Steps with COMMIT markers` | That's execution instructions | Claude Code prompts |
| `Git workflow` | Panel doesn't write code | Claude Code prompts |
| Long narrative context | Panel needs facts, not stories | trim to Current State bullets |
| Your preferred answer | Biases the panel | keep it to yourself until synthesis |
| "Questions for Panel" as open discussion | Panel needs options to pick from | rewrite as A/B/C/D options |

---

## Choosing the Right Mode

| Situation | Mode | Example |
|-----------|------|---------|
| "Which option should we pick?" | `pick` (default) | "Monorepo vs polyrepo?" |
| "What approaches exist?" | `ideas` | "What caching strategies should we consider?" |
| "Is this plan good?" | `judge` | "Is this microservices design production-ready?" |
| "What does the community/research say?" | `research` | "Best HTAP databases in 2026" |

Usually let auto-detection handle it. Force mode only when auto-detection would guess wrong.

---

## Choosing the Panel

| Situation | Panel |
|-----------|-------|
| Important architectural decision | `--full` (all 5 models) |
| Standard decision | Default 3: claude, gemini, openai |
| Quick opinion | 2 models: `--models claude,gemini` + `--rounds 1` |
| Cost-sensitive | Default 3 + `--rounds 1` |

**Synthesizer rule:** The synthesizer must NOT be on the panel. Default: gemini (cheapest at $0.04/debate). If gemini is already on the panel (default), it still works but set `synthesizer: openai` for true non-participation.

---

## Size Guide

| Section | Target length |
|---------|--------------|
| YAML frontmatter | 3-5 lines |
| Question headline | 1 sentence |
| Current State | 10-20 lines (facts + numbers) |
| Questions | 3-7 questions × 3-4 options each |
| Constraints | 4-8 bullet points |
| **Total** | **40-80 lines** |

If your debate file is over 100 lines, you're including narrative that should be facts. Trim.

---

## Quick Template

Copy this, fill in, save as `.md` in `council_inbox/`:

```markdown
---
models: claude,gemini,deepseek,grok
synthesizer: openai
rounds: 2
---

## Question: [one sentence]

### Current State

[What exists today — concrete facts, numbers, paths, tools]

[What changed recently — if relevant]

### Questions

1. **[sub-question]?**
   - A: [option with trade-off]
   - B: [option with trade-off]
   - C: [option with trade-off]

2. **[sub-question]?**
   - A: [option]
   - B: [option]
   - C: [option]

### Constraints

- [hard constraint that eliminates options]
- [hard constraint]
- [hard constraint]
```

Then run:
```bash
python -m ai_council.cli --inbox
```
