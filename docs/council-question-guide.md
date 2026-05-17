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
| models | `claude,gemini,openai,deepseek,grok` | Default = all 5. Use `--lite` for 3-model (claude, gemini, openai) on simple questions |
| synthesizer | `gemini` | Default = gemini ($0.04). Use `openai` if gemini is already on the panel |
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

## Neutralizing bias in question framing

A biased question is the one failure mode the Council cannot recover from.
Blind voting (ADR-03), the multi-model panel (ADR-02), and adversarial personas
are all *downstream* defenses — they reduce how much models bias each other and
how the synthesizer distorts the result. None of them can repair a biased
question: every model on the panel reads the same prompt, so a question that
leans one way leans the whole debate that way, and the votes then agree on a
contaminated basis. Question framing is the only bias-control point with no
safety net. It is the highest-leverage step in running a debate.

The Council's success metric is the elimination of cognitive bias. This section
is the checklist that serves that metric directly.

### Framing biases and how to neutralize them

**Leading / framing.** The headline presupposes an answer. "Should we adopt
ChromaDB for the vault?" has already chosen ChromaDB as the frame and reduced
the debate to yes/no. *Fix:* state the **problem**, not a candidate solution —
"What storage layer best fits the vault's access patterns?" Candidate solutions
belong in the options, never in the headline.

**Anchoring.** The first option, or the one described at greatest length,
becomes the reference point the others are judged against. *Fix:* describe every
option at equal length and equal depth. Order options neutrally (alphabetical,
or state explicitly that the order is arbitrary). Do not editorialize inside an
option description.

**Confirmation / asker-leakage.** The question reveals which way the asker leans
— directly ("I think we should keep the monorepo") or subtly ("the obvious
choice is X, but…"). Models drift toward agreement with the perceived asker
preference. Blind voting does not catch this: it hides models from *each other*,
not from the asker's leak. *Fix:* never state your current leaning, a prior
decision, or what you hope the answer is. Strip "I think", "probably",
"ideally", "obviously". Present the decision as genuinely undecided — for the
debate to be worth running, it is.

**False dichotomy.** The question forces a binary where a hybrid or third path
exists. "Tach or code review?" excludes "both", "neither", and "a lighter
linter". *Fix:* ask which mechanisms fit the situation, not which of two to
pick; or explicitly admit hybrid and "none" as legitimate options.

**Loaded terminology.** Value-laden words prejudge the outcome. "Should we clean
up the bloated ingest module?" has already ruled that the module is bad and that
reduction is the answer. *Fix:* replace adjectives with observable facts — line
count, number of responsibilities, measured coupling — and let the panel judge.

**Choice-set bias.** The option you did not think of cannot win, because it is
not on the ballot. A three-option `pick` debate silently decides that those
three are the universe. *Fix:* always include an explicit escape option — "a
different approach (name it)" — and instruct the panel to use it if warranted.
For consequential decisions, run an `ideas` round first to generate the option
set before the `pick` debate that chooses among it. The choice set is itself a
decision; make it deliberately.

**Availability.** Framing the question around the most recent or most salient
incident lets that one event dominate. "After the ChromaDB migration pain,
should we avoid vector DBs?" *Fix:* ask the general case. If a specific incident
motivates the question, name it as one data point, not as the frame.

### Pre-flight self-check

Before submitting any Council question, in any mode, answer these:

1. Does the headline name a **problem**, or a pre-chosen answer?
2. Does any wording reveal which option I prefer?
3. Is every option described at equal length and equal charity?
4. Could a reasonable option be missing from the set?
5. Are there value-laden adjectives I can replace with observable facts?
6. If the panel agreed with me instantly — would that be because I am right, or
   because I told them what I wanted?

Question 6 is the sharpest test. If a fast unanimous agreement would not
surprise you, the question is probably leading.

### Research mode: the bias is sharper

In `research` mode a biased question does not just skew opinion — it pre-selects
which **evidence** gets surveyed. "What evidence supports event-driven
architecture?" returns a curated pro-list and never looks for disconfirming
evidence. *Fix:* make the question symmetric — "What does the evidence say about
event-driven versus alternative architectures for X?" — so disconfirming results
and null findings have a path into the answer.

---

## Research-mode questions

Research mode is the most commonly mis-formulated mode. Authors who want a
research debate often write the question as a decision question, and the
Council answers in kind — producing opinion for a specific situation instead
of a survey of external evidence. This section gives a recognition test and
formulation rules to prevent that.

### Recognition test

A `research`-mode question is identified by the *output wanted*, not the topic:

- **Research mode** — the output is a survey of what the field, industry, or
  literature knows about something. The asker applies that survey to their own
  situation themselves.
- **Decision mode** (`pick` / `judge` / `ideas`) — the output is a decision or
  recommendation for the asker's specific situation.

Quick test: if the question asks *"what should I do?"*, it is **not** research
mode. If it asks *"what does the field know about X?"*, it **may be** research
mode.

### Formulation rules

When a question is genuinely research mode, formulate it so the Council can
survey evidence rather than reason from first principles:

- **The headline asks what the field knows**, not what the asker should do.
  "How do production RAG systems handle stale embeddings?" — not "Should we
  re-embed nightly?"
- **Options are evidence-testable candidate approaches.** Name real systems,
  real tools, or real studies wherever possible rather than abstract positions —
  the Council can then check each against published evidence.
- **Source-corpus constraints are valid and encouraged.** Specify recency
  windows (e.g. "last 3 years"), exclude marketing material, and distinguish
  peer-reviewed from practitioner sources where the distinction matters.
- **The question must be answerable by surveying external evidence.** If it can
  only be answered by reasoning from first principles, it is a decision
  question, not a research question.

### The breadth-over-depth trap

A research question with more than three distinct sub-questions dilutes evidence
depth. The Council will attempt to cover every sub-question and produce thin
coverage of each. If a research question has more than three sub-questions,
either:

- **split it** into separate debates, or
- **explicitly instruct** the Council to prioritize the best-evidenced
  sub-questions and go deep rather than wide.

---

## Choosing the Panel

| Situation | Panel |
|-----------|-------|
| Important / standard decision | Default: all 5 models |
| Quick opinion | 3 models via `--lite` (claude, gemini, openai) |
| Cost-sensitive | `--lite` + `--rounds 1` |
| Custom selection | `--models claude,openai,grok` |

**Synthesizer rule:** The synthesizer must NOT be on the panel. Default: gemini (cheapest at $0.04/debate). Gemini is always on the full default panel, so the default synthesizer already satisfies this — it picks a non-participant automatically.

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

---

## After every debate — distil the transcript to an ADR

A council debate produces a decision, and a decision must be recorded. After a run completes, distil the synthesizer's verdict into a lightweight ADR (Status / Context / Decision / Consequences) in the target project's `docs/decisions/`. The transcript in `docs/decisions/transcripts/` is the evidence; the ADR is the canonical record. This step is part of the council workflow, not optional — a debate is not done until its ADR exists.
