---
scale: M
last_reviewed: 2026-05-19
status: active
owner: Rob
---

# Architecture — `ai-council`

> Living document. Updated after structural changes.
> Last updated: `2026-05-19` (`initial ADR-51 conformance authoring`)

## Purpose [CORE]

`ai-council` is a multi-model AI debate and research CLI with four operating modes (pick, ideas, judge, research). It coordinates a configurable panel of AI models (Anthropic, OpenAI, Google, xAI, DeepSeek) through structured rounds of parallel debate and blind-vote critique, synthesizes a final verdict from a non-participating synthesizer, and routes transcripts to both its operational archive and target project directories. Within the `Dev/` ecosystem it is the primary mechanism for producing Architecture Decision Records: Council debate produces a verdict; that verdict is authored into an ADR that is ratified and distributed by `.dev-knowledge` to all downstream repos.

---

## Codemap [CORE]

> **Open item.** The codemap generator output spec is undecided (ADR-51 open question). This codemap is hand-maintained in the transitional text form until the generator ships.

<!-- CODEMAP:START -->
```
src/ai_council/
  cli.py              CLI entry; Click args → RunRequest; health-check gate   (interface)
  inbox.py            File-based batch input; YAML frontmatter; archive        (interface)
  orchestrator.py     CouncilRunner; debate lifecycle coordinator               (orchestration)
  runner.py           Panel selection; provider init; mode resolution           (orchestration)
  debate.py           Parallel debate rounds; blind-vote critique; retry        (core)
  synthesis.py        Final synthesis; transcript assembly; DebateResult build  (core)
  mode_detector.py    LLM-based question classification (pick/ideas/judge)      (core)
  providers/          5 debate-provider implementations                         (core)
    base.py           AIProvider ABC; error classification
    anthropic.py      Claude (Anthropic SDK)
    openai_provider.py  GPT (OpenAI SDK)
    gemini.py         Gemini (google-genai SDK)
    xai.py            Grok (OpenAI-compatible base_url)
    deepseek.py       DeepSeek (OpenAI-compatible base_url)
  output.py           Rich console render + markdown archive write               (output)
  routing.py          TargetResolver; secondary transcript routing (ADR-43)      (output)
  models.py           Pure dataclasses; all shared data shapes; no logic         (foundation)
  metrics.py          Token-count + cost tracking per provider call              (foundation)
  healthcheck.py      Provider health checks; startup gate                       (foundation)
  policy.py           RunPolicy; retry backoff spec                              (foundation)
  research/           Research mode subsystem (isolated code path)               (core — research)
    runner.py         Research orchestrator: cache → parallel → merge → summarize
    provider.py       ResearchProvider ABC
    cache.py          File-based cache (TTL; dedup by question hash)
    merger.py         Dedup sources; synthesize report
    models.py         ResearchResult, MergedResearchReport, Source
    display.py        Live progress display
    output.py         Save research results to markdown
    providers/        5 research providers (Perplexity/OAI-mini/OAI-deep/Gemini/Grok)
config/
  config_loader.py    Type-safe YAML → dataclass loaders; appConfig singleton   (foundation)
  settings.yaml       Single config source of truth: models, prompts, personas  (foundation)
```
<!-- CODEMAP:END -->

---

## Layer Boundaries & Invariants [CORE]

### Layer model

Four layers in dependency order (interface → orchestration → core → foundation). `output` is a cross-cutting concern that writes results produced by `core`.

**Enforcement tool:** Convention + code review. No automated import-linter at current scale (no Tach).  
**Config file:** N/A  
**Where enforced:** Code review; `scripts/check.ps1` (pytest + mypy + ruff)

### Module-to-layer assignment

| Layer | Modules |
|-------|---------|
| `interface` | `cli.py`, `inbox.py` |
| `orchestration` | `orchestrator.py`, `runner.py` |
| `core` | `debate.py`, `synthesis.py`, `mode_detector.py`, `providers/`, `research/` |
| `foundation` | `models.py`, `metrics.py`, `healthcheck.py`, `policy.py`, `config/` |
| `output` (cross-cutting) | `output.py`, `routing.py` |

Utility-exemption modules (importable from any layer): none.

### Invariants

1. `models.py` contains no logic — pure dataclasses only; the sole source of shared data shapes across all layers.
2. `cli.py` performs no business logic — it translates CLI arguments into a `RunRequest` and immediately delegates to `CouncilRunner`.
3. Config strings (model names, prompts, personas, cost rates) are defined solely in `config/settings.yaml` — none hard-coded in Python source.
4. `_anonymize_responses()` in `debate.py` is the sole implementation of blind voting (ADR-03); its shuffle order is part of the contract and must not be altered without an ADR.
5. The synthesizer is a non-participating observer — it receives the full transcript but does not vote in debate rounds, and must not be a member of the debating panel.
6. The research subsystem (`research/`) is an isolated code path — features added to interactive debate mode are not automatically available to research mode; they must be explicitly mirrored.
7. All mutable session state flows through `DebateResult` and `RunRequest` dataclasses — provider implementations are stateless between calls.

→ Related decisions: `docs/decisions/ADR-03-blind-voting.md`, `docs/decisions/ADR-04-mode-system.md`, `docs/decisions/ADR-05-research-integration.md`, `docs/decisions/ADR-07-dual-output-paths.md`

---

## Data Flow [L-opt]

End-to-end debate pipeline:

```
1. Input arrives via CLI (`council "question"`) or inbox file
       |
2. mode_detector.py classifies the question (pick / ideas / judge / research)
       |
3. runner.py selects panel + resolves synthesizer; healthcheck.py gates startup
       |
4. debate.py runs Round 1: parallel provider calls; anonymizes + shuffles responses (ADR-03)
       |
5. debate.py runs critique rounds: each panellist sees anonymized responses from prior round
       |
6. synthesis.py calls non-participating synthesizer on full transcript; builds DebateResult
       |
7. output.py writes Rich console display + markdown archive → ai-council/output/
       |
8. routing.py optionally copies curated transcript to target project dir (ADR-43)
```

Research mode branches at step 2: `research/runner.py` → cache check → parallel provider calls → `research/merger.py` → `research/output.py`. If fewer than 3 providers succeed, run continues but exits with code 3 and an alarm banner (ADR-08).

---

**Maintained by:** Rob
