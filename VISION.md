---
version: "1.0"
owner: rob
last_reviewed: "2026-06-02"
status: active
---

# VISION — ai-council

<!-- scope: meta -->

## Vision

Multi-model AI debate and research tool for architectural decision-making across the dev
ecosystem. Produces binding ADRs that govern all repos under `Dev/`. Standalone CLI tool
consumed by other repos via the `council` entry point.

## Scope

- **5 debate providers**: Claude Opus, Gemini, GPT, Grok, DeepSeek
- **5 research providers**: Perplexity, Gemini Deep Research, OpenAI o4-mini deep research, Grok x_search, OpenAI o3 deep research
- **4 modes**: pick / ideas / judge / research
- **CLI entry point**: `council`
- **Synthesizer**: Gemini (deliberate selection, ADR-01)
- **Output paths (dual)**: operational metrics + transcripts in `ai-council/output/`; curated transcripts in `.dev-knowledge/docs/decisions/transcripts/`
- Standalone tool — invoked by other repos, not embedded as a library

## Values

- **Blind deliberation over authority** — Round-2 responses are anonymized and shuffled (ADR-03) so verdicts rest on argument, not provider reputation.
- **Config as the single source of truth** — model strings, prompts, personas, cost rates live only in `config/settings.yaml`, never hard-coded.
- **Fail loud at the boundary** — unknown routing targets and degraded research runs surface immediately (RoutingError; exit code 3) rather than failing silently.
- **Cost-aware deliberation** — panel size and rounds are tunable; `--lite` and per-provider cost tracking keep debate proportional to the question.

## Relationships

- No inbound code dependencies
- Called by other repos (browser chats, Claude Code in any project under `Dev/`) via the `council` CLI
- Produces binding ADRs governing ecosystem repos
- Reads `.dev-knowledge` PLAYBOOK conventions for output formatting and dual-path routing

## Lifecycle

- Active development with continuous improvement focus
- Roadmap reviewed at session boundaries; improvements emerge from real usage and lessons captured in `LESSONS.md`
- Recent additions: Grok as 5th research provider, downloads auto-scan, `--models` flag
- Review triggers: provider API changes, new model availability, governance requirement changes, real-usage friction surfacing improvement opportunities
- Static-maintenance posture is an exception requiring explicit declaration

## References

- `ARCHITECTURE.md` — structural model, layers, invariants
- `docs/decisions/` — local tool-design ADRs (ADR-01 onward; index in `docs/decisions/README.md`)
- `.dev-knowledge/protocols/AI_COUNCIL_PROCESS.md` — the end-to-end Council process this tool implements
- `BACKLOG.md` — the story-map of pending work
