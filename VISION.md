---
version: "1.0"
owner: rob
last_reviewed: "2026-05-27"
status: active
---

# VISION — ai-council

## Mission

Multi-model AI debate and research tool for architectural decision-making across the dev ecosystem. Produces binding ADRs that govern all repos under `Dev/`. Standalone CLI tool consumed by other repos via the `council` entry point.

## Scope

- **5 debate providers**: Claude Opus, Gemini, GPT, Grok, DeepSeek
- **5 research providers**: Perplexity, Gemini Deep Research, OpenAI o4-mini deep research, Grok x_search, OpenAI o3 deep research
- **4 modes**: pick / ideas / judge / research
- **CLI entry point**: `council`
- **Synthesizer**: Gemini (deliberate selection, ADR-01)
- **Output paths (dual)**: operational metrics + transcripts in `ai-council/output/`; curated transcripts in `.dev-knowledge/docs/decisions/transcripts/`
- Standalone tool — invoked by other repos, not embedded as a library

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
- Static-maintenance posture is exception requiring explicit declaration
