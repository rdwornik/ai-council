# Architectural Decision Records

ADRs documenting ai-council architectural choices. Kebab-case naming grandfathered (ADR-01 through ADR-07) per ADR-29 of `.dev-knowledge`. Future ADRs use underscore naming per ADR-34.

Cross-repo ADRs affecting routing semantics live in `.dev-knowledge/docs/decisions/` (e.g., ADR-43).

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-01](ADR-01-synthesizer-selection.md) | Synthesizer Selection | Revised (2026-03-29) |
| [ADR-02](ADR-02-default-panel.md) | Default Panel Composition | Accepted |
| [ADR-03](ADR-03-blind-voting.md) | Blind Voting in Round 2 | Accepted |
| [ADR-04](ADR-04-mode-system.md) | Mode System (pick/ideas/judge/research) | Accepted |
| [ADR-05](ADR-05-research-integration.md) | Research Mode Integration | Accepted |
| [ADR-06](ADR-06-cost-optimization.md) | Cost Optimization Strategy | Accepted (partial — Qwen trial pending) |
| [ADR-07](ADR-07-dual-output-paths.md) | Dual Output Paths | Superseded by ADR-43 (opt-in `target-project` routing replaces always-on secondary write) |

## Cross-repo references

- **ADR-43** (Cross-project transcript routing) — lives in `.dev-knowledge/docs/decisions/`. Amendments tracked there. ai-council implementation reflects amendment cycle 1 (schema: `dev_root` + `target_projects` list, not old `dict[name, path]`).
