# Architectural Decision Records

ADRs documenting ai-council architectural choices. Hyphen (kebab-case) naming per ADR-34 of `.dev-knowledge`. Future ADRs use hyphen naming per ADR-34.

Cross-repo ADRs affecting routing semantics live in `.dev-knowledge/docs/decisions/` (e.g., ADR-43).

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-01](ADR-01-synthesizer-selection.md) | Synthesizer Selection | Revised (2026-07-18) |
| [ADR-02](ADR-02-default-panel.md) | Default Panel Composition | Revised (2026-05-11) |
| [ADR-03](ADR-03-blind-voting.md) | Blind Voting in Round 2 | Accepted |
| [ADR-04](ADR-04-mode-system.md) | Mode System (pick/ideas/judge/research) | Accepted |
| [ADR-05](ADR-05-research-integration.md) | Research Mode Integration | Accepted |
| [ADR-06](ADR-06-cost-optimization.md) | Cost Optimization Strategy | Revised 2026-05-11 (Qwen trial deferred/abandoned) |
| [ADR-07](ADR-07-dual-output-paths.md) | Dual Output Paths | Superseded by ADR-43 (opt-in `target-project` routing replaces always-on secondary write) |
| [ADR-08](ADR-08-research-degradation-alarm.md) | Research-panel degradation alarm | Accepted |
| [ADR-09](ADR-09-protocols-invocation-surface.md) | protocols/ as the invocation surface | Accepted (2026-07-17) |
| [ADR-10](ADR-10-output-routing.md) | Output routing — local default + return-dir override | Accepted (2026-07-17) |
| [ADR-11](ADR-11-delegated-invocation-contract.md) | Delegated Invocation Contract — two lanes, one machine-readable surface | Accepted (2026-07-05) |
| [ADR-12](ADR-12-provider-backend-engine-and-cost-lanes.md) | Provider Backend Engine — CLI-subscription seats and two-lane cost policy | Accepted (2026-07-05; §5 default-flip evidence-gated) |
| [ADR-13](ADR-13-invocation-contract-versioning.md) | Invocation-contract versioning (ratifies DRAFT-INT-2) | Accepted (2026-07-18) |
| [ADR-14](ADR-14-adr-lifecycle-states.md) | ADR lifecycle states (ratifies DRAFT-GOV-1) | Accepted (2026-07-17) |

## Cross-repo references

- **ADR-43** (Cross-project transcript routing) — lives in `.dev-knowledge/docs/decisions/`. Amendments tracked there. ai-council implementation reflects amendment cycle 1 (schema: `dev_root` + `target_projects` list, not old `dict[name, path]`).
