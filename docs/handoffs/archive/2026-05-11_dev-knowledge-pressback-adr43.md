# Delivery feedback — Cross-project transcript routing

| Field | Value |
|---|---|
| From | `.dev-knowledge` (client / ecosystem strażnik) |
| To | `ai-council` (producer) |
| Original feature | Cross-project transcript routing (R1-R8) |
| Delivery report received | 2026-05-11 |
| Type | Press-back, single substantive point |
| Status | Delivery accepted, one press-back |

## TL;DR

Delivery accepted. Implementation quality high — 8/8 requirements met, 44 new tests, 0 critical/high Codex findings, hybrid mechanism (option 4 + 2) defensibly architected, `secondary_output_dir` legacy resolution clean.

**One press-back:** delivery report §6.3 classifies `.dev-knowledge`-side ADR creation as "optional but recommended." Client position: **not optional. ADR-43 is required.**

No code action requested in `ai-council`. This artifact is awareness + flag for future amendments.

## Press-back: §6.3 ADR-43 is required, not optional

Delivery report §6.3 rationale for "optional": "if you skip this ADR, the convention is still adequately documented in `ai-council/CLAUDE.md` and `README.md`, but ecosystem-level audit trail will be lighter."

Client side rejects "adequate" framing. Four reasons:

### 1. Cross-repo audit trail is the point

`ai-council/CLAUDE.md` is invisible from `.dev-knowledge`'s perspective. As ecosystem strażnik, `.dev-knowledge` cannot detect drift, contradictions, or future amendments to a decision documented only in another repo's tool-specific config. ADRs in `.dev-knowledge/docs/decisions/` are the cross-repo governance currency precisely because they sit in the meta-layer everyone reads.

### 2. Drift prevention — Item 0 just cleaned exactly this drift class

Item 0 work in `.dev-knowledge` on 2026-05-11 was substantially about cleaning up decisions that were made conversationally or in tool-specific docs, never ratified as ecosystem-level ADRs, becoming stale or contradicted by later work. Examples: ESSENTIALS aspirational dual-write claim, PLAYBOOK internal contradiction at "Council Debate Archival Protocol," decisions README with only 1 of 16 ADRs indexed.

Skipping ADR-43 reintroduces the same drift class on day one of a fresh feature.

### 3. Decision substance warrants ADR

Two-layer routing model; single `TargetResolver` pattern; fail-loud unknown-target semantics; `secondary_output_dir` legacy deprecation path; best-effort mirror semantics with canonical-first guarantee. Architectural, not implementation. Affects multiple modules, multiple modes, and future repos joining the ecosystem.

### 4. Delivery report itself acknowledges the audit gap

The cost is named in the report: "audit trail will be lighter." Lighter audit trail is the failure mode `.dev-knowledge` exists to prevent.

## What `.dev-knowledge` is doing

Creating `ADR-43_cross_project_transcript_routing.md` in `.dev-knowledge/docs/decisions/` as a `.dev-knowledge`-side action. Content outline already provided in delivery report §6.3. The delivery report itself will serve as the debate artifact.

No `ai-council` action required for ADR creation itself.

## What `ai-council` should flag back to `.dev-knowledge` in future

If future `ai-council` changes affect routing semantics — adding auto-detection, changing fail-loud behavior, deprecating `target_paths`, removing `secondary_output_dir` entirely — those changes amend ADR-43.

Flag back to `.dev-knowledge` browser chat at the design stage, not after merging. Cross-repo invariant: architectural decisions affecting `.dev-knowledge`-documented conventions trigger ADR amendment in `.dev-knowledge`.

## No other press-back points

| Aspect | Verdict |
|---|---|
| Hybrid mechanism (option 4 + 2) | Defensible. Single source of truth via shared `TargetResolver`. |
| `secondary_output_dir` resolution (option A) | Right balance. Legacy retained-off, default flipped, documented. |
| Skipping Council debate during implementation | Defensible per "Defer requires justification." |
| Verification anchors | Thorough and traceable. |
| 44 new tests across 6 modules | Solid coverage. |

End of artifact.
