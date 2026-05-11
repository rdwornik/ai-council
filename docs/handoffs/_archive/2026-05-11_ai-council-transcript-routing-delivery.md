---
type: delivery-report
date: 2026-05-11
from: ai-council (implementer)
to: .dev-knowledge (client / ecosystem strażnik)
feature: cross-project transcript routing
status: delivered, merged, awaiting .dev-knowledge side follow-up
---

# Delivery Report — Cross-project transcript routing

## TL;DR

Feature request for cross-project transcript routing (originally filed as `.dev-knowledge` BACKLOG Stream C P3 "Council CLI dual-write trigger logic", elevated to P1) is **delivered, merged, and verified** in `ai-council`. All 8 client requirements (R1-R8) met. All acceptance criteria checked. One architectural discovery during implementation required a small scope addition — resolved in line with operator's "deterministic over implicit" preference. `.dev-knowledge` has three small follow-ups to close the loop.

## 1. Requirements coverage (R1-R8)

| ID | Requirement | Status |
|---|---|---|
| R1 | CLI routes to target project, in addition to canonical | ✓ |
| R2 | Deterministic, no auto-detection | ✓ |
| R3 | YAML frontmatter `target-project:` (string or list) | ✓ |
| R4 | Config-driven path resolution, no hardcoded paths | ✓ |
| R5 | All 4 modes (pick / ideas / judge / research) | ✓ |
| R6 | Unknown target → fail loud, list known | ✓ |
| R7 | No retroactive migration of existing transcripts | ✓ |
| R8 | Existing `--output PATH` flag unchanged | ✓ |

## 2. Acceptance criteria

All 8 checked — frontmatter parsed, target_projects map present, dual-write working, unknown-key fails loud, all 4 modes, +44 unit tests, README updated, CHANGELOG entry.

## 3. Mechanism chosen (Council-debate deferral resolved)

Hybrid of option 4 (YAML frontmatter `target-project:`) and option 2 (`--target-project` CLI flag for direct mode where no frontmatter file exists, per R5). Both invocation paths feed the same `TargetResolver` instance — single resolution truth, no forked logic.

Council debate was not formally run. Operator accepted the hybrid directly per "Defer requires justification" — running a Council debate for a spec-preferred mechanism would have been deferral without information value. This delivery report and the `.dev-knowledge`-side press-back artifact serve as the debate artifact.

## 4. Architectural discovery — secondary_output_dir overlap

ESSENTIALS described dual-write as aspirational; partly true. The CLI did have a `secondary_output_dir` always-on mechanism, statically pointed to `.dev-knowledge` transcripts — a partial legacy implementation. After implementing `target_paths`, both mechanisms fired on `--target-project .dev-knowledge`.

Resolution (operator chose option A): `secondary_output_enabled` default flipped to `false`. Code path retained for explicit-enable backwards compatibility. Tests still green. `target_paths` is now the canonical routing mechanism.

## 5. Verification anchors

- ai-council main HEAD post-cleanup: 11 commits ahead of pre-feature state, not pushed
- Tests: 342 → 354 passing (+44 new), 6 deselected
- Codex review: 0 Critical / 0 High / 1 Medium / 2 Low — all addressed pre-merge

Key code locations:
- `src/ai_council/routing.py` — `TargetResolver` + `RoutingError`
- `src/ai_council/inbox.py` — frontmatter parser
- `src/ai_council/output.py` — canonical + best-effort mirror
- `src/ai_council/cli.py` — `--target-project` Click option
- `config/settings.yaml` — `target_projects` map; `secondary_output_enabled: false`

User-facing docs: `README.md` and `CLAUDE.md` "Transcript Routing" sections.

## 6. What `.dev-knowledge` needs to do

### 6.1 ESSENTIALS update — required

Replace "Council output convention" section in `protocols/ESSENTIALS.md`. Proposed replacement reflects per-invocation routing via `target-project` frontmatter or `--target-project` CLI flag; paths resolved from `target_projects` map; unknown target fails loud; no target = canonical only; legacy `secondary_output_dir` defaulted off.

### 6.2 BACKLOG cleanup — required

Remove or mark completed: "Stream C P3: Council CLI dual-write trigger logic" entry. Superseded by this work.

### 6.3 ADR — optional but recommended

Suggested title: `ADR-NN_cross_project_transcript_routing.md`. Content outline: Context (history of secondary_dir, manual archival, multi-project demand), Decision (two-layer model, single resolver, canonical-first + best-effort mirror, fail-loud unknown, secondary_dir deprecation), Consequences, Alternatives (5 options enumerated, hybrid 4+2 chosen).

If skipped, convention documented in ai-council CLAUDE.md + README, but ecosystem-level audit trail lighter.

## 7. Open follow-ups (not blocking)

- `git push` ai-council main when operator ready
- Additional `target_projects` entries as new repos join
- Optional: full deprecation of `secondary_output_dir` in future cycle

## 8. Routing of this report

Drop into a fresh `.dev-knowledge` browser chat for §6.1/6.2/6.3 execution.

End of report.
