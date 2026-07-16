# Codex Review — plan-of-record

**Date:** 2026-07-16
**Branch:** `docs/plan-of-record`
**HEAD:** `320bd70`
**Diff range:** `main..docs/plan-of-record`
**Codex version:** codex-cli 0.144.5
**Mode:** doc-review

---

## Focus

- Docs-only arc: intake Reading-Map pointer fix (b849653 relocation), BACKLOG reconciliation (#22-#31, new [E7]/[S10]-[S12], #16 pre-work clause), new docs/intake/2026-07-16-plan-of-record.md
- Check disposition-faithfulness: phase->task map vs the frozen phase plan; gates G1-G3 vs the intake doc; seam rules vs intake §3
- Check cross-doc consistency: new task refs (lane docs, fleet-recon F3/F12, refactoring-guide A/B items, CONTRACT §7) and dangling references
- BACKLOG structural integrity: ADR-66 story-map form, task-line grammar, depends-on #16

---

## Findings
Found 6 prose/structural issues.

```text
## CRITICAL

(none)

## HIGH

## HIGH BACKLOG.md:60 — CliProvider v1 scope contradicts ADR-12

What: The task still names Claude/Gemini/Codex backends and permits completion with “at least one” backend, while the added ADR-12 clause says v1 is Claude+Codex only.
Why: This leaves a ratified exclusion (Gemini) in scope and allows the phase-map promise of Claude+Codex to close only partially.
Fix direction: Replace the stale adapter list and make the done condition explicitly cover the ratified v1 set.

## HIGH docs/intake/2026-07-16-plan-of-record.md:49 — P4 doctor pre-work conflicts with the P6 placement of #22/#23

What: #25 requires refactoring-guide A2 before doctor, but A2’s prescribed steps implement both D2 parity fixes (#22 frontmatter parsing and #23 research return-dir); the map places those tasks in P6 after P4.
Why: Either P4 completes P6’s mapped work early, or doctor is built without its stated prerequisite sequence.
Fix direction: Move #22/#23 into or before P4, or explicitly split A2 so the doctor prerequisite excludes those closures.

## MEDIUM

## MEDIUM docs/intake/2026-07-16-plan-of-record.md:71 — #24 is said to supersede #1 without reconciling the dependency graph

What: The note says EPI-1 supersedes #1’s Gemini-only sample, but #1 retains that done condition and #2 remains BLOCKED on #1; #31 does not include this reconciliation.
Why: The roadmap retains two incompatible evidence paths, and #24 cannot formally unblock #2.
Fix direction: Update #1/#2 to make #24 the authoritative evidence task, or remove the supersession claim and retain both methods intentionally.

## MEDIUM BACKLOG.md:8 — Theme count is stale

What: The overview says the backlog has six themes, while the same document now defines [E1] through [E7].
Why: The top-level summary is internally inconsistent and misstates the ADR-66 story-map structure.
Fix direction: Change the summary to seven themes.

## MEDIUM docs/intake/2026-07-06-technical-architect-intake.md:30 — Reading Map still describes the old six-epic backlog

What: The Reading Map says BACKLOG is a “6-epic story map (Epics A–F),” repeated in the appendix at line 102, despite the new [E7].
Why: The entry-point document now directs readers to an obsolete structural description.
Fix direction: Update both references to the current seven-epic, named-theme story map.

## MEDIUM docs/intake/2026-07-16-plan-of-record.md:43 — P0 closure citation is missing

What: P0 is declared complete through “this doc, cited by its merge SHA,” but no merge SHA or location of that citation is provided.
Why: The stated closure rule is not independently verifiable, especially once this immutable record is merged.
Fix direction: Name the merge SHA and its citation location, or redefine P0’s closure event without requiring an absent citation.

## LOW

(none)
```
