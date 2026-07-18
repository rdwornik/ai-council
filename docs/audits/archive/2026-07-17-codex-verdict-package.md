# Codex Review — verdict-package

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — #26 verdict package shipped `fd40585`; terra fixes `7df9092`. No open remainder. [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-17
**Branch:** `feat/verdict-package`
**HEAD:** `4de0734`
**Diff range:** `main..feat/verdict-package`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

- #26 verdict package (DRAFT-INT-1). Key invariants to check:
- save_verdict_package must be a SIBLING of save_to_file; save_to_file must gain zero package lines (pure orchestration after A4).
- The deterministic <ts> for council-verdict must equal the transcript's <ts> (single _ts() source; derived from transcript stem).
- The package must CONSUME the seats[]/synthesis namespaces by reference only — it must not redesign them.
- Each heuristic-extracted field (decision/rationale/options/dissent) must carry a machine-readable source annotation ('extraction' vs 'record').
- contract_version=null (no invented version); exit_semantics=0 for a completed debate.
- Routing: verdict must reach every destination via _write_routed; behavior-preserving A4 decompose.
- Look for correctness bugs, seam violations, credential/secret leakage into the JSON, and any regression risk.

---

## Findings
## CRITICAL

### CRITICAL — src/ai_council/output.py:647 — `seats[]` namespace is redesigned

**What:** The verdict creates a second, incomplete `panel.seats` schema, omitting `cli`, `identity_channel`, and `fallback_events`, then relocates fallback data under `degradation`.

**Why:** This violates the explicit namespace seam and permits the verdict and canonical metrics `seats[]` schemas to drift.

**Fix direction:** Reference or reuse the canonical `seats[]` serialization; derive only the required panel summary without defining another seat shape.

## HIGH

### HIGH — src/ai_council/output.py:490 — Judge verdict extracts recommendations as the decision

**What:** `"recommendation"` is prioritized before `"overall verdict"` and substring-matches the judge template’s `## Recommendations` heading.

**Why:** Judge packages report the first recommended action instead of the actual `## Overall Verdict`, potentially producing an incorrect downstream ADR.

**Fix direction:** Use mode-specific exact heading precedence, ensuring `Overall Verdict` wins for judge mode.

### HIGH — src/ai_council/output.py:606 — Dissent pointer can reference a nonexistent minority artifact

**What:** `minority_artifact` is derived from the transcript stem, while `save_minority_report` independently obtains its timestamp at line 452.

**Why:** A one-second rollover makes the JSON pointer differ from the actual minority filename.

**Fix direction:** Populate the pointer from the emitted `minority_paths` record, or derive every run artifact from one shared transcript base.

### HIGH — src/ai_council/output.py:705 — Required return-dir verdict failures remain silent

**What:** `_write_routed` treats `return_dir` failures as best-effort, and `save_verdict_package` does not verify that the requested destination was written.

**Why:** A completed Lane A run can exit successfully without placing its required verdict package in the caller’s return directory.

**Fix direction:** Fail loudly when a requested `return_dir` is absent from the returned paths while retaining best-effort semantics for optional mirrors.

### HIGH — src/ai_council/output.py:625 — Verdict artifact omits its destinations

**What:** The verdict’s own `artifacts[]` entry contains only `kind` and `filename`, unlike other entries that include written paths.

**Why:** This violates the manifest requirement to list every artifact with destinations and leaves provenance incomplete.

**Fix direction:** Record the verdict’s actual successful routed paths in its artifact entry.

## MEDIUM

(none)

## LOW

(none)

Static review only: pytest collection could not run because the read-only environment provides no writable temporary directory.
