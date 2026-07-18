# Codex Review — verdict-package-pass2

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — pass-2 fixes `caa395f` (strictly reductive, unit-tested); open: #33 (pass-3 terra WAIVED to 2026-07-23). _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-17
**Branch:** `feat/verdict-package`
**HEAD:** `7df9092`
**Diff range:** `main..feat/verdict-package`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

- Pass-2 VERIFICATION of #26 verdict package. Pass-1 (2026-07-17-codex-verdict-package.md) found 1 Critical + 4 High; verify each is resolved and check for any NEW issues introduced by the fixes:
- (was CRITICAL) seats[] redesign: verify panel.seats now uses ONE canonical serializer (_seat_payload) shared with _metrics.json — no parallel seat schema; degradation.fallback_events is a derived view.
- (was HIGH) judge decision: verify '## Overall Verdict' now beats '## Recommendations' in _DECISION_HEADING_MARKERS ordering.
- (was HIGH) minority pointer drift: verify save_minority_report shares the transcript stem via stem_base and the verdict points at the actual emitted minority filename.
- (was HIGH) silent return-dir failure: verify save_verdict_package raises OutputRoutingError when a required return_dir was not written.
- (was HIGH) manifest completeness: verify the verdict's own artifacts[] entry lists its destinations.
- Also check: no secret/credential leakage into the JSON; no regression in _write_routed/save_to_file behavior; correctness of the new _route_dirs prediction vs _write_routed.

---

## Findings
## CRITICAL

(none)

## HIGH

### HIGH — src/ai_council/output.py:687 — Minority pointer can still reference a nonexistent artifact

**What:** When `written["minority"]` is absent or empty, `save_verdict_package()` fabricates a filename even though it never emits that file.

**Why:** Direct callers can receive a successful verdict package whose `minority_artifact` pointer does not resolve; the fix only guarantees correctness through the orchestrator path.

**Fix direction:** Require an actual emitted minority path for non-unanimous verdicts, or use `null`/raise when none was supplied.

### HIGH — src/ai_council/output.py:784 — Verdict manifest records planned rather than successful destinations

**What:** `_route_dirs()` populates the verdict’s `paths` before `_write_routed()` attempts best-effort writes, so failed target or return routes remain listed.

**Why:** A target write can fail while the canonical package succeeds and falsely claims the missing copy exists.

**Fix direction:** Build the verdict entry from paths actually returned by routing, then finalize the manifest consistently in every successful copy.

## MEDIUM

(none)

## LOW

(none)

The shared `_seat_payload`, judge-heading ordering, orchestrator stem sharing, and required return-directory exception are otherwise correctly implemented. Static review only; tests were not run under the read-only reviewer contract.
