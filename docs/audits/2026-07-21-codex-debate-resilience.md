# Codex Review — debate-resilience

**Date:** 2026-07-21
**Branch:** `worktree-fix-debate-resilience`
**HEAD:** `e3be11d`
**Diff range:** `38be0cc..HEAD`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

- P1-8 contract boundary: confirm the change stays strictly inside the contract-1.0 field set. No verdict-package field added, removed or renamed; contract_version stays "1.0"; exit_semantics stays 0. The new "lost" status is a VALUE-level correction to panel.dropped / degradation.failed_providers / degradation.degraded only.
- debate.py exception triage: confirm asyncio.gather(return_exceptions=True) plus the isinstance ladder preserves cancellation semantics. A BaseException that is not an Exception (CancelledError/KeyboardInterrupt/SystemExit) must be re-raised, never booked as a seat failure.
- P1-9 artifact safety: can the synthesis-failure path lose or double-write any artifact? Is re-raising the original exception AFTER the writes correct, and does skipping the verdict package + minority report leave any dangling reference?
- seat_router.py now catches Exception rather than ProviderError. Confirm this cannot swallow anything it should not, and that classify_cli_failure is safe on an arbitrary exception.
- General: any place where the new code could mask an original error, or where a comment claims a guarantee the code does not implement.

---

## Findings
## Critical

(none)

## High

### [HIGH] src/ai_council/orchestrator.py:178 — Unexpected synthesis exceptions still bypass artifact preservation

**What:** The preservation path catches only `ProviderError` and `RuntimeError`, although provider parsing and other synthesis code can raise `ValueError`, `TypeError`, `KeyError`, or `AttributeError`.

**Why:** Those exceptions still unwind before `save_to_file`, recreating P1-9’s loss of the paid-for debate.

**Fix direction:** Catch `Exception` at this preservation boundary while continuing to let `BaseException` cancellation/control-flow exceptions propagate.

### [HIGH] src/ai_council/orchestrator.py:319 — Routing failures mask the original synthesis exception

**What:** `raise_for_routing_failures()` executes before the saved `synthesis_error` is re-raised.

**Why:** If `--return-dir` fails, the CLI reports only `OutputRoutingError`; this contradicts the comment that the original synthesis cause remains accurate and unmasked.

**Fix direction:** When synthesis already failed, preserve it as the primary exception while attaching or logging routing failures; raise routing failures directly only on otherwise-successful runs.

### [HIGH] src/ai_council/output.py:437 — Failed transcripts contain a dangling verdict-package reference

**What:** `_build_header()` always emits the verdict summary, including “Dissent: unanimous” and “machine-readable fields are authoritative in the … JSON sibling,” even when synthesis failed and the orchestrator intentionally emits neither verdict package nor minority report.

**Why:** The preserved artifact falsely describes a verdict and points readers to a file guaranteed not to exist on this path.

**Fix direction:** Suppress verdict-only summary fields for failed-synthesis transcripts and render an explicit “no verdict produced” header instead.

### [HIGH] src/ai_council/synthesis.py:71 — Failure metrics fabricate zero synthesis usage and latency

**What:** The failure result always records a synthetic synthesis call with zero tokens, cost, and latency.

**Why:** Provider failures have measurable elapsed time, and the empty-content path already received a potentially billable `ModelResponse`; the sidecar therefore silently understates usage and contradicts the real total duration.

**Fix direction:** Preserve elapsed time and any available response usage, and represent genuinely unavailable metrics as unknown or omitted rather than observed zero.

## Medium

(none)

## Low

(none)

No Contract-1.0 verdict-field drift or cancellation-triage defect was found. The verdict payload retains `contract_version: "1.0"` and `exit_semantics: 0`; `BaseException` control-flow exceptions are re-raised; and `classify_cli_failure()` safely classifies ordinary arbitrary `Exception` instances. No double-write path was found.

Tests were not run because the repository’s reviewer policy permits read-only inspection only.
