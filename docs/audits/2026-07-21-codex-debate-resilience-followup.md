# Codex Review — debate-resilience-followup

**Date:** 2026-07-21
**Branch:** `worktree-fix-debate-resilience`
**HEAD:** `4badad0`
**Diff range:** `e3be11d..HEAD`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

This is a FOLLOW-UP review of fixes for four HIGH findings you raised in docs/audits/2026-07-21-codex-debate-resilience.md. Verify each is genuinely resolved and that the fixes introduce no new defect:
- H1: orchestrator.py preservation boundary now catches Exception (not just ProviderError/RuntimeError). Confirm BaseException still propagates and nothing that should escape is now swallowed.
- H2: raise_for_routing_failures no longer masks synthesis_error. Confirm routing failures are still surfaced (not silently dropped) and R4 fail-loud is preserved on the SUCCESS path.
- H3: output.py now renders a no-verdict block when synthesis failed, detected via synthesis_metrics.error_class != "none". Confirm this detection cannot false-positive on a successful run, and that no dangling pointer to a never-written council-verdict-*.json remains.
- H4: EmptySynthesisError carries the billed ModelResponse; real usage and latency are booked; unknown fields stay None. Confirm the sidecar no longer understates spend and that EmptySynthesisError subclassing RuntimeError preserves the documented Raises contract and existing callers.
Also confirm the contract-1.0 verdict field set is still untouched.

---

## Findings
```markdown
## Critical

(none)

## High

### [HIGH] src/ai_council/synthesis.py:84 — Arbitrary exception response can defeat artifact preservation

**What:** `getattr(error, "response", None)` treats any exception’s `response` attribute as a `ModelResponse` without validating its type.

**Why:** An unexpected SDK exception carrying an HTTP response causes `build_failed_synthesis_result()` to raise a new `AttributeError` before artifacts are written, masking the original error and recreating H1’s data-loss path.

**Fix direction:** Accept the response only from `EmptySynthesisError`, or require `isinstance(candidate, ModelResponse)`; otherwise preserve unknown metrics as `None`.

## Medium

(none)

## Low

(none)

## Follow-up verification

- H1: `Exception` is caught and `BaseException` still propagates, but the HIGH above leaves one preservation hole.
- H2: Resolved. Synthesis remains primary; routing failures are logged, and successful runs still raise the aggregate routing error.
- H3: Resolved. Successful synthesis always records `error_class="none"`; failed transcripts contain no verdict-package pointer.
- H4: The intended `EmptySynthesisError` path books its response usage and latency and remains compatible with `RuntimeError` callers.
- Contract 1.0: Verdict fields are unchanged; `contract_version` remains `"1.0"` and `exit_semantics` remains `0`.

Tests could not run because the read-only environment provides no writable temporary directory.
```
