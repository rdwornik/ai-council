# Codex Review — crux-check-18-pass2

**Date:** 2026-07-20
**Branch:** `feat/crux-check`
**HEAD:** `67880bb`
**Diff range:** `main..feat/crux-check`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

This is PASS 2. Pass 1 returned 1 Critical + 3 High; three were repaired. Verify the repairs and look for anything new.
- HIGH-1 repaired: _parse_crux now returns ParseState {CLAIM, NO_CRUX, MALFORMED}. Malformed/empty/refused/truncated -> RETRIEVAL_UNAVAILABLE, not a false no_empirical_crux. Negation prefixes narrowed so "There is no statistically significant difference between A and B" parses as a CLAIM. Confirm no remaining path reports an extraction failure as a valid no-crux success.
- HIGH-2 repaired: dead crux_check.max_tokens removed (AIProvider.generate has no per-call token override). Artifact now bounded by _MAX_ARTIFACT_CHARS (assembled) and _MAX_CLAIM_CHARS, not just body. Confirm the injected artifact is genuinely bounded.
- HIGH-3 repaired: debate.py except now records CruxArtifact(RETRIEVAL_UNAVAILABLE) instead of None, so a service crash is distinguishable from "feature disabled".
- CRITICAL was DOWNGRADED not fixed. Argument: CruxCheckService.check() receives only the anonymized str block, never list[ModelResponse], so it cannot learn which panelist authored which proposal. A vendor name appearing in retrieved research prose is topic content, not blind-voting attribution. ADR-03 governs attribution. Challenge this reasoning if it is wrong.
- Also re-check: contract_version stays "1.0", crux never reaches _build_verdict_payload, output.py byte-identical to main, and source-compat of run_debate / build_debate_metrics.

---

## Findings
## Critical

(none)

## High

### HIGH [src/ai_council/crux_check.py:108](C:/Users/1028120/Documents/Dev/ai-council/.claude/worktrees/t1-crux-check/src/ai_council/crux_check.py:108) — `NONE` prefix matching still swallows valid claims

**What:** `startswith("none")` classifies claims such as “Nonetheless, deployments fail more often” or “None of the benchmarks met the target” as `NO_CRUX`.  
**Why:** Retrieval is silently skipped and the run reports a valid no-crux success for an empirical claim, so HIGH-1 is not fully repaired.  
**Fix direction:** Match the `NONE` sentinel exactly and recognize only complete, explicit no-crux responses; add regressions for `Nonetheless` and `None of ...`.

### HIGH [src/ai_council/crux_check.py:112](C:/Users/1028120/Documents/Dev/ai-council/.claude/worktrees/t1-crux-check/src/ai_council/crux_check.py:112) — Headed refusals are accepted as claims

**What:** Any nonempty text below `## Crux` that misses the no-crux prefixes becomes `CLAIM`, including “I cannot determine a crux because the input is incomplete.”  
**Why:** An extraction failure can proceed through retrieval and be reported as `GROUNDED`, contrary to the repaired three-state contract.  
**Fix direction:** Require an explicit structured state or reject refusal/uncertainty forms as `MALFORMED`, with tests for refusals beneath a valid heading.

### HIGH [tests/test_output.py:1901](C:/Users/1028120/Documents/Dev/ai-council/.claude/worktrees/t1-crux-check/tests/test_output.py:1901) — Equality test includes independently generated timestamps

**What:** The test calls `_build_verdict_payload()` twice and compares entire payloads, but each call generates a fresh microsecond-resolution `timestamp`.  
**Why:** The assertion is nondeterministic and will normally fail despite crux having no effect on the payload.  
**Fix direction:** Freeze `_iso_now()`, normalize the timestamp before comparison, or compare all contract fields except `timestamp`.

## Medium

(none)

## Low

(none)

Confirmed separately: the assembled artifact has a fixed finite bound; the service-crash path records `RETRIEVAL_UNAVAILABLE`; the blind-voting downgrade is sound because no panel attribution mapping reaches the service; `output.py` is byte-identical to `main`; the verdict remains version `1.0` without crux fields; and both requested function signatures remain source-compatible.
