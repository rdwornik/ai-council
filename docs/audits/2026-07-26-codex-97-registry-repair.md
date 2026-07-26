# Codex Review — 97-registry-repair

**Date:** 2026-07-26
**Branch:** `fix/97-registry-repair`
**HEAD:** `5b0fe4f`
**Diff range:** `main..fix/97-registry-repair`
**Codex version:** codex-cli 0.145.0
**Mode:** diff-review

---

## Focus

- Is the external expected-set test genuinely external, or does it re-derive from vc.RULES anywhere?
- Can the computed coverage block in format_report ever overstate coverage, or miscount when results are partial?
- Is the rule-12 structural exemption asserted non-tautologically?
- Does registering rules 1 and 7 as stubs create any false impression that they check something?
- Any way the KNOWN LIMITATIONS disclosure is inaccurate about what is and is not detected?

---

## Findings
## Critical

(none)

## High

(none)

## Medium

(none)

## Low

(none)

The expected set is independently literal in the test and not derived from `vc.RULES`; coverage correctly reports omitted results as absent; rule 12 is externally constrained and disclosed; rules 1/7 explicitly report `SKIP` and “check nothing”; limitations disclose the manually maintained spec-set blind spot.
