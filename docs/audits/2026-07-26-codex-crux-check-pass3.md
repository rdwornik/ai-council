---
date: 2026-07-26
topic: crux-check-pass3
surface: src/ai_council/crux_check.py (313 lines) + tests/test_crux_check.py
reviewer: terra (gpt-5.6-terra), direct `codex exec -s read-only`
mode: adversarial pass 3 (non-diff bounded prompt — the arc is long merged, so the
  review-script diff lane would exit on its empty-diff guard; per #73's ruling)
discharges: "#83"
---

# Codex pass 3 — crux-check surface

Passes 1 and 2 ran on the [#18] crux-check arc and their findings were repaired. Pass 3
never ran (codex usage limit, date-gated to 2026-07-25), so the **pass-2 repairs had never
been independently reviewed**. This is that pass.

Repairs under review:
- (a) the `_NO_CRUX_EXACT` vs `_NO_CRUX_PREFIXES` split
- (b) `_REFUSAL_MARKERS` routed to `MALFORMED`
- (c) the timestamp-excluded verdict-payload equality test

## Findings

## Critical

(none)

## High

### [HIGH] `crux_check.py:137` — refusal substring matching rejects valid empirical claims

`any(marker in normalized ...)` does bare substring matching, so a claim containing
`"AI cannot"` matches the marker `"i cannot"`. A legitimate crux is silently classified
`MALFORMED` and reported as retrieval-unavailable.

**Fix direction:** match refusal language at response level with word boundaries, not as
arbitrary substrings.

### [HIGH] `crux_check.py:212` — extractor exceptions outside `ProviderError` escape `check()`

Only `ProviderError` is caught around `self._extractor.generate()`, while the method's own
docstring at `:203` states **"Never raises."** A timeout or provider implementation bug
breaks that contract.

**Fix direction:** degrade expected extractor-call failures to `RETRIEVAL_UNAVAILABLE`,
preserving cancellation semantics.

## Medium

(none)

## Low

(none)

## Verdict

**Does the surface pass a pass-3 adversarial review? No.** Repairs (a) and (c) hold — the
exact-sentinel split is sound for the demonstrated `NONE`-prefix cases and its tests are
non-tautological, and the timestamp-excluded payload comparison genuinely compares every
non-timestamp field. Repair (b) introduced a high-impact false-positive path, and the
surrounding extractor error handling does not meet the stated non-raising contract.

## First-hand verification (not accepted from the reviewer)

Both High findings were reproduced against source before filing:

- **H1 reproduces.** Executing the real `_REFUSAL_MARKERS` tuple against the claim
  *"Whether an AI cannot determine tumour grade from images is the crux"* matches **two**
  markers — `i cannot` (inside "AI cannot") and `cannot determine` — yielding `MALFORMED`.
- **H2 confirmed by reading.** `:203` docstrings "Never raises."; `:214` catches
  `ProviderError` alone.

Filed as **#119** (H1) and **#120** (H2). **#83 discharges** through its done-when's
"…or its findings are filed" clause, **not** because the surface came back clean — it did
not.
