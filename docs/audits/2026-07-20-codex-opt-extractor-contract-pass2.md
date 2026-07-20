# Codex Review — opt-extractor-contract-pass2

**Date:** 2026-07-20
**Branch:** `fix/opt-extractor-contract`
**HEAD:** `0ecf045`
**Diff range:** `main..fix/opt-extractor-contract`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

This is pass 2. Pass 1 found 3 HIGH (code spans not atomic, escaped delimiters, quadratic backtracking, list-wrapped thematic breaks); all were repaired by replacing the emphasis regex with a linear scanner in _unwrap_emphasis. Review the REPAIR:
- Is the _unwrap_emphasis scanner correct? Check the backtick-fence matching loop carefully, the equal-length pairing rule, the opener-stack truncation, and the _MAX_EMPHASIS_NESTING cap.
- Can any input still lose payload characters, fabricate an item, or run superlinearly?
- Is the {items, source, heading} shape and _extracted_options signature still intact?
- Are the new tests actually pinning the behaviour they claim?

---

## Findings
The return shape `{items, source, heading}` and `_extracted_options(sections, question_sections=None)` signature remain intact.

## CRITICAL

(none)

## HIGH

### HIGH src/ai_council/output.py:831 — Longer backtick fences can be partially consumed

**What:** After rejecting the start of a longer backtick run, the search restarts inside that run and may accept its trailing suffix. For example, a one-backtick opener followed by a three-backtick run is incorrectly paired and mangled.

**Why:** This violates the equal-length and “unpaired delimiters remain verbatim” contracts, silently corrupting `options_considered`.

**Fix direction:** Scan maximal backtick runs and accept a closer only when the complete run length equals the opener length; add one-vs-three and two-vs-five fence tests.

### HIGH src/ai_council/output.py:829 — Backtick scanning remains superlinear

**What:** Every unmatched backtick run calls `text.find()` across the remaining suffix. Descending run lengths produce repeated suffix scans—Θ(k³) work for Θ(k²) input.

**Why:** A delimiter-heavy model response can still stall verdict generation despite the scanner’s linear-time contract.

**Fix direction:** Tokenize all maximal backtick runs in the initial forward pass and pair them without rescanning suffixes; add a descending-fence performance regression.

### HIGH src/ai_council/output.py:875 — Nesting cap is skipped for dual-purpose delimiters

**What:** When a delimiter can both close and open but finds no match, it is appended at line 876 without applying `_MAX_EMPHASIS_NESTING`. Repeated reverse searches over these uncapped openers are superlinear.

**Why:** The advertised 64-entry bound is ineffective on this path. The timing test at tests/test_output.py:1511 uses delimiters with `close=False`, so it does not exercise or pin this behavior.

**Fix direction:** Centralize opener insertion so every append applies the cap, and add a pathological test using unmatched delimiters flanked by non-whitespace on both sides.

## MEDIUM

(none)

## LOW

(none)
