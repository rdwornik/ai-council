# Codex Review — opt-extractor-contract-pass3

**Date:** 2026-07-20
**Branch:** `fix/opt-extractor-contract`
**HEAD:** `bd29df3`
**Diff range:** `main..fix/opt-extractor-contract`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

Pass 3. Pass 1 found 3 HIGH (regex-based emphasis), pass 2 found 3 HIGH in the replacement scanner. Both repaired. Review the CURRENT state for convergence:
- _pair_code_spans: is the opener-first pairing with per-length cursors correct AND linear? Check the cursor advancement carefully for any case where a run is skipped or double-paired.
- _unwrap_emphasis: any remaining input that loses payload, fabricates an item, or runs superlinearly?
- _top_level_bullets: marker grammar and both thematic-break guards.
- Do the tests pin what their names and docstrings claim?
Report ONLY defects you can demonstrate with a concrete input. Do not restate prior findings that are now fixed.

---

## Findings
## Critical

(none)

## High

### [HIGH] [src/ai_council/output.py:827](C:/Users/1028120/Documents/Dev/ai-council/.claude/worktrees/fix-opt-extractor/src/ai_council/output.py:827) — Escaped backtick can steal the next code-span opener

**What:** For `r"- \` `__init__`\n"`, the escaped backtick is paired with the next backtick, so extraction returns ``"` `init`"`` instead of ``"` __init__"``.  
**Why:** The real code span is no longer atomic and its `__` payload is deleted as emphasis; escaped punctuation must not acquire Markdown meaning outside a code span. [CommonMark](https://spec.commonmark.org/current/)  
**Fix direction:** Prevent an escaped run from acting as an opener while retaining its eligibility as a closer inside an already-open code span; add this composed escape/code/emphasis case to the tests.

### [HIGH] [src/ai_council/output.py:897](C:/Users/1028120/Documents/Dev/ai-council/.claude/worktrees/fix-opt-extractor/src/ai_council/output.py:897) — Punctuation-flanked literal asterisks are deleted

**What:** `"- a*.*b\n"` produces `["a.b"]`, although `a*.*b` contains no valid emphasis.  
**Why:** `can_open` and `can_close` consider only whitespace, omitting Markdown’s punctuation-flanking rules and silently deleting literal payload. [CommonMark](https://spec.commonmark.org/current/)  
**Fix direction:** Implement left/right-flanking punctuation conditions for `*` and `_`, and add `a*.*b` as a negative emphasis test.

### [HIGH] [src/ai_council/output.py:776](C:/Users/1028120/Documents/Dev/ai-council/.claude/worktrees/fix-opt-extractor/src/ai_council/output.py:776) — Ordered-marker regex accepts non-Markdown markers

**What:** `"1234567890. ordinary prose\n"` produces `["ordinary prose"]`; `\d+` also accepts non-ASCII digits.  
**Why:** Markdown ordered-list markers contain 1–9 ASCII digits, so ordinary prose beginning with ten digits can fabricate an option. [CommonMark](https://spec.commonmark.org/current/)  
**Fix direction:** Use `[0-9]{1,9}` and add rejection tests for ten-digit and non-ASCII-digit prefixes.

## Medium

(none)

## Low

(none)
