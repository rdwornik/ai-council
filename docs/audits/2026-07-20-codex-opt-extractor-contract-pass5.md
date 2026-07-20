# Codex Review — opt-extractor-contract-pass5

**Date:** 2026-07-20
**Branch:** `fix/opt-extractor-contract`
**HEAD:** `5072b8d`
**Diff range:** `main..fix/opt-extractor-contract`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

Pass 5, convergence check. Passes 1-4 each found 3-4 real HIGH, most recently a scanner HANG on a Windows path. All repaired; fuzz guards for termination and non-fabrication added.
Review the CURRENT state of _pair_code_spans, _unwrap_emphasis, _is_punctuation, _top_level_bullets.
- Report ONLY defects you can demonstrate with a concrete input string, giving actual vs expected output.
- Prioritise: (a) non-termination or superlinear runtime, (b) payload loss, (c) fabricated items. Cosmetic CommonMark divergences that neither lose payload nor fabricate items are NOT defects for this contract -- say so rather than reporting them.
- If you find nothing demonstrable in those three classes, say so plainly.

---

## Findings
```text
No non-termination or superlinear-runtime defect reproduced. Four HIGH data-integrity defects remain; cosmetic-only CommonMark divergences are omitted.

## CRITICAL

(none)

## HIGH

### HIGH src/ai_council/output.py:837 — Escaped backtick hides a following code fence

**What:** Input `"- \```__init__``\n"` returns `["```init``"]`; expected `["`__init__"]`. The escaped first backtick is literal, while the remaining two backticks delimit an atomic code span.
**Why:** `_pair_code_spans` merges the escaped backtick with the following unescaped run, exposing `__init__` to emphasis removal and losing payload.
**Fix direction:** Represent the escaped first backtick separately from the remaining run while preserving the rule that backslashes are inert inside an already-open code span.

### HIGH src/ai_council/output.py:1008 — Unicode line separator creates a false Markdown line

**What:** Input `"ordinary prose\u2028- fabricated option\n"` returns `["fabricated option"]`; expected `[]`.
**Why:** `str.splitlines()` treats U+2028 as a line boundary, although it is not a CommonMark line ending, fabricating an option from inline payload.
**Fix direction:** Split only on Markdown line endings: LF, CR, and CRLF.

### HIGH src/ai_council/output.py:1011 — Leading Unicode whitespace is stripped into a list marker

**What:** Input `"\u00A0- ordinary prose\n"` returns `["ordinary prose"]`; expected `[]`.
**Why:** `raw.strip()` removes the leading NBSP before marker matching, fabricating a top-level bullet where no marker exists at the start of the Markdown line.
**Fix direction:** Match list markers against the unstripped line; normalize only explicitly permitted characters after recognizing the marker.

### HIGH src/ai_council/output.py:1021 — Unicode whitespace turns option payload into a thematic break

**What:** Input `"- *\u00A0*\u00A0*\n"` returns `[]`; expected `["*\u00A0*\u00A0*"]`.
**Why:** `_THEMATIC_BREAK_RE` uses `\s*`, so NBSP-separated asterisks are misclassified as a thematic break and the complete option payload is lost.
**Fix direction:** Restrict thematic-break separators to ASCII spaces and tabs.

## MEDIUM

(none)

## LOW

(none)
```
